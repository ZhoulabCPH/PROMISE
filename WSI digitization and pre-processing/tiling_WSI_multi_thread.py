#!/usr/bin/env python3
"""
Tile whole-slide images into 224 x 224 patches at 10x magnification.

Processing rule implemented by this script:
    1. Read each WSI at level 0.
    2. Compute the level-0 extraction window that corresponds to the requested
       target magnification, 10x by default.
    3. Downsample each extracted window to 224 x 224 pixels.
    4. Segment tissue with Otsu thresholding.
    5. Discard patches with less than 60% tissue area.

The default settings match the following methods description:
    "WSIs were downsampled to 10x magnification and tiled into 224 x 224-pixel
    patches. Tissue regions were segmented using Otsu thresholding, and patches
    with < 60% tissue area were discarded."

Example:
    python wsi_10x_otsu_tiler.py \
        --slide /path/to/WSIs \
        --out /path/to/patches \
        --px 224 \
        --target_mag 10 \
        --min_tissue_area 0.60
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import imageio.v2 as imageio
import numpy as np
import openslide as ops
import pandas as pd
import shapely.geometry as sg
from PIL import Image


Image.MAX_IMAGE_PIXELS = None

DEFAULT_NUM_THREADS = 6
DEFAULT_MPP = 0.2494
DEFAULT_BASE_MAGNIFICATION = 40.0
DEFAULT_TARGET_MAGNIFICATION = 10.0
DEFAULT_PATCH_SIZE_PX = 224
DEFAULT_MIN_TISSUE_FRACTION = 0.60

JSON_ANNOTATION_SCALE = 10
THUMBNAIL_TARGET_AREA = 4096 * 4096
MIN_EXPORTED_TILE_COUNT = 4
FILLED_TILE_FOLDER_THRESHOLD = 6

SUPPORTED_SLIDE_SUFFIXES = {
    ".svs",
    ".mrxs",
    ".ndpi",
    ".scn",
    ".tif",
    ".tiff",
    ".jpg",
    ".jpeg",
}


@dataclass
class AnnotationObject:
    """Container for one ROI polygon."""

    name: str
    coordinates: list[tuple[int, int]] = field(default_factory=list)

    def add_coord(self, coord: tuple[int, int]) -> None:
        """Append one coordinate to this annotation polygon."""
        self.coordinates.append(coord)

    def add_shape(self, shape: Sequence[Sequence[float]]) -> None:
        """Append all points from a JSON-style polygon."""
        for point in shape:
            self.add_coord((int(round(point[0])), int(round(point[1]))))


@dataclass(frozen=True)
class SlideRecord:
    """Metadata for one slide to process."""

    name: str
    path: Path
    file_type: str
    category: str = "None"


@dataclass(frozen=True)
class TileConfig:
    """User-facing tiling configuration."""

    size_px: int
    target_magnification: float
    target_mpp: float | None
    base_magnification: float
    stride_div: float
    min_tissue_fraction: float
    save_folder: Path
    skip_wsi_without_annotations: bool = False
    augment: bool = False
    num_threads: int = DEFAULT_NUM_THREADS


@dataclass(frozen=True)
class SlideProcessingResult:
    """Execution result for one slide."""

    slide_name: str
    status: str
    tile_count: int = 0
    message: str = ""


class JPGSlide:
    """OpenSlide-like wrapper for large JPEG inputs.

    JPEG images do not contain standard WSI metadata. The wrapper therefore uses
    a configurable default MPP and base magnification so that JPEGs can still be
    processed in the same code path as OpenSlide-supported WSIs.
    """

    def __init__(
        self,
        path: Path,
        mpp: float = DEFAULT_MPP,
        base_magnification: float = DEFAULT_BASE_MAGNIFICATION,
    ) -> None:
        self.loaded_image = imageio.imread(path)
        self.dimensions = (self.loaded_image.shape[1], self.loaded_image.shape[0])
        self.properties = {
            ops.PROPERTY_NAME_MPP_X: str(mpp),
            ops.PROPERTY_NAME_OBJECTIVE_POWER: str(base_magnification),
        }
        self.level_dimensions = [self.dimensions]
        self.level_count = 1

    def get_thumbnail(self, dimensions: tuple[int, int]) -> np.ndarray:
        """Return a resized RGB thumbnail."""
        return cv2.resize(self.loaded_image, dsize=dimensions, interpolation=cv2.INTER_CUBIC)

    def read_region(
        self,
        topleft: tuple[int, int],
        level: int,
        window: tuple[int, int],
    ) -> np.ndarray:
        """Return a region with an OpenSlide-compatible RGBA shape."""
        del level

        x0, y0 = topleft
        width, height = window
        region = self.loaded_image[y0:y0 + height, x0:x0 + width]

        if region.ndim == 2:
            region = np.stack([region] * 3, axis=-1)

        if region.shape[2] == 3:
            alpha = np.full(region.shape[:2] + (1,), 255, dtype=region.dtype)
            region = np.concatenate([region, alpha], axis=2)

        return region


class SlideReader:
    """Read one slide and lazily generate valid 10x tissue-rich tiles."""

    def __init__(self, slide_record: SlideRecord, export_folder: Path, logger: logging.Logger) -> None:
        self.slide_record = slide_record
        self.export_folder = export_folder
        self.logger = logger

        self.coord: list[list[int | bool]] = []
        self.annotations: list[AnnotationObject] = []
        self.ann_polys: list[sg.Polygon] = []
        self.tile_mask: np.ndarray | None = None
        self.tile_records: list[dict[str, int | float | str]] = []

        self.extract_px: int | None = None
        self.target_mpp_used: float | None = None
        self.shape: tuple[int, int] | None = None
        self.filter_dimensions: tuple[int, int] | None = None
        self.filter_magnification: float | None = None
        self.thumb_file: Path | None = None
        self.MPP: float | None = None
        self.base_magnification: float | None = None

        self.has_anno = True
        self.no_mpp = False
        self.not_able_to_load = False

        self.slide = self._open_slide()
        if self.slide is None:
            self.not_able_to_load = True
            return

        self._load_annotations()
        self._initialize_metadata()

    def _open_slide(self):
        """Open a WSI or a large JPEG slide."""
        file_type = self.slide_record.file_type.lower()
        slide_path = self.slide_record.path

        if file_type in {"svs", "mrxs", "ndpi", "scn", "tif", "tiff"}:
            try:
                return ops.OpenSlide(str(slide_path))
            except Exception as exc:
                self.logger.error("Unable to read slide %s: %s", slide_path, exc)
                return None

        if file_type in {"jpg", "jpeg"}:
            try:
                return JPGSlide(slide_path)
            except Exception as exc:
                self.logger.error("Unable to read JPEG slide %s: %s", slide_path, exc)
                return None

        self.logger.error("Unsupported slide type for %s", slide_path)
        return None

    def _load_annotations(self) -> None:
        """Load optional sibling CSV or JSON ROI annotations."""
        roi_path_csv = self.slide_record.path.with_suffix(".csv")
        roi_path_json = self.slide_record.path.with_suffix(".json")

        if roi_path_csv.exists() and roi_path_csv.stat().st_size > 0:
            self.load_csv_roi(roi_path_csv)
        elif roi_path_json.exists() and roi_path_json.stat().st_size > 0:
            self.load_json_roi(roi_path_json)
        else:
            self.has_anno = False

    def _initialize_metadata(self) -> None:
        """Read slide dimensions, MPP, objective power, and thumbnail metadata."""
        try:
            self.shape = tuple(self.slide.dimensions)
            self.filter_dimensions = tuple(self.slide.level_dimensions[-1])
            self.filter_magnification = self.filter_dimensions[0] / self.shape[0]
        except Exception as exc:
            self.logger.error("Unable to load dimensions for %s: %s", self.slide_record.path, exc)
            self.not_able_to_load = True
            return

        thumbs_path = self.export_folder / "thumbs"
        thumbs_path.mkdir(parents=True, exist_ok=True)

        try:
            y_x_ratio = self.shape[1] / self.shape[0]
            thumb_x = sqrt(THUMBNAIL_TARGET_AREA / y_x_ratio)
            thumb_y = thumb_x * y_x_ratio
            thumb = self.slide.get_thumbnail((int(thumb_x), int(thumb_y)))
            self.thumb_file = thumbs_path / f"{self.slide_record.name}_thumb.jpg"
            imageio.imwrite(self.thumb_file, self._to_rgb_array(thumb))
        except Exception as exc:
            self.logger.warning("Unable to save thumbnail for %s: %s", self.slide_record.path, exc)

        try:
            if ops.PROPERTY_NAME_MPP_X in self.slide.properties:
                self.MPP = float(self.slide.properties[ops.PROPERTY_NAME_MPP_X])
            elif "tiff.XResolution" in self.slide.properties:
                self.MPP = 1 / float(self.slide.properties["tiff.XResolution"]) * 10000
            else:
                self.no_mpp = True
                self.logger.error("MPP metadata is missing for %s", self.slide_record.path)
        except Exception as exc:
            self.no_mpp = True
            self.logger.error("Unable to parse MPP for %s: %s", self.slide_record.path, exc)

        try:
            if ops.PROPERTY_NAME_OBJECTIVE_POWER in self.slide.properties:
                self.base_magnification = float(self.slide.properties[ops.PROPERTY_NAME_OBJECTIVE_POWER])
        except Exception:
            self.base_magnification = None

    def loaded_correctly(self) -> bool:
        """Return True when the slide can be tiled."""
        return bool(self.shape) and not self.not_able_to_load and not self.no_mpp

    @staticmethod
    def _to_rgb_array(region) -> np.ndarray:
        """Convert a PIL image or NumPy array to an RGB NumPy array."""
        if isinstance(region, Image.Image):
            return np.asarray(region.convert("RGB"))

        array = np.asarray(region)
        if array.ndim == 2:
            return np.stack([array] * 3, axis=-1)

        if array.shape[2] == 4:
            return array[:, :, :3]

        if array.shape[2] == 1:
            return np.repeat(array, 3, axis=2)

        return array

    def _read_region_rgb(self, x: int, y: int, level: int, size: int) -> np.ndarray:
        """Read one square region and return it as RGB."""
        region = self.slide.read_region((x, y), level, (size, size))
        return self._to_rgb_array(region)

    @staticmethod
    def _compute_otsu_tissue_fraction(region: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute tissue area fraction using Otsu thresholding.

        Otsu thresholding is applied to a blurred grayscale image. Because the
        glass/background area is usually bright and tissue is usually darker, an
        inverse binary threshold is used: tissue becomes 1 and background becomes 0.
        """
        if region.ndim != 3:
            raise ValueError("Otsu tissue segmentation expects an RGB image.")

        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        _, tissue_mask = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        # Remove tiny isolated artifacts before measuring tissue area.
        kernel = np.ones((3, 3), dtype=np.uint8)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)

        tissue_fraction = float(np.count_nonzero(tissue_mask) / tissue_mask.size)
        return tissue_fraction, tissue_mask

    def _within_annotations(self, x: int, y: int) -> bool:
        """Return True if the tile center falls inside an ROI polygon."""
        if not self.ann_polys:
            return True

        if self.extract_px is None:
            raise RuntimeError("extract_px has not been initialized.")

        center = sg.Point(int(x + self.extract_px / 2), int(y + self.extract_px / 2))
        return any(poly.contains(center) for poly in self.ann_polys)

    @staticmethod
    def _tile_name(case_name: str, x_coord: int, y_coord: int) -> str:
        """Create a stable tile filename from level-0 coordinates."""
        return f"{case_name}_x{x_coord}_y{y_coord}.jpg"

    def _export_tile(
        self,
        region: np.ndarray,
        tiles_path: Path,
        case_name: str,
        x_coord: int,
        y_coord: int,
        export_unique_only: bool,
        unique_tile: bool,
    ) -> str | None:
        """Write one tile to disk and return its filename."""
        if export_unique_only and not unique_tile:
            return None

        tile_name = self._tile_name(case_name, x_coord, y_coord)
        imageio.imwrite(tiles_path / tile_name, region)
        return tile_name

    def _export_augmented_tiles(self, region: np.ndarray, tiles_path: Path, case_name: str, x: int, y: int) -> None:
        """Write seven rotated/flipped augmentations for one tile."""
        augmentations = [
            np.rot90(region),
            np.flipud(region),
            np.flipud(np.rot90(region)),
            np.fliplr(region),
            np.fliplr(np.rot90(region)),
            np.flipud(np.fliplr(region)),
            np.flipud(np.fliplr(np.rot90(region))),
        ]

        for idx, augmented in enumerate(augmentations, start=1):
            imageio.imwrite(tiles_path / f"{case_name}_x{x}_y{y}_aug{idx}.jpg", augmented)

    def _resolve_target_mpp(
        self,
        target_magnification: float,
        target_mpp: float | None,
        base_magnification: float,
    ) -> float:
        """Resolve the target microns-per-pixel value for the requested magnification."""
        if self.MPP is None:
            raise RuntimeError("Slide MPP is missing.")

        if target_mpp is not None:
            return target_mpp

        base_mag = self.base_magnification or base_magnification
        return self.MPP * (base_mag / target_magnification)

    def build_generator(
        self,
        size_px: int,
        target_magnification: float,
        target_mpp: float | None,
        base_magnification: float,
        stride_div: float,
        min_tissue_fraction: float,
        case_name: str,
        tiles_path: Path,
        category: str,
        export: bool = False,
        augment: bool = False,
    ):
        """Build a generator that yields valid 10x Otsu-filtered tiles."""
        del category

        if not self.loaded_correctly():
            def empty_generator() -> Iterator[tuple[np.ndarray, int, bool]]:
                if False:
                    yield np.empty((0, 0, 3)), 0, False

            return empty_generator, 0, 0, 0

        if self.MPP is None or self.shape is None:
            raise RuntimeError("Slide metadata was not initialized correctly.")

        self.target_mpp_used = self._resolve_target_mpp(
            target_magnification=target_magnification,
            target_mpp=target_mpp,
            base_magnification=base_magnification,
        )

        # This is the key conversion for "downsampled to 10x and tiled into
        # 224 x 224 patches". For a 40x scan at 0.25 MPP, a 10x 224-pixel tile
        # covers about 896 level-0 pixels, then it is resized to 224 x 224.
        self.extract_px = max(int(round(size_px * self.target_mpp_used / self.MPP)), 1)
        stride = max(int(round(self.extract_px * stride_div)), 1)

        slide_x_size = max(self.shape[0] - self.extract_px, 0)
        slide_y_size = max(self.shape[1] - self.extract_px, 0)

        self.coord = []
        self.tile_records = []

        if self.shape[0] < self.extract_px or self.shape[1] < self.extract_px:
            self.logger.warning(
                "Slide %s is smaller than the requested extraction window (%d px).",
                self.slide_record.path,
                self.extract_px,
            )

        for y in range(0, max((self.shape[1] + 1) - self.extract_px, 0), stride):
            for x in range(0, max((self.shape[0] + 1) - self.extract_px, 0), stride):
                is_unique = (y % self.extract_px == 0) and (x % self.extract_px == 0)
                self.coord.append([x, y, is_unique])

        self.ann_polys = [
            sg.Polygon(annotation.coordinates)
            for annotation in self.annotations
            if len(annotation.coordinates) >= 3
        ]
        tile_mask = np.zeros(len(self.coord), dtype=np.uint8)

        def generator() -> Iterator[tuple[np.ndarray, int, bool]]:
            for coord_index, coord_item in enumerate(self.coord):
                x_coord, y_coord, unique_tile = coord_item

                if not self._within_annotations(x_coord, y_coord):
                    continue

                try:
                    # Read the level-0 region corresponding to one 224 x 224
                    # tile at the target magnification, then downsample it.
                    region = self._read_region_rgb(x_coord, y_coord, 0, self.extract_px)
                    region = cv2.resize(region, dsize=(size_px, size_px), interpolation=cv2.INTER_AREA)
                except Exception as exc:
                    self.logger.warning(
                        "Unable to read tile for %s at (%d, %d): %s",
                        self.slide_record.name,
                        x_coord,
                        y_coord,
                        exc,
                    )
                    continue

                tissue_fraction, _tissue_mask = self._compute_otsu_tissue_fraction(region)
                if tissue_fraction < min_tissue_fraction:
                    continue

                tile_mask[coord_index] = 1
                export_unique_only = stride_div == 1
                tile_name = self._tile_name(case_name, x_coord, y_coord)

                if export:
                    written_tile_name = self._export_tile(
                        region=region,
                        tiles_path=tiles_path,
                        case_name=case_name,
                        x_coord=x_coord,
                        y_coord=y_coord,
                        export_unique_only=export_unique_only,
                        unique_tile=bool(unique_tile),
                    )
                    if written_tile_name is None:
                        continue
                    tile_name = written_tile_name

                    if augment and (not export_unique_only or unique_tile):
                        self._export_augmented_tiles(region, tiles_path, case_name, x_coord, y_coord)

                self.tile_records.append(
                    {
                        "tile_name": tile_name,
                        "x_level0": int(x_coord),
                        "y_level0": int(y_coord),
                        "read_size_level0_px": int(self.extract_px),
                        "output_size_px": int(size_px),
                        "target_magnification": float(target_magnification),
                        "target_mpp": float(self.target_mpp_used),
                        "level0_mpp": float(self.MPP),
                        "tissue_fraction": float(tissue_fraction),
                    }
                )

                yield region, coord_index, bool(unique_tile)

            self.tile_mask = tile_mask

        return generator, slide_x_size, slide_y_size, stride

    def write_tile_metadata(self, tiles_path: Path) -> None:
        """Write per-tile coordinates and Otsu tissue fractions."""
        if not self.tile_records:
            return

        metadata_path = tiles_path / "tile_metadata.csv"
        pd.DataFrame(self.tile_records).to_csv(metadata_path, index=False)

    def load_csv_roi(self, path: Path) -> None:
        """Load ROI polygons from a CSV file with X_base and Y_base columns."""
        reader = pd.read_csv(path)
        headers = [str(col).strip() for col in reader.columns]
        if "X_base" not in headers or "Y_base" not in headers:
            raise IndexError('Unable to find "X_base" and "Y_base" columns in CSV file.')

        index_x = headers.index("X_base")
        index_y = headers.index("Y_base")
        self.annotations.append(AnnotationObject(f"Object{len(self.annotations)}"))

        for _, row in reader.iterrows():
            x_value = str(row.iloc[index_x]).strip()
            y_value = str(row.iloc[index_y]).strip()

            if x_value == "X_base" or y_value == "Y_base":
                self.annotations.append(AnnotationObject(f"Object{len(self.annotations)}"))
                continue

            if not x_value or not y_value or x_value.lower() == "nan" or y_value.lower() == "nan":
                continue

            x_coord = int(float(x_value))
            y_coord = int(float(y_value))
            self.annotations[-1].add_coord((x_coord, y_coord))

    def load_json_roi(self, path: Path) -> None:
        """Load ROI polygons from a LabelMe-like JSON file."""
        with path.open("r", encoding="utf-8") as json_file:
            json_data = json.load(json_file).get("shapes", [])

        for shape in json_data:
            points = shape.get("points", [])
            if len(points) < 3:
                continue

            area_scaled = np.multiply(points, JSON_ANNOTATION_SCALE)
            annotation_name = shape.get("label", f"Object{len(self.annotations)}")
            annotation = AnnotationObject(annotation_name)
            annotation.add_shape(area_scaled)
            self.annotations.append(annotation)


class Convoluter:
    """Coordinate multi-slide tiling with multi-threading."""

    def __init__(self, config: TileConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.SLIDES: dict[str, SlideRecord] = {}

    def load_slides(self, slides_array: Sequence[str | Path], category: str = "None") -> dict[str, SlideRecord]:
        """Load slide records and reject duplicate basenames."""
        duplicates: dict[str, list[str]] = {}
        slides: dict[str, SlideRecord] = {}

        for slide in slides_array:
            slide_path = Path(slide).resolve()
            slide_name = slide_path.stem
            duplicates.setdefault(slide_name, []).append(str(slide_path))
            slides[slide_name] = SlideRecord(
                name=slide_name,
                path=slide_path,
                file_type=slide_path.suffix.lstrip("."),
                category=category,
            )

        repeated_names = {name: paths for name, paths in duplicates.items() if len(paths) > 1}
        if repeated_names:
            duplicate_message = "; ".join(f"{name}: {paths}" for name, paths in repeated_names.items())
            raise ValueError(f"Duplicate slide basenames detected. Resolve naming conflicts first. {duplicate_message}")

        self.SLIDES = dict(sorted(slides.items()))
        self.logger.info("Total slides queued for tiling: %d", len(self.SLIDES))
        return self.SLIDES

    def convolute_slides(self) -> list[SlideProcessingResult]:
        """Run tiling for all queued slides."""
        blocks_root = self.config.save_folder / "BLOCKS"
        blocks_root.mkdir(parents=True, exist_ok=True)

        results: list[SlideProcessingResult] = []
        with ThreadPoolExecutor(max_workers=self.config.num_threads) as executor:
            futures = {
                executor.submit(self.export_tiles, slide): slide_name
                for slide_name, slide in self.SLIDES.items()
            }
            for future in as_completed(futures):
                slide_name = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    self.logger.exception("Unhandled error while processing %s: %s", slide_name, exc)
                    result = SlideProcessingResult(
                        slide_name=slide_name,
                        status="failed",
                        message="unhandled worker exception",
                    )

                results.append(result)
                self.logger.info(
                    "[%s] %s | tiles=%d%s",
                    result.status,
                    result.slide_name,
                    result.tile_count,
                    f" | {result.message}" if result.message else "",
                )

        return sorted(results, key=lambda result: result.slide_name)

    def export_tiles(self, slide: SlideRecord) -> SlideProcessingResult:
        """Export all valid tiles for one slide."""
        whole_slide = SlideReader(slide_record=slide, export_folder=self.config.save_folder, logger=self.logger)

        if whole_slide.not_able_to_load:
            return SlideProcessingResult(slide_name=slide.name, status="failed", message="slide could not be opened")

        if whole_slide.no_mpp:
            return SlideProcessingResult(slide_name=slide.name, status="failed", message="MPP metadata missing")

        if not whole_slide.has_anno and self.config.skip_wsi_without_annotations:
            return SlideProcessingResult(slide_name=slide.name, status="skipped", message="no annotation file found")

        tiles_path = self.config.save_folder / "BLOCKS" / slide.name
        tiles_path.mkdir(parents=True, exist_ok=True)

        existing_tile_count = len([path for path in tiles_path.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if existing_tile_count > FILLED_TILE_FOLDER_THRESHOLD:
            return SlideProcessingResult(
                slide_name=slide.name,
                status="skipped",
                tile_count=existing_tile_count,
                message="output folder already populated",
            )

        generator, _, _, _ = whole_slide.build_generator(
            size_px=self.config.size_px,
            target_magnification=self.config.target_magnification,
            target_mpp=self.config.target_mpp,
            base_magnification=self.config.base_magnification,
            stride_div=self.config.stride_div,
            min_tissue_fraction=self.config.min_tissue_fraction,
            case_name=slide.name,
            tiles_path=tiles_path,
            category=slide.category,
            export=True,
            augment=self.config.augment,
        )

        tile_count = 0
        for _tile, _coord, _unique in generator():
            tile_count += 1

        whole_slide.write_tile_metadata(tiles_path)

        if tile_count < MIN_EXPORTED_TILE_COUNT:
            self.logger.warning(
                "Very few valid tiles were extracted for %s (%d tiles).",
                slide.path,
                tile_count,
            )

        return SlideProcessingResult(slide_name=slide.name, status="completed", tile_count=tile_count)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate 10x 224 x 224 WSI patches using Otsu tissue filtering."
    )
    parser.add_argument(
        "-s",
        "--slide",
        default="path/to/WSIs",
        help="Path to a slide file or a directory containing slides.",
    )
    parser.add_argument(
        "-o",
        "--out",
        default="path/to/patches",
        help="Directory used to save extracted patches and run logs.",
    )
    parser.add_argument(
        "--skipws",
        action="store_true",
        help="Skip slides that do not have a sibling ROI annotation file (.csv or .json).",
    )
    parser.add_argument(
        "--px",
        type=int,
        default=DEFAULT_PATCH_SIZE_PX,
        help="Output patch size in pixels. Default: 224.",
    )
    parser.add_argument(
        "--target_mag",
        type=float,
        default=DEFAULT_TARGET_MAGNIFICATION,
        help="Target magnification used for tiling. Default: 10.",
    )
    parser.add_argument(
        "--target_mpp",
        type=float,
        default=None,
        help=(
            "Optional target microns-per-pixel. If provided, this overrides "
            "--target_mag when computing the level-0 extraction window."
        ),
    )
    parser.add_argument(
        "--base_mag",
        type=float,
        default=DEFAULT_BASE_MAGNIFICATION,
        help=(
            "Fallback objective magnification used when slide metadata does not "
            "contain objective-power information. Default: 40."
        ),
    )
    parser.add_argument(
        "--min_tissue_area",
        type=float,
        default=DEFAULT_MIN_TISSUE_FRACTION,
        help="Minimum Otsu tissue area fraction required to keep a patch. Default: 0.60.",
    )
    parser.add_argument(
        "--ov",
        type=float,
        default=1.0,
        help="Stride ratio relative to the extraction window. 1.0 means no overlap; 0.5 means 50%% overlap.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Export additional flipped/rotated copies for each saved tile.",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=DEFAULT_NUM_THREADS,
        help="Number of worker threads for multi-slide tiling.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""
    if args.px <= 0:
        raise ValueError("--px must be a positive integer.")

    if args.target_mag <= 0:
        raise ValueError("--target_mag must be a positive number.")

    if args.target_mpp is not None and args.target_mpp <= 0:
        raise ValueError("--target_mpp must be a positive number when provided.")

    if args.base_mag <= 0:
        raise ValueError("--base_mag must be a positive number.")

    if not (0 < args.min_tissue_area <= 1):
        raise ValueError("--min_tissue_area must be in the interval (0, 1].")

    if not (0 < args.ov <= 1):
        raise ValueError("--ov must be in the interval (0, 1].")

    if args.num_threads <= 0:
        raise ValueError("--num_threads must be a positive integer.")


def setup_logger(output_dir: Path) -> logging.Logger:
    """Create a file and console logger."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "report.txt"

    logger = logging.getLogger("tiling_wsi")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def discover_slides(slide_path: Path) -> list[Path]:
    """Discover one slide file or all supported slide files under a directory."""
    slide_path = slide_path.resolve()

    if slide_path.is_file():
        if slide_path.suffix.lower() not in SUPPORTED_SLIDE_SUFFIXES:
            raise ValueError(f"Unsupported input slide type: {slide_path.suffix}")
        return [slide_path]

    if not slide_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {slide_path}")

    return sorted(
        path
        for path in slide_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SLIDE_SUFFIXES
    )


def filter_preprocessed_slides(slide_list: Sequence[Path], output_dir: Path) -> list[Path]:
    """Skip slides whose output tile folder already contains enough images."""
    blocks_root = output_dir / "BLOCKS"
    if not blocks_root.exists():
        return list(slide_list)

    filtered: list[Path] = []
    for slide_path in slide_list:
        slide_output_dir = blocks_root / slide_path.stem
        if slide_output_dir.exists():
            image_count = len(
                [
                    path
                    for path in slide_output_dir.iterdir()
                    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
                ]
            )
            if image_count > FILLED_TILE_FOLDER_THRESHOLD:
                continue
        filtered.append(slide_path)

    return filtered


def log_run_configuration(logger: logging.Logger, args: argparse.Namespace) -> None:
    """Log the reproducible tiling configuration."""
    logger.info("Tiling configuration")
    logger.info("Input path: %s", Path(args.slide).resolve())
    logger.info("Output path: %s", Path(args.out).resolve())
    logger.info("Output patch size (px): %d", args.px)
    logger.info("Target magnification: %.4f", args.target_mag)
    logger.info("Target MPP override: %s", args.target_mpp)
    logger.info("Fallback base magnification: %.4f", args.base_mag)
    logger.info("Minimum Otsu tissue area fraction: %.4f", args.min_tissue_area)
    logger.info("Stride ratio: %.4f", args.ov)
    logger.info("Skip slides without annotation: %s", args.skipws)
    logger.info("Augmentation enabled: %s", args.augment)
    logger.info("Worker threads: %d", args.num_threads)


def summarize_results(logger: logging.Logger, results: Sequence[SlideProcessingResult]) -> None:
    """Log a compact run summary."""
    completed = sum(result.status == "completed" for result in results)
    skipped = sum(result.status == "skipped" for result in results)
    failed = sum(result.status == "failed" for result in results)
    total_tiles = sum(result.tile_count for result in results if result.status == "completed")

    logger.info("Run summary")
    logger.info("Completed slides: %d", completed)
    logger.info("Skipped slides: %d", skipped)
    logger.info("Failed slides: %d", failed)
    logger.info("Extracted tiles: %d", total_tiles)


def main() -> None:
    """Program entry point."""
    args = parse_args()
    validate_args(args)

    slide_path = Path(args.slide)
    output_dir = Path(args.out)
    logger = setup_logger(output_dir)
    log_run_configuration(logger, args)

    slide_list = discover_slides(slide_path)
    slide_list = filter_preprocessed_slides(slide_list, output_dir)

    if not slide_list:
        logger.warning("No slides need processing.")
        return

    config = TileConfig(
        size_px=args.px,
        target_magnification=args.target_mag,
        target_mpp=args.target_mpp,
        base_magnification=args.base_mag,
        stride_div=args.ov,
        min_tissue_fraction=args.min_tissue_area,
        save_folder=output_dir.resolve(),
        skip_wsi_without_annotations=args.skipws,
        augment=args.augment,
        num_threads=args.num_threads,
    )

    convoluter = Convoluter(config=config, logger=logger)
    convoluter.load_slides(slide_list)
    results = convoluter.convolute_slides()
    summarize_results(logger, results)


if __name__ == "__main__":
    main()
