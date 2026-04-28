#!/usr/bin/env python3
"""
Extract patch-level features for each whole-slide image folder.

This script supports multiple pathology foundation models and writes one Feather
file per slide. Each output file uses patch image names as the index and feature
dimensions as columns.

Example:
    python optimized_patch_feature_extraction.py \
        --model UNI_v2 \
        --patches_dir /path/to/patches \
        --output_dir /path/to/slides_features \
        --batch_size 64 \
        --device auto
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Iterable


SUPPORTED_MODELS = ("CONCH", "CONCH_v1_5", "UNI", "UNI_v2", "Virchow2")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# Human-friendly package names used in dependency error messages.
BASE_DEPENDENCIES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "torch": "torch",
    "PIL": "Pillow",
    "huggingface_hub": "huggingface_hub",
}

MODEL_DEPENDENCIES = {
    "CONCH": {"conch": "conch"},
    "CONCH_v1_5": {"transformers": "transformers"},
    "UNI": {"timm": "timm"},
    "UNI_v2": {"timm": "timm"},
    "Virchow2": {"timm": "timm"},
}


class PatchDataset:
    """A minimal PyTorch-compatible dataset for loading patch images.

    The class does not inherit from torch.utils.data.Dataset on purpose, so the
    script can show dependency errors cleanly before importing torch.
    """

    def __init__(self, slide_dir: Path, patch_paths: list[Path], preprocess: Callable[[Any], Any]):
        self.slide_dir = slide_dir
        self.patch_paths = patch_paths
        self.preprocess = preprocess

    def __len__(self) -> int:
        """Return the number of patch images in the current slide."""
        return len(self.patch_paths)

    def __getitem__(self, idx: int) -> tuple[str, Any]:
        """Load one image, convert it to RGB, and apply model preprocessing."""
        patch_path = self.patch_paths[idx]

        try:
            from PIL import Image

            with Image.open(patch_path) as image:
                image = image.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to read image: {patch_path}") from exc

        # Use a relative path as the patch id. This avoids collisions when
        # --recursive is enabled and patch files are stored in nested folders.
        patch_id = patch_path.relative_to(self.slide_dir).as_posix()
        return patch_id, self.preprocess(image)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract patch features for each slide folder using a selected "
            "pathology foundation model."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="CONCH",
        choices=SUPPORTED_MODELS,
        help="Foundation model used for patch feature extraction.",
    )
    parser.add_argument(
        "--patches_dir",
        type=str,
        required=True,
        help=(
            "Root directory containing slide folders. If this directory directly "
            "contains images, it is treated as a single slide."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=(
            "Root directory for output feature files. Files are written to "
            "<output_dir>/<model>/<slide>.feather."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size used for patch feature extraction. Must be positive.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers. Use 0 on Windows if multiprocessing is unstable.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device, for example: auto, cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help=(
            "Optional Hugging Face access token. If omitted, the script checks "
            "HF_TOKEN, HUGGINGFACE_HUB_TOKEN, HF_AUTH_TOKEN, and local login cache."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Feather files.",
    )
    parser.add_argument(
        "--disable_amp",
        action="store_true",
        help="Disable automatic mixed precision on CUDA.",
    )
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="float16",
        choices=("float16", "bfloat16"),
        help="Automatic mixed precision dtype used on CUDA.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively collect patch images inside each slide folder.",
    )
    return parser.parse_args()


def require_dependencies(model_name: str) -> None:
    """Fail early with a clear message if required Python packages are missing."""
    required = dict(BASE_DEPENDENCIES)
    required.update(MODEL_DEPENDENCIES[model_name])

    missing = [
        install_name
        for module_name, install_name in required.items()
        if importlib.util.find_spec(module_name) is None
    ]

    if missing:
        packages = " ".join(sorted(set(missing)))
        raise SystemExit(
            "Missing required dependencies.\n"
            f"Install them first, for example:\n\n    pip install {packages}\n"
        )


def validate_args(args: argparse.Namespace) -> None:
    """Validate arguments that argparse cannot fully validate."""
    if args.batch_size <= 0:
        raise SystemExit("--batch_size must be a positive integer.")

    if args.num_workers < 0:
        raise SystemExit("--num_workers must be 0 or a positive integer.")

    patches_dir = Path(args.patches_dir)
    if not patches_dir.exists():
        raise SystemExit(f"--patches_dir does not exist: {patches_dir}")

    if not patches_dir.is_dir():
        raise SystemExit(f"--patches_dir must be a directory: {patches_dir}")


def resolve_device(device_arg: str):
    """Resolve the requested inference device."""
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is False.")

    return device


def resolve_hf_token(cli_token: str | None) -> str | None:
    """Resolve a Hugging Face token from CLI input or common environment variables."""
    if cli_token:
        return cli_token

    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_AUTH_TOKEN"):
        token = os.getenv(env_name)
        if token:
            return token

    return None


def maybe_hf_login(hf_token: str | None) -> None:
    """Log in to Hugging Face only when a token is explicitly available."""
    if not hf_token:
        return

    from huggingface_hub import login

    login(token=hf_token, add_to_git_credential=False)


def load_conch(hf_token: str | None):
    """Load the CONCH model and preprocessing transform."""
    from conch.open_clip_custom import create_model_from_pretrained

    kwargs: dict[str, Any] = {}
    if hf_token:
        kwargs["hf_auth_token"] = hf_token

    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16",
        "hf_hub:MahmoodLab/conch",
        **kwargs,
    )
    return model, preprocess


def load_conch_v1_5(hf_token: str | None):
    """Load CONCH v1.5 from the MahmoodLab TITAN repository."""
    from transformers import AutoModel

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if hf_token:
        kwargs["token"] = hf_token

    titan = AutoModel.from_pretrained("MahmoodLab/TITAN", **kwargs)
    model, preprocess = titan.return_conch()
    return model, preprocess


def load_uni(hf_token: str | None):
    """Load the UNI model and preprocessing transform."""
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    model = timm.create_model(
        "hf-hub:MahmoodLab/UNI",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True,
    )
    preprocess = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, preprocess


def load_uni_v2(hf_token: str | None):
    """Load the UNI v2 model and preprocessing transform."""
    import torch
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from timm.layers import SwiGLUPacked

    model = timm.create_model(
        "hf-hub:MahmoodLab/UNI2-h",
        pretrained=True,
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        init_values=1e-5,
        embed_dim=1536,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=True,
    )
    preprocess = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, preprocess


def load_virchow2(hf_token: str | None):
    """Load the Virchow2 model and preprocessing transform."""
    import torch
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from timm.layers import SwiGLUPacked

    model = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    preprocess = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, preprocess


def load_model_and_preprocess(model_name: str, hf_token: str | None):
    """Dispatch model loading based on the selected model name."""
    loader_map = {
        "CONCH": load_conch,
        "CONCH_v1_5": load_conch_v1_5,
        "UNI": load_uni,
        "UNI_v2": load_uni_v2,
        "Virchow2": load_virchow2,
    }
    return loader_map[model_name](hf_token)


def call_encode_image(model: Any, batch_images: Any) -> Any:
    """Call encode_image while handling small API differences across models."""
    try:
        return model.encode_image(batch_images, proj_contrast=False, normalize=False)
    except TypeError:
        return model.encode_image(batch_images)


def first_available_output(outputs: Any, preferred_keys: Iterable[str]) -> Any:
    """Select a tensor-like output from common model output formats."""
    if isinstance(outputs, dict):
        for key in preferred_keys:
            if key in outputs:
                return outputs[key]

        if outputs:
            return next(iter(outputs.values()))

        raise RuntimeError("Model returned an empty dictionary.")

    if isinstance(outputs, (tuple, list)):
        if not outputs:
            raise RuntimeError("Model returned an empty tuple/list.")
        return outputs[0]

    return outputs


def ensure_2d_features(features: Any, model_name: str) -> Any:
    """Ensure the extracted features have shape [batch_size, feature_dim]."""
    if not hasattr(features, "ndim"):
        raise RuntimeError(f"{model_name} did not return a tensor-like feature output.")

    # Some models return token sequences with shape [B, N, C]. For generic models,
    # use the CLS token as the image-level representation.
    if features.ndim == 3:
        features = features[:, 0]

    if features.ndim != 2:
        raise RuntimeError(
            f"{model_name} features must have shape [B, C], but got {tuple(features.shape)}."
        )

    return features


def extract_conch_features(model: Any, batch_images: Any) -> Any:
    """Extract CONCH features."""
    return ensure_2d_features(call_encode_image(model, batch_images), "CONCH")


def extract_conch_v1_5_features(model: Any, batch_images: Any) -> Any:
    """Extract CONCH v1.5 features from encode_image or forward outputs."""
    if hasattr(model, "encode_image"):
        return ensure_2d_features(call_encode_image(model, batch_images), "CONCH_v1_5")

    outputs = model(batch_images)
    outputs = first_available_output(
        outputs,
        preferred_keys=("image_features", "features", "pooler_output", "last_hidden_state"),
    )
    return ensure_2d_features(outputs, "CONCH_v1_5")


def extract_forward_features(model: Any, batch_images: Any) -> Any:
    """Extract features from models whose forward pass returns image embeddings."""
    outputs = model(batch_images)
    outputs = first_available_output(
        outputs,
        preferred_keys=("features", "pooler_output", "last_hidden_state"),
    )
    return ensure_2d_features(outputs, model.__class__.__name__)


def extract_virchow2_features(model: Any, batch_images: Any) -> Any:
    """Extract Virchow2 features by concatenating CLS and mean patch tokens."""
    import torch

    outputs = model(batch_images)
    outputs = first_available_output(outputs, preferred_keys=("last_hidden_state", "features"))

    if not hasattr(outputs, "ndim") or outputs.ndim != 3:
        raise RuntimeError(
            "Virchow2 output is expected to be a token sequence with shape [B, N, C]."
        )

    # Virchow2 uses multiple register tokens. The commonly used representation
    # concatenates the CLS token with the mean of patch tokens after the registers.
    cls_token = outputs[:, 0]
    patch_tokens = outputs[:, 5:]
    return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)


def get_feature_extractor(model_name: str) -> Callable[[Any, Any], Any]:
    """Return the feature extraction function for a selected model."""
    extractor_map = {
        "CONCH": extract_conch_features,
        "CONCH_v1_5": extract_conch_v1_5_features,
        "UNI": extract_forward_features,
        "UNI_v2": extract_forward_features,
        "Virchow2": extract_virchow2_features,
    }
    return extractor_map[model_name]


def is_image_file(path: Path) -> bool:
    """Return True if a path is a supported image file."""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_patch_paths(slide_dir: Path, recursive: bool = False) -> list[Path]:
    """Collect supported patch image paths from one slide directory."""
    iterator = slide_dir.rglob("*") if recursive else slide_dir.iterdir()
    return sorted(path for path in iterator if is_image_file(path))


def collect_slide_dirs(patches_dir: Path) -> list[Path]:
    """Collect slide directories or treat patches_dir itself as a single slide."""
    # If images are stored directly under patches_dir, process it as one slide.
    if any(is_image_file(path) for path in patches_dir.iterdir()):
        return [patches_dir]

    slide_dirs = sorted(path for path in patches_dir.iterdir() if path.is_dir())
    if not slide_dirs:
        raise RuntimeError(f"No slide folders or patch images found in: {patches_dir}")

    return slide_dirs


def write_slide_features(slide_features: Any, save_path: Path) -> None:
    """Write a slide feature table atomically to a Feather file."""
    import pyarrow as pa
    import pyarrow.feather as feather

    save_path.parent.mkdir(parents=True, exist_ok=True)
    slide_features.index.name = "patches_name"

    table = pa.Table.from_pandas(slide_features, preserve_index=True)

    # Write to a temporary file first so interrupted runs do not leave a partial
    # output file with the final name.
    temp_path = save_path.with_name(f"{save_path.name}.tmp")
    feather.write_feather(table, temp_path)
    temp_path.replace(save_path)


def get_amp_context(device: Any, use_amp: bool, amp_dtype: str):
    """Create the automatic mixed precision context for CUDA inference."""
    if device.type != "cuda" or not use_amp:
        return nullcontext()

    import torch

    dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def extract_slide_features(
    slide_dir: Path,
    save_path: Path,
    preprocess: Callable[[Any], Any],
    model: Any,
    feature_extractor: Callable[[Any, Any], Any],
    device: Any,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
    amp_dtype: str,
    recursive: bool,
) -> int:
    """Extract all patch features for a single slide and save them to disk."""
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    patch_paths = collect_patch_paths(slide_dir, recursive=recursive)
    if not patch_paths:
        print(f"[skip] {slide_dir.name}: no patch images found.")
        return 0

    dataset = PatchDataset(slide_dir=slide_dir, patch_paths=patch_paths, preprocess=preprocess)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    patch_ids: list[str] = []
    feature_chunks: list[np.ndarray] = []

    with torch.inference_mode():
        for batch_ids, batch_images in data_loader:
            batch_images = batch_images.to(device, non_blocking=device.type == "cuda")

            with get_amp_context(device=device, use_amp=use_amp, amp_dtype=amp_dtype):
                embeddings = feature_extractor(model, batch_images)

            # Keep features as NumPy arrays instead of Python lists. This is both
            # faster and more memory efficient for large numbers of patches.
            embeddings_np = embeddings.detach().float().cpu().numpy()
            if embeddings_np.ndim != 2:
                raise RuntimeError(
                    f"Expected a 2D feature array, got shape {embeddings_np.shape}."
                )

            patch_ids.extend(str(patch_id) for patch_id in batch_ids)
            feature_chunks.append(embeddings_np)

    feature_matrix = np.concatenate(feature_chunks, axis=0)
    slide_features = pd.DataFrame(feature_matrix, index=patch_ids)
    write_slide_features(slide_features, save_path)
    return len(patch_ids)


def main() -> None:
    """Run feature extraction for all slides."""
    args = parse_args()
    validate_args(args)
    require_dependencies(args.model)

    hf_token = resolve_hf_token(args.hf_token)
    maybe_hf_login(hf_token)

    patches_dir = Path(args.patches_dir).resolve()
    output_dir = (Path(args.output_dir) / args.model).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    use_amp = not args.disable_amp

    print(f"Model: {args.model}")
    print(f"Patches dir: {patches_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")
    print(f"AMP enabled: {use_amp and device.type == 'cuda'}")
    print(f"Recursive image search: {args.recursive}")

    model, preprocess = load_model_and_preprocess(args.model, hf_token)
    model = model.to(device)
    model.eval()

    feature_extractor = get_feature_extractor(args.model)
    slide_dirs = collect_slide_dirs(patches_dir)
    total_slides = len(slide_dirs)

    for slide_idx, slide_dir in enumerate(slide_dirs, start=1):
        save_path = output_dir / f"{slide_dir.name}.feather"

        if save_path.exists() and not args.overwrite:
            print(f"[skip] {slide_idx}/{total_slides} {slide_dir.name}: {save_path.name} exists.")
            continue

        print(f"[run]  {slide_idx}/{total_slides} {slide_dir.name}")
        num_patches = extract_slide_features(
            slide_dir=slide_dir,
            save_path=save_path,
            preprocess=preprocess,
            model=model,
            feature_extractor=feature_extractor,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_amp=use_amp,
            amp_dtype=args.amp_dtype,
            recursive=args.recursive,
        )
        print(f"[save] {save_path} ({num_patches} patches)")


if __name__ == "__main__":
    main()
