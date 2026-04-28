import sys
sys.path.append('../')
import torch
import logging
import os
import copy
from misc.utils import log_info
from docopt import docopt

"""run_infer.py

Usage:
  run_infer.py [options] [--help] <command> [<args>...]
  run_infer.py --version
  run_infer.py (-h | --help)

Options:
  -h --help                   Show this string.
  --version                   Show version.

  --gpu=<id>                  GPU list. [default: 0]
  --nr_types=<n>              Number of nuclei types to predict. [default: 0]
  --type_info_path=<path>     Path to a json define mapping between type id, type name, 
                              and expected overlaid color. [default: '']

  --model_path=<path>         Path to saved checkpoint.
  --model_mode=<mode>         Original HoVer-Net or the reduced version used PanNuke and MoNuSAC, 
                              'original' or 'fast'. [default: fast]
  --nr_inference_workers=<n>  Number of workers during inference. [default: 8]
  --nr_post_proc_workers=<n>  Number of workers during post-processing. [default: 16]
  --batch_size=<n>            Batch size per 1 GPU. [default: 32]

Two command mode are `tile` and `wsi` to enter corresponding inference mode
    tile  run the inference on tile
    wsi   run the inference on wsi

Use `run_infer.py <command> --help` to show their options and usage.
"""

tile_cli = """
Arguments for processing tiles.

usage:
    tile (--input_dir=<path>) (--output_dir=<path>) \
         [--draw_dot] [--save_qupath] [--save_raw_map] [--mem_usage=<n>]
    
options:
   --input_dir=<path>     Path to input data directory. Assumes the files are not nested within directory.
   --output_dir=<path>    Path to output directory..

   --mem_usage=<n>        Declare how much memory (physical + swap) should be used for caching. 
                          By default it will load as many tiles as possible till reaching the 
                          declared limit. [default: 0.2]
   --draw_dot             To draw nuclei centroid on overlay. [default: False]
   --save_qupath          To optionally output QuPath v0.2.3 compatible format. [default: False]
   --save_raw_map         To save raw prediction or not. [default: False]
"""


#-------------------------------------------------------------------------------------------------------
def MyCell_Meg():
    model_mode = 'fast'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    nr_gpus = torch.cuda.device_count()
    model_path = './MyModels/hovernet_fast_pannuke_type_tf2pytorch'
    nr_types = 6
    type_info_path = 'type_info.json'
    batch_size = 1
    nr_inference_workers = 2
    nr_post_proc_workers = 4
    sub_cmd = 'tile'
    input_dir = 'Input_PutDir/1019382/Multimodel_Image/'
    output_dir = 'Out_PutDir/1019382/Multimodel_Image/'
    draw_dot = False
    save_qupath = False
    save_raw_map = False
    mem_usage = 0.2
    method_args = {
        'method': {
            'model_args': {
                'nr_types': nr_types,
                'mode': model_mode,
            },
            'model_path': model_path,
        },
        'type_info_path': None if type_info_path == '' \
            else type_info_path,
    }
    # ***
    run_args = {
        'batch_size': batch_size * nr_gpus,
        'nr_inference_workers': nr_inference_workers,
        'nr_post_proc_workers': nr_post_proc_workers,
    }
    if model_mode == 'fast':
        run_args['patch_input_shape'] = 256
        run_args['patch_output_shape'] = 164
    else:
        run_args['patch_input_shape'] = 270
        run_args['patch_output_shape'] = 80

    if sub_cmd == 'tile':
        run_args.update({
            'input_dir': input_dir,
            'output_dir': output_dir,

            'mem_usage': mem_usage,
            'draw_dot': draw_dot,
            'save_qupath': save_qupath,
            'save_raw_map': save_raw_map,
        })
    # ***

    if sub_cmd == 'tile':
        from infer.tile import InferManager
        infer = InferManager(**method_args)
        infer.process_file_list(run_args)
    else:
        from infer.wsi import InferManager
        infer = InferManager(**method_args)
        infer.process_wsi_list(run_args)

#python run_infer.py --gpu='1' --nr_types=6 --type_info_path=type_info.json --batch_size=16 --model_mode=fast --model_path=F:\Yangzijian\project\hover_net-master\pretrained\hovernet_fast_pannuke_type_tf2pytorch.tar --nr_inference_workers=2 --nr_post_proc_workers=2 tile --input_dir=K:/BC/BC_visualisation/2024-03-14/patches_224/BRCA_mut --output_dir=K:/BC/BC_visualisation/2024-03-14/preds/patches_224/BRCA_mut


if __name__ == '__main__':
    # MyCell_Meg()
    model_mode='fast'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    nr_gpus = torch.cuda.device_count()
    model_path='./MyModels/hovernet_fast_pannuke_type_tf2pytorch'
    nr_types=6
    type_info_path='type_info.json'
    batch_size=1
    nr_inference_workers=2
    nr_post_proc_workers=4
    sub_cmd = 'tile'
    Path="Input_PutDir/Low_Image_CCL/"
    Images_=Path
    input_dir = Images_
    output_dir = "Out_PutDir/Low_Image_CCL/"
    draw_dot=False
    save_qupath=False
    save_raw_map=False
    mem_usage=0.2
    method_args = {
        'method' : {
            'model_args' : {
                'nr_types'   : nr_types,
                'mode'       : model_mode,
            },
            'model_path' : model_path,
        },
        'type_info_path'  : None if type_info_path == '' \
                            else type_info_path,
    }
    # ***
    run_args = {
        'batch_size' : batch_size * nr_gpus,
        'nr_inference_workers' : nr_inference_workers,
        'nr_post_proc_workers' : nr_post_proc_workers,
    }
    if model_mode== 'fast':
        run_args['patch_input_shape'] = 256
        run_args['patch_output_shape'] = 164
    else:
        run_args['patch_input_shape'] = 270
        run_args['patch_output_shape'] = 80

    if sub_cmd == 'tile':
        run_args.update({
            'input_dir'      : input_dir,
            'output_dir'     : output_dir,

            'mem_usage'   : mem_usage,
            'draw_dot'    : draw_dot,
            'save_qupath' : save_qupath,
            'save_raw_map': save_raw_map,
        })
    # ***

    if sub_cmd == 'tile':
        from infer.tile import InferManager
        infer = InferManager(**method_args)
        infer.process_file_list(run_args)
    else:
        from infer.wsi import InferManager
        infer = InferManager(**method_args)
        infer.process_wsi_list(run_args)
