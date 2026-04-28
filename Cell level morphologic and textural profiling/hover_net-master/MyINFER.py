import torch
import logging
import os
from misc.utils import log_info

# 示例配置：选择任务模式（'tile' 或 'wsi'）
sub_cmd = 'tile'  # 在此切换模式：'tile' 或 'wsi'

# 全局参数配置（所有模式共用）
global_args = {
    '--gpu': '0',           # 使用的GPU编号，多个用逗号分隔，如'0,1'
    '--nr_types': '6',      # 分类任务类型数，分割设为0
    '--type_info_path': 'type_info.json', # 类型信息json路径，分类任务需填写
    '--model_path': './MyModels/hovernet_fast_pannuke_type_tf2pytorch', # 模型路径
    '--model_mode': 'fast', # 模型模式：'fast'（分割）或'original'（分类）
    '--nr_inference_workers': '2',
    '--nr_post_proc_workers': '2',
    '--batch_size': '2'
}
# python run_infer.py --gpu='1' --nr_types=6 --type_info_path=type_info.json --batch_size=16 --model_mode=fast --model_path=F:\Yangzijian\project\hover_net-master\pretrained\hovernet_fast_pannuke_type_tf2pytorch.tar --nr_inference_workers=2 --nr_post_proc_workers=2 tile --input_dir=K:/BC/BC_visualisation/2024-03-14/patches_224/BRCA_mut --output_dir=K:/BC/BC_visualisation/2024-03-14/preds/patches_224/BRCA_mut

# 子命令参数配置（根据模式选择设置）
tile_args = {
    '--input_dir': '../../将10X对应到20X的Patch上/02将10Xpatch分割成更小Patch块存储/High_Image/',   # 输入图像目录
    '--output_dir': 'Out_PutDir/High_Image/',        # 输出目录
    '--mem_usage': '0.3',                  # 内存使用限制（0~1）
    '--draw_dot': 'True',                   # 绘制中心点
    '--save_qupath': 'False',
    '--save_raw_map': 'False'
}

wsi_args = {
    '--input_dir': './sample_data/wsi',
    '--output_dir': './output/wsi',
    '--cache_path': './cache',
    '--input_mask_dir': '',
    '--proc_mag': '40',
    '--ambiguous_size': '128',
    '--chunk_shape': '10000',
    '--tile_shape': '2048',
    '--save_thumb': 'False',
    '--save_mask': 'False'
}

# 不需要修改以下代码 ------------------------------------------------------
if __name__ == '__main__':
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d|%H:%M:%S',
        handlers=[
            logging.FileHandler("debug.log"),
            logging.StreamHandler()
        ]
    )

    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = global_args['--gpu']
    nr_gpus = torch.cuda.device_count()
    log_info(f'Detect #GPUS: {nr_gpus}')

    # 处理参数格式
    args = {k.replace('--', ''): v for k, v in global_args.items()}
    sub_args = tile_args if sub_cmd == 'tile' else wsi_args
    sub_args = {k.replace('--', ''): v for k, v in sub_args.items()}

    # 检查必要参数
    if not args['model_path']:
        raise ValueError("必须在global_args中设置model_path")

    # 构建方法参数
    nr_types = int(args['nr_types']) if int(args['nr_types']) > 0 else None
    method_args = {
        'method': {
            'model_args': {
                'nr_types': nr_types,
                'mode': args['model_mode'],
            },
            'model_path': args['model_path'],
        },
        'type_info_path': args['type_info_path'] or None
    }

    # 构建运行参数
    run_args = {
        'batch_size': int(args['batch_size']) * nr_gpus,
        'nr_inference_workers': int(args['nr_inference_workers']),
        'nr_post_proc_workers': int(args['nr_post_proc_workers'])
    }

    # 设置输入尺寸
    if args['model_mode'] == 'fast':
        run_args.update({
            'patch_input_shape': 256,
            'patch_output_shape': 164
        })
    else:
        run_args.update({
            'patch_input_shape': 270,
            'patch_output_shape': 80
        })

    # 添加模式特定参数
    run_args.update({
        'input_dir': sub_args['input_dir'],
        'output_dir': sub_args['output_dir']
    })

    if sub_cmd == 'tile':
        run_args.update({
            'mem_usage': float(sub_args['mem_usage']),
            'draw_dot': sub_args['draw_dot'] == 'True',
            'save_qupath': sub_args['save_qupath'] == 'True',
            'save_raw_map': sub_args['save_raw_map'] == 'True'
        })
    else:
        run_args.update({
            'cache_path': sub_args['cache_path'],
            'input_mask_dir': sub_args['input_mask_dir'],
            'proc_mag': int(sub_args['proc_mag']),
            'ambiguous_size': int(sub_args['ambiguous_size']),
            'chunk_shape': int(sub_args['chunk_shape']),
            'tile_shape': int(sub_args['tile_shape']),
            'save_thumb': sub_args['save_thumb'] == 'True',
            'save_mask': sub_args['save_mask'] == 'True'
        })

    # 执行推理
    if sub_cmd == 'tile':
        from infer.tile import InferManager
        infer = InferManager(**method_args)
        infer.process_file_list(run_args)
    else:
        from infer.wsi import InferManager
        infer = InferManager(**method_args)
        infer.process_wsi_list(run_args)