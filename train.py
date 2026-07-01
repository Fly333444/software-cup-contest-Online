# -*- coding: utf-8 -*-
"""
Training script for B榜 Fire Detection (single-class: firebig).
Uses Faster R-CNN R50 FPN with PaddleDetection framework, VOC dataset.
两阶段检测: RPN先提候选区域(回归定位) → 再分类+精修回归
640×480 原生输入, 无缩放无变形

Usage:
    python train.py                          # Default config
    python train.py --config configs/xxx.yml # Custom config
    python train.py -r output/N              # Resume from checkpoint N
"""
import os
import sys
import time
import signal
import subprocess
import atexit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PADDLE_DET = os.path.join(BASE_DIR, 'PaddleDetection')
sys.path.insert(0, PADDLE_DET)

CONFIG_FILE = os.path.join(BASE_DIR, "configs", "faster_rcnn_r50_fpn_firebig_b01.yml")


def kill_orphaned_train_processes():
    """Kill any orphaned Python training processes from this project before starting."""
    our_winpid = None
    try:
        with open('/proc/self/winpid') as f:
            our_winpid = int(f.read().strip())
    except Exception:
        pass
    if our_winpid is None:
        return
    killed = 0
    try:
        result = subprocess.run(
            ['ps', '-aW'], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            cmd = parts[6].lower()
            winpid_str = parts[3] if len(parts) >= 4 else ''
            try:
                winpid = int(winpid_str)
            except ValueError:
                continue
            if winpid == our_winpid:
                continue
            if any(kw in cmd for kw in ['picodet', 'ppdet', 'paddledetection']):
                subprocess.run(
                    ['/c/Windows/System32/taskkill.exe', '/F', '/PID', str(winpid)],
                    capture_output=True, timeout=3
                )
                print(f"[train.py] Killed orphaned training PID={winpid}")
                killed += 1
    except Exception:
        pass
    if killed > 0:
        print(f"[train.py] Killed {killed} orphaned training process(es). "
              "Waiting 2s for GPU release...")
        time.sleep(2)


def check_gpu_memory():
    """Check current GPU memory usage before starting training."""
    try:
        result = subprocess.run(
            ['/c/Windows/System32/nvidia-smi.exe',
             '--query-gpu=memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        line = result.stdout.strip()
        if line:
            parts = line.split(',')
            used = int(parts[0].strip())
            total = int(parts[1].strip())
            print(f"[train.py] GPU memory: {used}MiB / {total}MiB")
            if used > 1024:
                print(f"  Warning: {used}MiB already in use. "
                      "Clearing CUDA cache if possible...")
            return used
    except Exception:
        pass
    return None


def cleanup_gpu():
    """Release GPU memory on exit."""
    try:
        import paddle
        if paddle.device.cuda.device_count() >= 1:
            paddle.device.cuda.empty_cache()
            print("[train.py] CUDA cache cleared on exit.")
    except Exception:
        pass


def signal_handler(sig, frame):
    """Handle Ctrl+C / SIGINT gracefully."""
    print("\n[train.py] Received interrupt signal. Cleaning up GPU resources...")
    cleanup_gpu()
    sys.exit(0)


atexit.register(cleanup_gpu)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer
from ppdet.utils.cli import ArgsParser, merge_args
import ppdet.utils.check as check
from ppdet.utils.logger import setup_logger


def main():
    parser = ArgsParser()
    parser.add_argument(
        "-r", "--resume", default=None,
        help="weights path for resume (e.g. -r output/49)")

    argv = sys.argv[1:] if len(sys.argv) > 1 else []
    has_config = any(a in ('-c', '--config') for a in argv)
    if not has_config:
        argv = ['--config', CONFIG_FILE] + argv

    FLAGS = parser.parse_args(argv)
    cfg = load_config(FLAGS.config)
    merge_config(FLAGS.opt)

    # 全量训练: 所有图片 → ImageSets/Main/train.txt
    train_txt = os.path.join(BASE_DIR, 'B_data', 'ImageSets', 'Main', 'train.txt')
    if os.path.exists(train_txt):
        cfg.TrainDataset['anno_path'] = train_txt
    cfg.EvalDataset['anno_path'] = cfg.TrainDataset['anno_path']  # 全量训练，eval也用同一份(仅看loss)
    cfg.weights = f'output/{os.path.splitext(os.path.basename(FLAGS.config))[0]}/model_final'

    cfg.use_gpu = True
    cfg.use_npu = False
    cfg.use_xpu = False
    cfg.use_mlu = False

    import paddle
    paddle.set_device('gpu')

    check.check_config(cfg)
    check.check_gpu(cfg.use_gpu)
    check.check_version()

    logger = setup_logger('train')
    logger.info(f"Config: {FLAGS.config}")
    logger.info(f"Train: {cfg.TrainDataset['anno_path']}")
    logger.info(f"Eval: {cfg.EvalDataset['anno_path']}")
    logger.info(f"Num classes: {cfg.num_classes}")
    logger.info(f"Epochs: {cfg.epoch}")
    logger.info(f"Batch size: {cfg.TrainReader['batch_size']}")
    logger.info(f"Base LR: {cfg.LearningRate['base_lr']}")

    trainer = Trainer(cfg, mode='train')

    if FLAGS.resume is not None:
        trainer.resume_weights(FLAGS.resume)
    elif 'pretrain_weights' in cfg and cfg.pretrain_weights:
        pw = cfg.pretrain_weights
        if pw.startswith('http://') or pw.startswith('https://'):
            # Download from URL
            import urllib.request
            local_name = os.path.basename(pw)
            download_dir = os.path.join(BASE_DIR, 'pretrain')
            os.makedirs(download_dir, exist_ok=True)
            local_path = os.path.join(download_dir, local_name)
            if not os.path.exists(local_path):
                logger.info(f"Downloading pretrain weights from {pw} ...")
                urllib.request.urlretrieve(pw, local_path)
                logger.info(f"Downloaded to {local_path}")
            pretrain_path = local_path
        else:
            pretrain_path = os.path.join(BASE_DIR, pw)
        if os.path.exists(pretrain_path):
            logger.info(f"Loading pretrain weights: {pretrain_path}")
            trainer.load_weights(pretrain_path)
        else:
            logger.error(f"Pretrain weights not found: {pretrain_path}")
            sys.exit(1)

    do_eval = getattr(FLAGS, 'eval', True)
    trainer.train(do_eval)


if __name__ == "__main__":
    kill_orphaned_train_processes()
    check_gpu_memory()
    main()
