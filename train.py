# -*- coding: utf-8 -*-
"""
Training script for Fire Detection Competition (3-class: battery, board, fire).
Uses PicoDet-L with PaddleDetection framework (v15).

Features:
    - Automatically kills orphaned training processes before starting
    - Gracefully handles Ctrl+C to release GPU memory
    - Clears CUDA cache on exit

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

CONFIG_FILE = os.path.join(BASE_DIR, "configs", "picodet_l_fire_v15.yml")


def kill_orphaned_train_processes():
    """Kill any orphaned Python training processes from this project before starting.

    On Windows, PaddlePaddle DataLoader worker processes are not reliably killed
    when Ctrl+C is pressed, leaving orphan processes that hold GPU memory.
    This prevents running multiple trainings simultaneously and exhausting VRAM.
    """
    our_winpid = None
    try:
        with open('/proc/self/winpid') as f:
            our_winpid = int(f.read().strip())
    except Exception:
        pass
    if our_winpid is None:
        return

    # Use ps -aW to find python processes with training-related command lines.
    # Format: PID PPID PGID WINPID TTY UID STIME COMMAND
    # The last field (COMMAND) contains the full path which reveals our training.
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
            # Skip self
            if winpid == our_winpid:
                continue
            # Only kill if it's clearly a PaddleDetection training process
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
            if used > 1024:  # More than 1GB used before training
                print(f"  ⚠  Warning: {used}MiB already in use. "
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
    """Handle Ctrl+C / SIGINT gracefully: log message and exit cleanly."""
    print("\n[train.py] Received interrupt signal. Cleaning up GPU resources...")
    cleanup_gpu()
    sys.exit(0)


# Register cleanup hooks
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
        "-r", "--resume", default=None, help="weights path for resume (e.g. -r output/49)")

    argv = sys.argv[1:] if len(sys.argv) > 1 else []
    has_config = any(a in ('-c', '--config') for a in argv)
    if not has_config:
        argv = ['--config', CONFIG_FILE] + argv

    FLAGS = parser.parse_args(argv)
    cfg = load_config(FLAGS.config)

    cfg.TrainDataset.dataset_dir = BASE_DIR
    cfg.EvalDataset.dataset_dir = BASE_DIR
    cfg.TestDataset.dataset_dir = BASE_DIR

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
    logger.info(f"Num classes: {cfg.num_classes}")
    logger.info(f"Epochs: {cfg.epoch}")
    logger.info(f"Batch size: {cfg.TrainReader['batch_size']}")
    logger.info(f"Base LR: {cfg.LearningRate['base_lr']}")

    trainer = Trainer(cfg, mode='train')

    if FLAGS.resume is not None:
        trainer.resume_weights(FLAGS.resume)
    elif 'pretrain_weights' in cfg and cfg.pretrain_weights:
        # Use local cache if available (bypass proxy-blocked remote MD5 check)
        local_path = os.path.expanduser(
            '~/.cache/paddle/weights/' +
            os.path.basename(cfg.pretrain_weights)
        )
        if os.path.exists(local_path):
            logger.info(f"Loading pretrain weights from local cache: {local_path}")
            trainer.load_weights(local_path)
        else:
            trainer.load_weights(cfg.pretrain_weights)

    do_eval = getattr(FLAGS, 'eval', True)
    trainer.train(do_eval)


if __name__ == "__main__":
    # Kill orphaned training processes before starting
    kill_orphaned_train_processes()
    check_gpu_memory()
    main()
