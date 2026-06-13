#!/usr/bin/env python3
"""Monitor Stage 1 training progress and checkpoint status.
Usage: python monitor_progress.py
"""
import os, time, glob, json

BASE = 'E:/Study files/Ruanjian/submission_v12'
STAGE1_DIR = os.path.join(BASE, 'output', 'ppyoloe_plus_l_fire_v12_stage1')

while True:
    import subprocess
    # safe: static command, no user input
    subprocess.run(['cmd.exe', '/c', 'cls'] if os.name == 'nt' else ['clear'])

    # Check training process
    try:
        import psutil
        train_procs = [p for p in psutil.process_iter(['pid', 'cmdline', 'create_time'])
                      if p.info['cmdline'] and 'train.py' in ' '.join(p.info['cmdline'][:2]) if p.info['cmdline']]
    except:
        train_procs = []

    if train_procs:
        for p in train_procs:
            elapsed = time.time() - p.info['create_time']
            print(f"Training process: PID={p.info['pid']}, running for {elapsed/3600:.1f}h")
    else:
        print("Training process: NOT RUNNING")

    # Check checkpoints (config sets weights=output/ppyoloe_plus_l_fire_v12_stage1/model_final)
    from collections import defaultdict
    # Check all possible output dirs
    all_ckpts = sorted(glob.glob(os.path.join(BASE, 'output', '**', '*[0-9].pdparams'), recursive=True), key=os.path.getmtime)
    # Filter out best_model and model_final
    train_ckpts = [c for c in all_ckpts if not any(x in os.path.basename(c) for x in ['best_model', 'model_final'])]

    # Check model_final
    if os.path.exists(os.path.join(STAGE1_DIR, 'model_final.pdparams')):
        print(">>> Stage 1 COMPLETE! <<<")

    # Check train log progress
    log_file = os.path.join(BASE, 'train_stage1.log')
    if os.path.exists(log_file):
        with open(log_file) as f:
            lines = f.readlines()
        epoch_lines = [l for l in lines if 'Epoch:' in l]
        if epoch_lines:
            last = epoch_lines[-1]
            import re
            m = re.search(r'Epoch: \[(\d+)\]', last)
            if m:
                current_epoch = int(m.group(1))
                pct = current_epoch / 100 * 100
                print(f"Current progress: Epoch {current_epoch}/100 ({pct:.0f}%)")

                m2 = re.search(r'eta: (\d+):(\d+):(\d+)', last)
                if m2:
                    h, m, s = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
                    remaining = h + m/60 + s/3600
                    total_est = remaining / (100 - current_epoch) * 100 if current_epoch > 0 else 0
                    print(f"Per-epoch eta: {remaining/(100 - current_epoch):.1f}h" if current_epoch > 0 else "")
                    print(f"~ETA finish: {time.strftime('%H:%M', time.localtime(time.time() + remaining))}")

    print(f"\nChecking again in 5 min... ({time.strftime('%H:%M:%S')})")
    time.sleep(300)
