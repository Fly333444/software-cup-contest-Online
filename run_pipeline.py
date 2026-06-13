#!/usr/bin/env python3
"""Auto pipeline runner for v12 (PP-YOLOE+_m, single stage).
Monitors training progress, then auto-runs: evaluate → export → sweep → package

Usage:
    python run_pipeline.py
"""
import os, sys, time, subprocess, glob

BASE = 'E:/Study files/Ruanjian/submission_v12'
os.chdir(BASE)
PY = 'E:/Anaconda/envs/fire_env1/python.exe'
LOG = 'pipeline_v12.log'

def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def wait_for_training(output_subdir, target_epochs=300):
    """Wait for training output/model_final.pdparams. Check every 2 min."""
    out_dir = os.path.join(BASE, 'output', output_subdir)
    log(f'Watching {output_subdir} for completion ({target_epochs} epochs)...')
    while True:
        ckpts = sorted(glob.glob(os.path.join(out_dir, '*[0-9].pdparams')), key=os.path.getmtime)
        if ckpts:
            latest = os.path.basename(ckpts[-1]).split('.')[0]
            log(f'  Latest: epoch {latest}')
        if os.path.exists(os.path.join(out_dir, 'model_final.pdparams')):
            log('model_final.pdparams found! Training complete.')
            return True
        if os.path.exists(os.path.join(BASE, 'output', 'model_final.pdparams')):
            log('model_final.pdparams found in output/ root! Training complete.')
            return True
        time.sleep(120)

def run_step(name, script, timeout_sec=3600):
    log(f'Running {name} ({script})...')
    result = subprocess.run([PY, script], capture_output=True, text=True, timeout=timeout_sec)
    out = result.stdout[-2000:] if result.stdout else ''
    if out:
        log(out)
    if result.returncode != 0:
        err = result.stderr[-1000:] if result.stderr else ''
        log(f'ERROR: {script} exited with code {result.returncode}')
        if err: log(err)
        return False
    log(f'{name} OK.')
    return True

log('='*50)
log('V12 PIPELINE STARTED')
log('='*50)

log('Waiting for training to finish...')
wait_for_training('ppyoloe_plus_m_fire_v12', 300)
time.sleep(10)

steps = [
    ('1/4 evaluate.py', 'evaluate.py', 3600),
    ('2/4 export_model.py', 'export_model.py', 3600),
    ('3/4 sweep_threshold.py', 'sweep_threshold.py', 3600),
    ('4/4 package.py', 'package.py', 600),
]

for name, script, timeout in steps:
    if not run_step(name, script, timeout):
        log(f'PIPELINE FAILED at {name}')
        sys.exit(1)

if os.path.exists('submission.zip'):
    sz = os.path.getsize('submission.zip')
    log(f'SUCCESS! submission.zip ready ({sz/1024/1024:.1f} MB)')
else:
    log('ERROR: submission.zip not found!')
    sys.exit(1)

log('='*50)
log('V12 PIPELINE COMPLETE')
log('='*50)
