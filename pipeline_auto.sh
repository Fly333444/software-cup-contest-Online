#!/bin/bash
# Auto pipeline — train → evaluate → export → sweep → package
# Usage: bash pipeline_auto.sh

PYTHON='E:\Anaconda\envs\fire_env1\python.exe'
DIR='E:\Study files\Ruanjian\submission_v9'

cd "$DIR" || exit 1

echo "============================================"
echo "v9 Pipeline — PicoDet-M 416 + TensorRT"
echo "============================================"

STEP=1
run_step() {
    echo ""
    echo "[$STEP/5] $1"
    echo "----------------------------------------"
    "$PYTHON" "$2" 2>&1 | tee -a pipeline.log
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "ERROR: $2 failed!"
        exit 1
    fi
    STEP=$((STEP + 1))
}

run_step "evaluate.py" "evaluate.py"
run_step "export_model.py" "export_model.py"
run_step "sweep_threshold.py" "sweep_threshold.py"
run_step "package.py" "package.py"

echo ""
echo "============================================"
echo "ALL DONE! submission.zip ready."
echo "============================================"
