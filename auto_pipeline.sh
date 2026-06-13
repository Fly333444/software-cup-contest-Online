#!/bin/bash
# V12 Auto Pipeline — runs after training completes
# Usage: bash auto_pipeline.sh

PYTHON='E:\Anaconda\envs\fire_env1\python.exe'
DIR='E:\Study files\Ruanjian\submission_v12'

cd "$DIR" || exit 1
exec >> pipeline_v12.log 2>&1

echo "============================================"
echo "V12 Auto Pipeline — $(date)"
echo "============================================"

run_step() {
    echo ""
    echo "[$1] $2"
    echo "----------------------------------------"
    "$PYTHON" "$3"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "ERROR: $3 failed!"
        exit 1
    fi
}

run_step 0 "augment_dataset.py (Board Copy-Paste)" "augment_dataset.py"
run_step 1 "evaluate.py" "evaluate.py"
run_step 2 "export_model.py" "export_model.py"
run_step 3 "sweep_threshold.py" "sweep_threshold.py"
run_step 4 "package.py" "package.py"

echo ""
echo "============================================"
echo "V12 PIPELINE COMPLETE — $(date)"
echo "submission.zip ready!"
echo "============================================"
