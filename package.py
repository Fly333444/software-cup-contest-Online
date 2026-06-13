# -*- coding: utf-8 -*-
"""Package submission.zip for AI Studio evaluation (v9).

Usage:
    python package.py
"""
import os
import sys
import zipfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def check_required_files():
    errors = []
    if not os.path.exists(os.path.join(SCRIPT_DIR, "predict.py")):
        errors.append("predict.py missing")

    model_dir = os.path.join(SCRIPT_DIR, "model")
    if not os.path.isdir(model_dir):
        errors.append("model/ directory missing")
    else:
        for fname in ["infer_cfg.yml", "model.pdmodel", "model.pdiparams"]:
            fp = os.path.join(model_dir, fname)
            if not os.path.exists(fp):
                errors.append(f"model/{fname} missing")
            else:
                size_kb = os.path.getsize(fp) / 1024
                print(f"  model/{fname}: {size_kb:.1f} KB")

        total = sum(
            os.path.getsize(os.path.join(model_dir, f))
            for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))
        )
        mb = total / (1024 * 1024)
        print(f"  Total model size: {mb:.1f} MB (limit: 200 MB)")
        if total > 200 * 1024 * 1024:
            errors.append(f"model/ too large: {mb:.1f} MB > 200 MB")

    deploy_dir = os.path.join(SCRIPT_DIR, "PaddleDetection", "deploy", "python")
    if not os.path.isdir(deploy_dir):
        errors.append("PaddleDetection/deploy/python/ missing")
    else:
        for fname in ["preprocess.py", "utils.py", "keypoint_preprocess.py"]:
            if not os.path.exists(os.path.join(deploy_dir, fname)):
                errors.append(f"PaddleDetection/deploy/python/{fname} missing")

    return errors


def main():
    print("Checking submission...")
    errors = check_required_files()
    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"  - {e}")
        return

    print("\nAll checks passed!")

    zip_path = os.path.join(SCRIPT_DIR, "submission.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # predict.py
        zf.write(os.path.join(SCRIPT_DIR, 'predict.py'), 'predict.py')

        # model files
        model_dir = os.path.join(SCRIPT_DIR, 'model')
        for f in ['infer_cfg.yml', 'model.pdmodel', 'model.pdiparams']:
            fp = os.path.join(model_dir, f)
            if os.path.exists(fp):
                zf.write(fp, os.path.join('model', f))
                print(f"  + model/{f}")

        # PaddleDetection deploy files
        deploy_base = os.path.join(SCRIPT_DIR, 'PaddleDetection', 'deploy', 'python')
        for f in ['preprocess.py', 'utils.py', 'keypoint_preprocess.py']:
            fp = os.path.join(deploy_base, f)
            if os.path.exists(fp):
                zf.write(fp, os.path.join('PaddleDetection', 'deploy', 'python', f))
                print(f"  + PaddleDetection/deploy/python/{f}")

    zip_size = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\nCreated: submission.zip ({zip_size:.1f} MB)")

    # Verify zip contents
    print("\nZip contents:")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for info in zf.infolist():
            size_mb = info.file_size / (1024 * 1024)
            if size_mb > 0.1:
                print(f"  {info.filename:50s} {size_mb:.1f} MB")
            else:
                print(f"  {info.filename:50s} {info.file_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
