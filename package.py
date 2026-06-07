# -*- coding: utf-8 -*-
"""Package submission.zip for AI Studio evaluation."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def check_required_files():
    errors = []
    if not os.path.exists(os.path.join(BASE_DIR, "predict.py")):
        errors.append("predict.py missing")

    model_dir = os.path.join(BASE_DIR, "model")
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

    deploy_dir = os.path.join(BASE_DIR, "PaddleDetection", "deploy", "python")
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

    zip_path = os.path.join(BASE_DIR, "submission.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)

    import zipfile

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # predict.py
        zf.write('predict.py', 'predict.py')

        # model files
        for f in ['infer_cfg.yml', 'model.pdmodel', 'model.pdiparams']:
            zf.write(os.path.join('model', f), os.path.join('model', f))

        # PaddleDetection deploy files
        deploy_base = 'PaddleDetection/deploy/python'
        for f in ['preprocess.py', 'utils.py', 'keypoint_preprocess.py']:
            zf.write(os.path.join(deploy_base, f), os.path.join(deploy_base, f))

    zip_size = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\nCreated: {zip_path} ({zip_size:.1f} MB)")

    # Verify zip contents
    print("\nZip contents:")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for info in zf.infolist():
            print(f"  {info.filename} ({info.file_size} bytes)")


if __name__ == "__main__":
    main()
