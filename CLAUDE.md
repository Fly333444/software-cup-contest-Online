# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

火情检测竞赛项目 (Fire Detection Competition) — "中国软件杯" AI 与 AR 融合四足机器狗消防侦察救援系统。基于 PaddleDetection 框架训练 3 类目标检测模型 (battery/board/fire)，导出 Paddle Inference 格式提交到 AI Studio 评测。

比赛规则在 `机器狗消防侦察救援系统.md` 中，包括对提交格式的要求。

**关键约束**: 模型 ≤200MB, V100 GPU, FPS ≥20 (不达标直接 0 分), 入口 `python predict.py data.txt result.json`

## 环境

- 虚拟环境: `E:\Anaconda\envs\fire_env1` (Python 3.9, PaddlePaddle GPU 2.6.2, PaddleDetection release/2.8.1)
- 激活: `conda activate fire_env1`

## 常用命令

```bash
conda activate fire_env1
python convert_labelme_to_coco.py    # LabelMe → COCO, 80/20 划分
python train.py                       # 训练 (配置在 CONFIG_FILE 变量)
python evaluate.py                    # 评估验证集 (mAP + 每类 F1)
python export_model.py                # 导出 Paddle Inference 模型到 model/
python predict.py data.txt result.json # 推理 (比赛入口)
python package.py                     # 打包 submission.zip
```

训练输出保存到 `output/`，每 10 epoch 一个 checkpoint，`best_model.pdparams` 是最佳验证 mAP 模型。

## 项目架构

### 入口脚本 (项目根目录，独立运行)

| 脚本 | 说明 |
|------|------|
| `train.py` | 训练入口，加载 YAML config → PaddleDetection Trainer |
| `predict.py` | 推理入口，加载 Paddle Inference 模型，独立于 PaddleDetection |
| `evaluate.py` | COCO mAP + per-class F1 @ IoU=0.5 评估 |
| `export_model.py` | checkpoint → Paddle Inference 静态图格式 (model.pdmodel + model.pdiparams) |
| `convert_labelme_to_coco.py` | LabelMe JSON → COCO JSON, 80/20 train/val 划分 |
| `package.py` | 验证文件并打包 submission.zip |

### 目录结构

- `A_train/` — 训练数据: `Image/` (405 JPEG, 1920×1080), `label/` (LabelMe JSON), `train.json`/`val.json` (COCO)
- `configs/` — YAML 训练配置 (PicoDet, PP-YOLOE 变体)
- `model/` — 导出的 Paddle Inference 模型 (pdmodel + pdiparams + infer_cfg.yml)
- `output/` — 训练 checkpoint (.pdparams/.pdopt/.pdema/.pdstates)
- `PaddleDetection/` — vendored PaddleDetection 框架 (不需要单独安装)

### 关键架构

- **配置系统**: YAML `_BASE_` 继承机制，config 文件通过 `PaddleDetection/ppdet/core/workspace.py` 加载和合并
- **模型架构**: PicoDet = LCNet backbone + LCPAN neck + PicoHeadV2 (GFL detection head)
- **推理流程**: predict.py 不依赖 PaddleDetection，直接使用 Paddle Inference API 加载 `.pdmodel`/`.pdiparams`
- **标注格式**: COCO JSON，3 类 ID: battery=1, board=2, fire=3 (1-indexed)
- **评测指标**: Mean F1 @ IoU=0.5 (3 类均值), FPS = 2026 / 推理秒数

### 当前模型状态

- PicoDet-S (LCNet 0.75x), 640×640, ~5MB
- 本地 val: Mean F1=0.9546, FPS=28.9
- 线上: Score=0.72362, FPS=20.32
- 训练数据: 405 张, 3 类

### 训练配置继承链

```
configs/picodet_s_fire_100e.yml
  ├── PaddleDetection/configs/datasets/coco_detection.yml
  ├── PaddleDetection/configs/runtime.yml
  ├── PaddleDetection/configs/picodet/_base_/picodet_v2.yml
  └── PaddleDetection/configs/picodet/_base_/optimizer_300e.yml
```

### 注意事项

- Paddle Inference C++ 引擎不支持中文路径，model/ 路径需全英文
- 部分图片不含目标 (负样本)，配置需 `allow_empty: true`
- 导出模型需要 `combine_params=True` 才能生成单文件 `model.pdiparams`
- `train.py` 通过 `PADDLE_DET` 路径将 PaddleDetection 加入 sys.path，不使用 pip 安装版本
