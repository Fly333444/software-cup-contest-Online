# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository. 与我对话请用简体中文

## Project Overview

火情检测竞赛项目 — v13.0 (历史最优 🏆)  
基于 PaddleDetection 框架训练 PicoDet-M 3 类目标检测模型 (battery/board/fire)，导出 Paddle Inference 格式提交到 AI Studio 评测。

**线上成绩**: F1=**0.89144** (历史最高), FPS=**40.37**  
**核心策略**: 原始比赛标注 + 人工修正，回归 v2.2/v9 验证配方 (PicoDet-M 416, GridMask+多尺度, 200e)

比赛规则在 `机器狗_AR目标检测比赛规则.md` 中。
**关键约束**: 模型 ≤200MB, V100 GPU, FPS ≥20, 入口 `python predict.py data.txt result.json`

## 环境

- 虚拟环境: `E:\Anaconda\envs\fire_env1` (Python 3.9, PaddlePaddle GPU 2.6.2, PaddleDetection release/2.8.1)
- 激活: `conda activate fire_env1`

## 常用命令

```bash
conda activate fire_env1
python convert_labelme_to_coco.py     # LabelMe → COCO, 80/20 划分
python train.py                        # 训练 (PicoDet-M 416, 200e)
python evaluate.py                     # 评估验证集 (mAP + 每类 F1)
python sweep_threshold.py              # 扫描 score_threshold 找最优值
python export_model.py                 # 导出 Paddle Inference 模型到 model/
python predict.py data.txt result.json # 推理入口
python package.py                      # 打包 submission.zip
```

## 项目架构

### 入口脚本 (项目根目录，独立运行)

| 脚本 | 说明 |
|------|------|
| `train.py` | 训练入口，默认加载 `configs/picodet_m_fire_v13.yml` |
| `predict.py` | 推理入口，加载 Paddle Inference 模型 |
| `evaluate.py` | COCO mAP + per-class F1 @ IoU=0.5 评估 |
| `export_model.py` | checkpoint → Paddle Inference 静态图格式 |
| `sweep_threshold.py` | 扫描 score_threshold 找最优 Mean F1 |
| `convert_labelme_to_coco.py` | LabelMe JSON → COCO JSON, 80/20 train/val |
| `package.py` | 验证文件并打包 submission.zip |

### 目录结构

- `A_train/` — 训练数据: `Image/` (405 JPEG), `label/` (405 LabelMe JSON), `train.json`/`val.json` (COCO)
- `configs/` — YAML 训练配置 (v13 用 `picodet_m_fire_v13.yml`)
- `model/` — 导出的 Paddle Inference 模型
- `output/` — 训练 checkpoint

### 当前模型状态 (v13.0)

**PicoDet-M 416×416, 200 epoch, GridMask + 多尺度 320~576**
- **线上结果**: F1=**0.89144** (历史最高), FPS=**40.37**
- **训练数据**: 原始比赛标注 (126+92+712) + 人工修正 (+10), 324 train / 81 val
- **Best mAP**: 0.707 (val), AP@0.50=0.865
- **本地 Mean F1**: 0.9751 (battery=0.9831, board=0.9500, fire=0.9924)
- **最优阈值**: global=0.05 (本地 F1=0.9590)
- **模型大小**: 13.7 MB
- **预训练**: COCO PicoDet-M (picodet_m_320_coco_lcnet.pdparams)

### 推理命令 (AI Studio)

```bash
python predict.py data.txt result.json 0.05
```
