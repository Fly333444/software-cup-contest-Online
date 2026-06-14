# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository. 与我对话请用简体中文

## Project Overview

火情检测竞赛项目 — v15.0 (PicoDet-L 升级版)  
基于 PaddleDetection 框架训练 PicoDet-L 3 类目标检测模型 (battery/board/fire)，导出 Paddle Inference 格式提交到 AI Studio 评测。

**v13 线上成绩 (Baseline)**: F1=**0.89144**, FPS=**40.37**  
**v15 升级**: PicoDet-M → PicoDet-L (LCNet 1.5x→2.0x, 3.46M→~5.8M params)  
**核心策略**: V13 历史最优配方 + 更大模型 + 原始标注 + 人工修正

比赛规则在 `机器狗_AR目标检测比赛规则.md` 中。
**关键约束**: 模型 ≤200MB, V100 GPU, FPS ≥20, 入口 `python predict.py data.txt result.json`

## 环境

- 虚拟环境: `E:\Anaconda\envs\fire_env1` (Python 3.9, PaddlePaddle GPU 2.6.2, PaddleDetection release/2.8.1)
- 激活: `conda activate fire_env1`

## 常用命令

```bash
conda activate fire_env1
python convert_labelme_to_coco.py     # LabelMe → COCO, 80/20 划分
python train.py                        # 训练 (PicoDet-L 416, 200e)
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
| `train.py` | 训练入口，默认加载 `configs/picodet_l_fire_v15.yml` |
| `predict.py` | 推理入口，加载 Paddle Inference 模型 |
| `evaluate.py` | COCO mAP + per-class F1 @ IoU=0.5 评估 |
| `export_model.py` | checkpoint → Paddle Inference 静态图格式 |
| `sweep_threshold.py` | 扫描 score_threshold 找最优 Mean F1 |
| `convert_labelme_to_coco.py` | LabelMe JSON → COCO JSON, 80/20 train/val |
| `package.py` | 验证文件并打包 submission.zip |

### 目录结构

- `A_train/` — 训练数据: `Image/` (408 JPEG), `label/` (405 LabelMe JSON), `train.json`/`val.json` (COCO)
- `configs/` — YAML 训练配置 (v15 用 `picodet_l_fire_v15.yml`)
- `model/` — 导出的 Paddle Inference 模型
- `output/` — 训练 checkpoint

### 当前模型状态 (v15.0 — PicoDet-L 新实验)

**PicoDet-L 416×416, 200 epoch, GridMask + 多尺度 320~576**
- **模型**: PicoDet-L (LCNet 2.0x, ~5.8M params, LCPAN 160ch)
- **预训练**: COCO PicoDet-L (`picodet_l_320_coco_lcnet.pdparams`)
- **数据集**: 原始比赛标注 + 人工修正, 324 train / 81 val
- **增强**: GridMask + BatchRandomResize 320~576 + RandomDistort
- **优化器**: Momentum + CosineDecay, base_lr=0.04, EMA

### 推理命令 (AI Studio)

```bash
python predict.py data.txt result.json 0.05
```

## Git 推送流程（推送 v15 到队长仓库）

### 基础信息

- **当前仓库**: `submission_v15/` 目录下的 `.git`（只有这个目录有完整 git 仓库）
- **远程**: `origin = https://github.com/Fly333444/software-cup-contest-Online.git`
- **分支**: `Summit-online-best-version`（最佳版本专用分支）
- **Git 可执行文件**: 用 GitHub Desktop 自带的（不要假设系统 PATH 有 git）
  ```bash
  GIT="/c/Users/LENOVO/AppData/Local/GitHubDesktop/app-*/resources/app/git/cmd/git.exe"
  ```
  注意版本号会变（当前 `app-3.5.4`），用通配符 `*`

### 完整推送命令（在 bash 中运行）

```bash
# 1. 进入 v15 目录（它自己有 .git）
cd /e/"Study files"/Ruanjian/submission_v15

# 2. 用完整路径调用 git（系统 PATH 可能找不到 git）
GIT=/c/Users/LENOVO/AppData/Local/GitHubDesktop/app-*/resources/app/git/cmd/git.exe

# 3. 确保在正确分支
$GIT branch  # 应该显示 * Summit-online-best-version

# 4. 添加所有变更、提交、推送
$GIT add -A
$GIT commit -m "描述变更内容"
$GIT push origin Summit-online-best-version
```

### 注意事项

1. **不要在 CMD 中运行**，必须在 bash 中（Claude Code 的环境）
2. **不用写 `-C` 参数**，因为已经在目录内，cd 进去即可
3. **Credential** 用 Windows 凭据管理器（已配置 `credential.helper=manager`），推送时会自动使用 GitHub Desktop 缓存的认证
4. **如果 Git 插件 403** → 走本地 git 方式，不要事和 GitHub API 插件死磕
