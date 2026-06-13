# 火情检测模型 — "中国软件杯" 四足机器狗消防侦察

基于 PaddleDetection 的 3 类火焰场景目标检测（battery / board / fire），导出 Paddle Inference 格式提交 AI Studio 评测。

## 比赛成绩

| 版本 | 模型 | 输入尺寸 | 线上 F1 | FPS | 说明 |
|------|------|----------|---------|-----|------|
| v1 | PicoDet-S | 640×640 | 0.72362 | 20.32 | 基线 |
| v2.2 | PicoDet-M | 416×416 | 0.88212 | 38.47 | GridMask + 多尺度 320~576 |
| v3.0 | PicoDet-M | 512×512 | 0.47479 | 30.21 | predict.py bug 导致暴降 |
| v3.1 | PicoDet-M | 512×512 | 0.48982 | 23.17 | 512 过拟合 |
| v3.2 | PicoDet-M | 416×416 | 0.88212 | 30.42 | 复用 v2.2 模型 + 批量推理 |
| v3.3 | PicoDet-M | 416×416 | 0.87148 | 38.65 | Copy-Paste + Mosaic + Mixup |
| v4.0 | PicoDet-M | 416×416 | 0.87628 | 36.38 | GridMask 300e |
| v5.0 | PicoDet-L | 416×416 | 0.87459 | 40.65 | +D-Fire 外部火焰，两阶段 |
| v6.0 | PP-YOLOE+_m | 416×416 | 0.86061 | 52.71 | Obj365 预训练 + 数据再平衡 |
| v7.0 | PicoDet-M | 416×416 | 0.84443 | 35.22 | 伪标注补全漏标 |
| v9.0 | PicoDet-M | 416×416 | 0.88858 | 31.12 | 回归 v2.2 模型 + v6.0 工程栈 |
| v10.0 | PicoDet-M | 416×416 | 0.8852 | 40.69 | 300e 长训练 |
| v11.0 | PicoDet-M | 416×416 | 0.88851 | 40.51 | Mosaic+Mixup+GridMask 300e |
| v12.0 | PP-YOLOE+_m | 416×416 | 0.88438 | 34.28 | Obj365 预训练, 300e |
| **v13.0** 🏆 | **PicoDet-M** | **416×416** | **0.89144** | **40.37** | **原始标注 + 人工修正** |

---

## v13.0 模型详解 🏆

### 概述

v13.0 以 **F1=0.89144** 刷新历史最高分，同时保持 **FPS=40.37** 的流畅推理速度。核心思路是**回归原始比赛标注 + 人工修正**，不再依赖任何伪标注、外部数据或大模型，验证了**标注质量 > 数据数量 > 模型规模**的迭代结论。

### 训练数据

| 数据来源 | 图片数 | battery | board | fire | 总标注数 |
|----------|--------|---------|-------|------|---------|
| 原始比赛标注 (A_train.zip) | 405 | 126 | 92 | 712 | 930 |
| 人工修正 (+10 标注) | 同 405 张 | 127 (+1) | 98 (+6) | 715 (+3) | 940 |

- 全部 405 张图片来自比赛原始数据集，**未引入任何外部数据**
- 人工修正主要补充 board 类（+6，最多）和少量的 battery/fire
- 按 80/20 随机划分：Train 324 张 / Val 81 张

### 模型配置

| 参数 | 值 |
|------|-----|
| 模型 | PicoDet-M (LCNet 1.5x, LCPAN 128ch) |
| 输入尺寸 | 416×416 (Resize keep_ratio=False) |
| 训练 Epoch | 200 |
| Batch Size | 8 |
| Base LR | 0.04 (CosineDecay + LinearWarmup 500 steps) |
| 预训练 | COCO PicoDet-M (picodet_m_320_coco_lcnet) |
| 增强 | GridMask (prob=0.3) + 多尺度随机 (320~576) + RandomFlip + RandomDistort |
| EMA | 启用 (decay=0.9998) |
| NMS | MultiClassNMS (score_threshold=0.025, nms_threshold=0.6, keep_top_k=100) |

### 线上成绩

| 指标 | 值 |
|------|-----|
| **Mean F1** | **0.89144** |
| **FPS** | **40.37** |
| 阈值 | global=0.05 |
| 模型大小 | 13.7 MB |

### 本地验证集成绩

| 类别 | Precision | Recall | F1 |
|------|-----------|--------|-----|
| battery | 1.0000 | 0.9667 | 0.9831 |
| board | 1.0000 | 0.9048 | 0.9500 |
| fire | 1.0000 | 0.9848 | 0.9924 |
| **Mean** | — | — | **0.9751** |

COCO mAP: **0.707**, AP@0.50: **0.865**

### 推理命令

```bash
# 提交到 AI Studio 后的评测入口
python predict.py data.txt result.json 0.05
```

### 关键启示

**标注质量 > 模型规模 > 数据数量。** 本次仅修正了 10 个标注（+0.3% 数据变化），就打破了 v9/v11 长达 4 个版本未能突破的 F1=0.888x 天花板。相比砸大模型（PP-YOLOE+_m, 87MB → F1=0.884）或堆外部数据（D-Fire 3628 张 → F1=0.874），人工检查标注质量是性价比最高的策略。

### 操作流程

```bash
conda activate fire_env1

# 1. 标注转换 (LabelMe → COCO)
python convert_labelme_to_coco.py

# 2. 训练 (PicoDet-M, 200 epoch)
python train.py

# 3. 评估
python evaluate.py

# 4. 导出模型 (→ model/)
python export_model.py

# 5. 阈值扫描 (找最优 score threshold)
python sweep_threshold.py

# 6. 打包 (→ submission.zip)
python package.py
```

---

## 历版本失败经验教训

以下是 12 次迭代中代价最高的失败尝试，按类别整理：

### 🔴 部署细节 Bug（损失最惨重）

| 版本 | 问题 | 分数变化 | 教训 |
|------|------|---------|------|
| v3.0 | **predict.py 路径解析不正确** + TensorRT FP16 精度偏移 + 缺少 padding | 0.882 → **0.475** (-0.407) | 线上环境与本地不同，predict.py 必须严格对齐比赛模板预处理。TensorRT FP16 在本地 V100 和线上 V100 可能有精度差异 |
| v3.2 → v3.0 | 修复上述 bug 后重新训练相同模型 | 0.882 恢复 | 确认是部署代码问题而非模型问题 |

**结论**: 修改 predict.py 后**必须**在本地先用测试数据跑一遍完整流程比对结果，任何预处理差异（resize 方式、padding、归一化）都可能导致分数暴降。

### 🔴 过拟合陷阱

| 版本 | 问题 | 分数变化 | 教训 |
|------|------|---------|------|
| v3.1 | 输入分辨率从 416→512 | 0.882 → **0.490** (-0.392) | 仅 324 张训练图，512 分辨率参数过多导致严重过拟合。FPS 也从 38 降到 23 |
| v10.0 | 200e→300e 长训练 | 0.889 → **0.885** (-0.004) | 最佳 checkpoint 在 E134，之后 mAP 不再增长。更长的训练不一定更好 |

**结论**: 小数据集（~300 张）下不宜增大分辨率。200 epoch 足够收敛，无需盲目加轮数。

### 🔴 数据增强过度

| 版本 | 问题 | 分数变化 | 教训 |
|------|------|---------|------|
| v3.3 | Copy-Paste + Mosaic + Mixup 全开 | 0.882 → **0.871** (-0.011) | 强增强在小数据集上反而降低线上泛化，本地验证集 F1 高不代表线上好 |
| v7.0 | 伪标注补全漏标（928 samples） | 0.882 → **0.844** (-0.038) | v4 模型 (F1=0.9902) 的伪标注引入错误标注，模型学偏。**伪标注质量不够时不如不用** |

**结论**: 数据增强不是越多越好。GridMask + 多尺度是 PicoDet-M 上验证过的最稳定的增强组合。

### 🔴 外部数据无收益

| 版本 | 问题 | 分数变化 | 教训 |
|------|------|---------|------|
| v5.0 | 引入 D-Fire 外部火焰 3628 张 | 0.876 → **0.875** (持平) | 外部火焰数据与比赛场景分布差异大，反而稀释了 VR 场景特征 |
| v5.0 | PicoDet-L (5.80M → 22MB) | 0.876 → **0.875** (持平) | 模型更大但线上无提升 |

**结论**: 对于比赛数据分布高度特定的任务（消防侦察机器狗的视角和场景），外部通用数据很难带来正向收益。

### 🔴 大模型反直觉

| 版本 | 问题 | 分数变化 | 教训 |
|------|------|---------|------|
| v6.0 | PP-YOLOE+_m (23M params, 87MB) | 0.882 → **0.861** (-0.021) | 大规模预训练 (Obj365) + 更大模型反而下跌 |
| v12.0 | PP-YOLOE+_m 再次尝试 (300e, Mosaic+Mixup) | 0.889 → **0.884** (-0.005) | 本地 mAP 创新高 (0.742) 但线上仍然不如 PicoDet-M |

**结论**: 小模型 PicoDet-M (3.46M) 比大模型更适合小数据集。更大的模型=更多参数=更容易过拟合 300+ 张图。"本地 mAP 高 ≠ 线上 F1 高"反复验证。

### 🔴 标注噪声的隐性成本

| 版本 | 问题 | 分数变化 | 教训 |
|------|------|---------|------|
| v6.0 | 数据再平衡（VR 5x + 外部减至 2K） | 0.861 | 改动数据分布对 PP-YOLOE+ 无效 |
| v7.0 | 伪标注补漏 | **0.844** (-0.038) | 最惨重的数据策略失败。模型输出的伪标注会继承模型自身的天花板，用不好的模型去修正自己的错误是死循环 |
| **v13.0** | **人工检查 + 修正 10 个标注** | **0.89144 (+0.003)** | **仅改 10 个标注打破 4 个版本的天花板** |

**结论**: 花时间**人工检查标注质量**远比跑更多实验有效。尤其是 board 类（本数据集最稀缺），每次人工修正都带来可量化的 F1 提升。

### 📊 总资产率图（线上 F1）

```
0.90 ┤
     │                                    🏆 v13 (0.89144)
0.89 ┤                              v9 ───╯  v11
     │                          v2.2 ──╯  ╱  v10
0.88 ┤                        ╱   v12
     │               v4 ─────╯   v5
0.87 ┤             ╱  v3.3
     │      v4 ───╯
0.86 ┤    v6
     │
0.85 ┤  v7
     │
0.50 ┼──────── v3.1 (512 过拟合)
     │
0.48 ┼── v3.0 (predict.py bug)
     │
     └───┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──
         v1 v2 v3 v4 v5 v6 v7 v8 v9 v10v11v12v13
```

### 核心结论

1. **标注质量 > 数据数量 > 模型规模** — 花时间检查标注比堆模型更有效
2. **predict.py 改完必测全流程** — 一个预处理 bug 就能让分数从 0.88 暴降到 0.47
3. **小模型在小数据集上更稳** — PicoDet-M (3.46M) 反复验证优于 PP-YOLOE+_m (23M)
4. **本地高 mAP ≠ 线上高 F1** — 分布偏移客观存在，信任线上分数
5. **不要迷信增强** — GridMask + 多尺度就够了，Mosaic/Mixup/Copy-Paste 加了反而降
6. **不要加外部数据** — 除非分布与比赛数据高度一致，否则稀释特征

---

## 比赛约束

- 模型 ≤ 200MB
- V100 GPU
- FPS ≥ 20（不达标直接 0 分）
- 入口：`python predict.py data.txt result.json`
- 评测指标：Mean F1 @ IoU=0.5（3 类均值），FPS = 2026 / 推理耗时

## 仓库与推送

- **队长仓库**: [Fly333444/software-cup-contest-Online](https://github.com/Fly333444/software-cup-contest-Online)
- **推送远端**: `git@github.com:Fly333444/software-cup-contest-Online.git`
- **本地分支**: v13-picodet-m

```bash
git push git@github.com:Fly333444/software-cup-contest-Online.git v13-picodet-m
```

## 环境配置

```bash
conda create -n fire_env1 python=3.9 -y
conda activate fire_env1
pip install paddlepaddle-gpu==2.6.2
pip install pycocotools numpy opencv-python PyYAML
```

PaddleDetection 已 vendored 在项目中，无需额外安装。

## 快速开始

```bash
conda activate fire_env1

# 标注转换（LabelMe JSON → COCO）
python convert_labelme_to_coco.py

# 训练（支持断点恢复）
python train.py                         # 从头开始
python train.py -r output/N             # 从 checkpoint N 恢复

# 评估
python evaluate.py                      # COCO mAP + per-class F1

# 导出
python export_model.py                  # → model/

# 阈值扫描
python sweep_threshold.py               # global + per-class 网格搜索

# 本地推理测试
python predict.py data.txt result.json 0.05

# 打包提交
python package.py                       # → submission.zip
```

## 项目结构

```
submission_v13/
├── train.py                  # 训练入口，支持 -r 恢复
├── predict.py                # 推理入口（比赛调用）
├── evaluate.py               # COCO mAP + per-class F1 评估
├── export_model.py           # checkpoint → Paddle Inference
├── sweep_threshold.py        # 阈值网格搜索
├── convert_labelme_to_coco.py # LabelMe → COCO
├── package.py                # 打包 submission.zip
├── configs/
│   └── picodet_m_fire_v13.yml   # v13 PicoDet-M 训练配方
├── A_train/
│   ├── Image/                # 405 张原始图片
│   ├── label/                # 405 个 LabelMe 标注
│   ├── train.json            # 324 张训练 COCO 标注
│   └── val.json              # 81 张验证 COCO 标注
├── PaddleDetection/          # Vendored 框架
└── model/                    # 导出模型
    ├── model.pdmodel         # 推理图
    ├── model.pdiparams       # 模型权重
    └── infer_cfg.yml         # 推理配置
```

## 模型架构

```
PicoDet-M (LCNet 1.5x backbone)
  ├── LCPAN (Light Cross-scale Path Aggregation Network)
  ├── PicoHeadV2 (GFL detection head)
  └── 输出: [cls_id, score, x1, y1, x2, y2]
```

训练配置继承链：
```
picodet_m_fire_v13.yml
  ├── coco_detection.yml
  ├── runtime.yml
  ├── picodet_v2.yml
  └── optimizer_300e.yml
```

## predict.py 特性

- **无 TensorRT**（改用 paddle 模式，v13 验证了 FPS=40 已足够）
- batch_size=64 批量推理
- 支持 global threshold：`0.05`
- 支持 per-class threshold：`{"1":0.05, "2":0.10, "3":0.10}`
- 自动从 `infer_cfg.yml` 读取预处理参数

## 注意事项

- Paddle Inference 不支持中文路径
- 部分图片含负样本，配置已开 `allow_empty: true`
- RTX 4060（CC 8.9）本地测试需关 TensorRT（`switch_ir_optim(False)`），提交 V100 时再开启
- 导出需 `combine_params=True` 生成单文件 `.pdiparams`
- **Train on RTX 4060, test on V100** — TensorRT 可能在 V100 上表现不同

## 类别映射

| ID | 类别 | 英文 |
|----|------|------|
| 1 | 电池 | battery |
| 2 | 电路板 | board |
| 3 | 火焰 | fire |

## License

MIT
