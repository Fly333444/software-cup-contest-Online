# B榜 火检测提交项目

---

## 一、比赛规则与理解

### 1.1 任务

基于 PaddlePaddle 训练目标检测模型，对 640×480 合成渲染图片中**面积最大的一处火焰 (firebig)** 进行定位，每张图输出 1 个框。

### 1.2 核心约束

| 项目 | 内容 |
|------|------|
| 类别 | 单类 `firebig`，类型编号 type=1 |
| 每图输出 | 1 个框（最大的 firebig），无火则不出 |
| 图像 | 640×480，RGB JPEG |
| 评测 | F1 Score @ IoU=0.5 |
| 速度 | FPS ≥ 20（4544 张 ≤ 227 秒），不达标分数强制为 0 |
| 推理框架 | Paddle Inference 静态图 |
| 提交 | `submission.zip`（predict.py + model/，≤200MB） |

### 1.3 评测调用

```bash
python predict.py <data_txt> <result_json> [threshold]
```

`data_txt` 每行一张图片路径，`result_json` 是输出文件。

### 1.4 输出 JSON

```json
{
  "result": [
    {
      "image_id": "frame_00000",
      "type": 1,
      "x": 120.5, "y": 45.0,
      "width": 200.0, "height": 180.0,
      "segmentation": []
    }
  ]
}
```

`x, y` 是框左上角坐标，`width, height` 是宽高，`segmentation` 填空列表。

### 1.5 评测公式

```
F1 = 2 × P × R / (P + R)
匹配: IoU ≥ 0.5，贪心一对一匹配
```

### 1.6 关键理解

这个比赛有陷阱：**"选面积最大的框"不等于在所有检测结果里用 max_area**。模型会输出大量低分假框（score 0.02~0.1），这些假框面积往往巨大（是全图的 3~9 倍），纯面积策略必然被欺骗。实验数据：

| 策略 | B_data IoU≥0.5 | sample100 IoU≥0.5 |
|------|---------------|-------------------|
| max_area（纯面积） | 5.6% | 3.0% |
| max_score（置信度优先） | 98.0% | 96.0% |

所以后处理逻辑应该：**先用置信度选择出所有可信的火焰框，再在可信框中选面积最大的**。

---

## 二、代码框架

```
B_v1/
├── split_dataset.py        # 1. 数据准备
├── train.py                # 2. 训练
├── export_model.py         # 3. 模型导出
├── predict.py              # 4. 推理入口（提交核心）
├── package.py              # 5. 打包提交
│
├── evaluate.py             # 辅助：本地评估
├── sweep_threshold.py      # 辅助：阈值搜索
│
├── configs/
│   ├── faster_rcnn_r50_fpn_firebig_b01.yml  # 当前训练配置（两阶段，640×480）
│   ├── picodet_l_firebig_b01.yml             # 备选（单阶段 PicoDet-L）
│   └── ppyoloe_plus_m_fire_b01.yml           # 备选（PP-YOLOE+_m）
│
├── B_data/                 # 训练数据
│   ├── *.jpg               # 2827张 640×480 图片
│   ├── annotations/        # 2736个 Pascal VOC XML
│   ├── ImageSets/Main/train.txt
│   └── label_list.txt
│
├── model/                  # 导出模型（≤200MB）
│   ├── infer_cfg.yml
│   ├── model.pdmodel
│   └── model.pdiparams
│
├── output/                 # 训练 checkpoint
├── PaddleDetection/        # 框架（最小部署）
├── PaddleDetection_full/   # 框架（完整训练）
├── annotation_tool/        # 标注复核工具
├── pretrain/               # 预训练权重
├── 评测集去氛围纯净版抽样sample100/  # 官方参考标注
└── submission_template_firedetect/  # 官方提交模板
```

---

## 三、逐个文件详解

### 3.1 split_dataset.py — 数据准备

**作用**：扫描 `B_data/annotations/` 下所有 Pascal VOC XML，生成训练所需的列表文件。

**输入**：`B_data/annotations/*.xml`

**输出**：
- `B_data/ImageSets/Main/train.txt`：每行 `frame_XXXXXX.jpg annotations/frame_XXXXXX.xml`（2736 行）
- `B_data/label_list.txt`：`firebig`

**逻辑**：
1. 用 `defusedxml.ElementTree` 解析每个 XML
2. 统计 fire/firebig 标注框数量
3. 把所有图片 ID 写入 train.txt（全量训练，无验证集划分）
4. 写 label_list.txt

**使用**：训练前必须跑一次。

### 数据流：从标注到模型学习

```
你的手工标注                    split_dataset.py              VOCDataSet读取               Faster R-CNN训练
──────────────────────────────────────────────────────────────────────────────────────────────────
B_data/annotations/         →  scan→ImageSets/Main/       →  每行：图片.jpg 标注.xml  →  逐epoch学习
  frame_000000.xml                train.txt (2736行)
  frame_000001.xml            B_data/label_list.txt       
    <object>                   "firebig"                  
    <name>firebig</name>                                  
    <bndbox>                                              
      <xmin>233</xmin>                                    
      <ymin>91</ymin>                                     
      <xmax>337</xmax>                                    
      <ymax>196</ymax>                                    
    </bndbox>                                             
    </object>
```

**VOCDataSet 读取逻辑**：
1. 逐行读 `train.txt`，每行是 `frame_XXXXXX.jpg annotations/frame_XXXXXX.xml`
2. 从 `B_data/` 目录加载 jpg 图片（640×480 RGB）
3. 解析 XML，找 `<object>` 标签中 `<name>` 与 label_list 匹配的框
4. label_list = `[firebig]`，所以**只取 firebig 框**，fire 被忽略
5. 每张图返回：`[image(3,H,W), gt_bbox(N×4), gt_class(N×1)]`，其中 N 是该图 firebig 框的数量（你标注的都是 1）

**模型实际学到的（两阶段）**：
- **第一阶段（RPN）**：扫描特征图，提候选区域 — 纯回归定位，"图中哪些位置可能有目标？"
- **第二阶段（ROI Head）**：对每个候选区域分类+精修 — "这个区域是 firebig 吗？框应该多精确？"
- 输入：640×480 原图（不缩放不变形）→ Resize [480,640] 等价于 identity → 直接喂给网络
- 监督信号：你手工标注的 2736 个 firebig 框坐标
- 损失函数：
  - **RPN 分类损失**（Binary Cross Entropy）：anchor 是前景还是背景？
  - **RPN 回归损失**（Smooth L1）：anchor 要怎样调整才能框住目标？
  - **ROI 分类损失**（Cross Entropy）：候选区域是 firebig 还是背景？
  - **ROI 回归损失**（Smooth L1）：候选框要怎样精修？
- 24 个 epoch 反复看这 2736 张图，RPN 先学会提候选区，ROI Head 再学会精细分类定位

---

### 3.2 train.py — 训练入口

**作用**：启动 PaddleDetection 训练。

**配置**：默认加载 `configs/faster_rcnn_r50_fpn_firebig_b01.yml`（Faster R-CNN R50 FPN，两阶段，640×480 原生输入）

**逻辑**：
1. 启动时清理 GPU 孤儿进程（用 `taskkill` 杀掉其他 PaddleDetection 训练进程，释放 GPU 显存）
2. 用 `nvidia-smi` 检查 GPU 显存状态
3. 用 PaddleDetection 的 `ArgsParser` 解析命令行参数：
   - 不带 `--config` 时默认用 `configs/faster_rcnn_r50_fpn_firebig_b01.yml`
   - `-r output/N` 从指定 epoch 的 checkpoint 恢复训练
   - `--config` 可切换到 `picodet_l_firebig_b01.yml` 等其他配置
4. 加载配置文件 → `merge_config` 合并命令行参数
5. 如果 `B_data/ImageSets/Main/train.txt` 存在，用它覆盖 `TrainDataset.anno_path`
6. `EvalDataset` 也指向 train.txt（全量训练，eval 只用来监控 loss）
7. 注册 `atexit` 和 signal handler，退出时清 CUDA 缓存
8. 如果有 `pretrain_weights`：
   - URL 开头 → 自动下载到 `pretrain/` 目录
   - 本地路径 → 直接加载
   - 否则报错退出
9. 调用 `Trainer.train(do_eval)` 开始训练，每 snapshot_epoch=12 保存一次 checkpoint
10. 输出到 `output/<config文件名>/`（如 `output/faster_rcnn_r50_fpn_firebig_b01/`）

**使用**：
```bash
python train.py                    # 默认配置全量训练
python train.py -r output/11       # 从 epoch 11 恢复
python train.py --config configs/picodet_l_firebig_b01.yml  # 换回 PicoDet
```

---

### 3.3 export_model.py — 模型导出

**作用**：把训练好的 `.pdparams` 权重转为 Paddle Inference 静态图。

**输入**：`output/<config_name>/` 下的 checkpoint（优先找 `best_model.pdparams`）

**输出**：
- `model/model.pdmodel`（模型结构 + 后处理 decode/NMS 内嵌）
- `model/model.pdiparams`（模型权重，ResNet50+FPN ~120MB）
- `model/infer_cfg.yml`（推理配置，自动生成）
- 预估总大小 ~130MB，低于 200MB 限制

**逻辑**：
1. 在 `output/<config_name>/` 找 `best_model.pdparams` 或 `model_final.pdparams`
2. 找不到回退到 `output/` 根目录
3. 覆盖 NMS 参数（score_threshold=0.025, nms_threshold=0.6）— 兼容 BBoxPostProcess 和 PicoHeadV2
4. 调用 `trainer.export(output_dir)` 导出（`export.post_process=true` 把 decode+NMS 嵌入静态图）
5. PaddleDetection 导出时会创建子目录，脚本自动把文件移到 `model/` 根目录
6. 打印每个文件大小，校验总大小

**使用**：训练完成后跑一次。

---

### 3.4 predict.py — 推理入口（提交核心）

**作用**：评测系统直接调用这个脚本。**完全自包含**，不依赖 PaddleDetection。

**调用**：`python predict.py <data_txt> <result_json> [threshold]`

**流程**：

1. **预处理**（内建，无需 import PaddleDetection，从 `infer_cfg.yml` 动态读取配置）
   - `decode_image`：读取 JPEG → BGR→RGB，检查损坏图片
   - `Resize`：640×480 → 640×480（target_size=[480,640]，等价于不做任何缩放变形）
   - `NormalizeImage`：`pixel/255 → (pixel - mean)/std`（ImageNet 均值标准差）
   - `Permute`：HWC → CHW

2. **模型加载**
   - GPU 优先：`config.enable_use_gpu(500, 0)`
   - 无 GPU 自动回退：`config.disable_gpu()`
   - 关闭 IR 优化（兼容 RTX 4060 CUDNN）

3. **批量推理**（batch_size=1，Faster R-CNN 固定输入尺寸，导出后不支持动态 batch）
   - 图片 resize 到 480×640
   - Paddle Inference 前向传播 → 输出 bboxes + boxes_num（后处理已内嵌在静态图中）

4. **后处理（核心算法）**
   ```
   对每张图：
     a. 收集所有 score ≥ threshold 的框（默认 threshold=0.3）
     b. 过滤宽高 ≤0 的无效框
     c. 如果无候选框 → 跳过该图（不输出）
     d. 按分数降序排列，取最高分
     e. 如果最高分 < 0.15 → 跳过（不可靠检测）
     f. 从与最高分相差 10% 以内的框中选面积最大的
     g. 截断坐标到图像范围内（0~640, 0~480）
     h. type 恒为 1（firebig）
   ```

5. **输出**：JSON 写入 `sys.argv[2]`

**关键修复历史**：
- Bug：`max()` on empty list → 无检出时崩溃 → 加 `continue`
- Bug：GPU 不可用崩溃 → 加 try/except 回退 CPU
- Bug：损坏图片崩溃 → 加 `im is None` 检查
- 算法：max_area → max_score+面积回退（IoU 从 3% → 96%）

---

### 3.5 package.py — 打包提交

**作用**：生成 `submission.zip`。

**校验项**：
- `predict.py` 存在
- `model/` 目录存在
- `model/infer_cfg.yml`, `model.pdmodel`, `model.pdiparams` 都存在
- 模型总大小 ≤ 200MB

**打包**：`predict.py` + 三个 model 文件 → `submission.zip`（当前预估 ~130MB）

**使用**：导出完成后跑一次。

---

### 3.6 evaluate.py — 本地评估

**作用**：用 PaddleDetection Trainer 在验证集上评估，计算逐类 F1。

**逻辑**：
1. 加载 checkpoint 权重
2. 调用 `trainer.evaluate()`（VOC 格式）
3. 如果存在 COCO JSON 格式的 gt/dt，用 pycocotools 计算 fire + firebig 的逐类 Precision/Recall/F1 @ IoU=0.50

**使用**：全量训练模式下主要用于在有验证集（如 fold 数据）时评估。

---

### 3.7 sweep_threshold.py — 阈值搜索

**作用**：暴力搜索最佳置信度阈值。

**逻辑**：
1. 用 Paddle Inference 加载模型，在 `val.json`（COCO 格式）上推理
2. 遍历 threshold ∈ [0.05, 0.10, ..., 0.50]
3. 每个阈值：过滤低分框 → 生成临时 COCO 结果 → pycocotools 计算 Mean F1
4. 找最佳阈值，打印 fire/firebig 各自表现

**使用**：有 COCO 格式验证集时，用来调参。

---


## 四、配置文件

### faster_rcnn_r50_fpn_firebig_b01.yml（当前）

| 参数 | 值 |
|------|-----|
| 架构 | **Faster R-CNN（两阶段）**：RPN 先提候选区域（纯回归定位）→ ROI Head 再分类+精修 |
| 骨干网络 | ResNet50 + FPN（neck 特征融合） |
| num_classes | 1（firebig） |
| 输入 | **640×480 原生**，keep_ratio=false（等价于 identity，不变形不缩放） |
| 训练增强 | Resize [480,640] + RandomFlip（无 RandomCrop/Distort，保持全图信息） |
| 归一化 | ImageNet（mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]） |
| 优化器 | Momentum + PiecewiseDecay（base_lr=0.01, milestones=[16,22]）+ LinearWarmup（500 step） |
| EMA | 开启 |
| epoch | 24 |
| batch_size | 1（Faster R-CNN 内存占用较大） |
| NMS | score_threshold=0.025, nms_threshold=0.6, keep_top_k=100 |
| RPN | anchor_sizes=[32,64,128,256,512], strides=[4,8,16,32,64] |
| 预训练 | COCO pretrained ResNet50 → 自动下载到 pretrain/ |

### picodet_l_firebig_b01.yml（备选：单阶段）

PicoDet-L（~5.8M params），416×416 硬压输入，单阶段耦合头同时分类+回归。仅作回退保留。

### ppyoloe_plus_m_fire_b01.yml（备选：单阶段）

PP-YOLOE+_m（~23M params），640×480 原生输入，Object365 预训练。标注中同时有 fire+firebig 时使用，num_classes=2。

---


## 五、模型信息

| 项目 | 内容 |
|------|------|
| 模型 | Faster R-CNN R50 FPN（~41M params，含 RPN + ROI Head + backbone） |
| 导出大小 | 预估 ~130MB（低于 200MB 限制） |
| 训练数据 | 2736 张手动标注（91 张待标注），2827 张图片 |
| 训练 epoch | 24（ResNet50 从 COCO 预训练启动，收敛较快） |
| 推理输入 | 640×480 RGB（不缩放，不变形） |
| 推理速度 | 预计 >20 FPS（V100），需评测验证 |
| 两阶段流程 | RPN（定位候选区）→ ROI Pooling（区域特征提取）→ Classification + BBox Reg（分类精修） |

---

## 六、快速命令

```bash
conda activate fire_env1

python split_dataset.py                              # 数据准备
python train.py                                      # 训练
python export_model.py                               # 导出
python predict.py test_data.txt test_result.json 0.3 # 测试
python package.py                                    # 打包
```
