# 机器狗 AR 目标检测赛比赛规则说明

## 一、比赛基础要求

### 1. 基础平台

参赛队伍使用的基础机器人平台为：

- 宇树科技 Go2 Edu 版本机器狗

### 2. AR 平台要求

参赛队伍必须在机器狗上挂载组委会指定的 AI 边缘计算板卡：

- X01 板卡

X01 板卡主要负责：

- 增强现实信息的实时渲染
- 虚拟坐标系与真实坐标系的对齐
- 深度学习模型推理

### 3. 全局定位系统要求

参赛队伍需要自行设计并搭建室内全局定位系统，用于获取机器狗的位置与姿态信息。

定位方案不限，可采用：

- ArUco 码定位
- SLAM 定位
- 其他室内全局定位方案

定位系统需要将机器狗的位置和位姿信息，按照组委会规定的数据帧格式发送给 X01 板卡。

> 具体通信协议及 API 接口文档将由组委会另行发布。

### 4. 软件框架要求

参赛系统的软件实现需满足以下要求：

| 模块 | 要求 |
|---|---|
| 图像识别 | 必须基于 PaddlePaddle / 飞桨 |
| 大模型理解与生成 | 必须基于文心大模型 |
| 模型训练与导出 | 必须使用 PaddlePaddle 框架 |
| 推理部署 | 使用 Paddle Inference 静态图格式 |

### 5. 网络环境要求

比赛现场提供基础无线局域网，用于调用大模型云端 API。

参赛队伍也可以自行准备：

- 5G 路由器
- 4G 路由器
- 其他稳定通信设备

以保障比赛现场网络通信稳定。

---

## 二、线上初赛安排

### 1. 赛题任务

本赛题聚焦于目标检测任务。

选手需要基于 PaddlePaddle 框架训练并部署目标检测模型，对测试图片中的目标进行检测与定位。

检测类别共 3 类：

| 类别编号 | 类别名称 | 中文描述 |
|---|---|---|
| 1 | battery | 电池 |
| 2 | board | 指示牌 / 电路板 |
| 3 | fire | 火焰 |

> 注意：提交结果中的 `type` 字段必须使用 1-indexed 编号，即 battery=1，board=2，fire=3。

### 2. 榜单时间安排

| 阶段 | 时间 | 说明 |
|---|---|---|
| A 榜 | 5 月 25 日 — 6 月 20 日 | 开放测试，可反复提交，支持实时评测反馈 |
| B 榜 | 6 月 21 日 — 7 月 1 日 | 混合测试，包含未公开数据，提交次数有限 |

### 3. A 榜说明

A 榜提供公开测试数据集与实时评测反馈。

参赛队伍可以反复提交模型，用于：

- 快速验证模型效果
- 进行参数调优
- 迭代训练策略
- 检查推理脚本是否符合要求

### 4. B 榜说明

B 榜采用混合数据集进行测评，其中包含部分全新未公开测试数据。

B 榜的作用包括：

- 检验模型真实泛化能力
- 防止针对 A 榜数据过拟合
- 作为队伍二次训练和迁移优化的补充依据

B 榜设置有限次提交机制，参赛队伍需谨慎提交。

---

## 三、数据集说明

### 1. 数据集概况

训练数据集名称为：

```text
A_train
```

数据集来源于 VR 环境的视频帧图像，图像分辨率为：

```text
1920 × 1080
```

数据集基本信息如下：

| 项目 | 内容 |
|---|---|
| 数据集名称 | A_train |
| 数据规模 | 405 张图片 + 405 个标注文件 |
| 图像分辨率 | 1920 × 1080 |
| 图像格式 | JPEG（.jpg） |
| 标注格式 | LabelMe JSON（.json） |
| 检测类别 | battery / board / fire |

### 2. 数据集下载地址

```text
链接：https://pan.baidu.com/s/1tCCZMUsj221E4WCq-IqcoQ?pwd=1b72
提取码：1b72
```

### 3. 数据集目录结构

```text
A_train/
├── Image/
│   ├── frame_00002.jpg
│   ├── frame_00006.jpg
│   └── ...
└── label/
    ├── frame_00002.json
    ├── frame_00006.json
    └── ...
```

其中：

- `Image/` 存放原始图片
- `label/` 存放对应标注文件
- 图片与标注文件一一对应
- 图片和标注文件文件名相同，仅后缀不同

例如：

```text
Image/frame_00002.jpg
label/frame_00002.json
```

### 4. 类别标注数量

| 类别编号 | 类别名称 | 中文描述 | 标注框数 |
|---|---|---|---|
| 1 | battery | 电池 | 126 |
| 2 | board | 电路板 / 指示牌 | 92 |
| 3 | fire | 火焰 | 712 |

> 部分图片中可能不含任何目标，属于正常样本。

---

## 四、标注格式说明

### 1. LabelMe JSON 格式

每张图片对应一个 LabelMe 格式的 JSON 标注文件。

示例：

```json
{
  "version": "4.0.0-beta.5",
  "shapes": [
    {
      "label": "fire",
      "shape_type": "rectangle",
      "points": [
        [120.5, 45.0],
        [320.5, 45.0],
        [320.5, 225.0],
        [120.5, 225.0]
      ]
    },
    {
      "label": "battery",
      "shape_type": "rectangle",
      "points": [
        [800.0, 300.0],
        [950.0, 300.0],
        [950.0, 480.0],
        [800.0, 480.0]
      ]
    }
  ],
  "imagePath": "frame_00000.jpg",
  "imageHeight": 1080,
  "imageWidth": 1920
}
```

### 2. 关键字段说明

| 字段 | 说明 |
|---|---|
| `shapes[].label` | 目标类别名称，可为 `battery`、`board`、`fire` |
| `shapes[].shape_type` | 标注形状类型，均为 `rectangle` |
| `shapes[].points` | 矩形框 4 个顶点坐标 |
| `imageHeight` | 图像高度，通常为 1080 |
| `imageWidth` | 图像宽度，通常为 1920 |

### 3. 从 points 提取检测框

LabelMe 中的矩形框由 4 个点表示，需要转换为检测任务常用的 `xywh` 格式。

转换方法如下：

```python
points = shape["points"]

x = min(p[0] for p in points)
y = min(p[1] for p in points)
w = max(p[0] for p in points) - x
h = max(p[1] for p in points) - y
```

其中：

| 字段 | 含义 |
|---|---|
| `x` | 检测框左上角横坐标 |
| `y` | 检测框左上角纵坐标 |
| `w` | 检测框宽度 |
| `h` | 检测框高度 |

---

## 五、模型训练与导出要求

### 1. 框架要求

参赛模型必须满足：

- 使用 PaddlePaddle 框架训练
- 使用 PaddlePaddle / PaddleDetection 进行目标检测模型开发
- 导出为 Paddle Inference 静态图格式

不接受其他深度学习框架训练或导出的模型。

### 2. 推荐工具

推荐使用 PaddleDetection 官方目标检测套件。

可选择的模型包括但不限于：

- PP-YOLOE
- PP-YOLOE-S
- PicoDet
- YOLO 系列 PaddleDetection 实现
- 其他满足速度和精度要求的 PaddleDetection 检测模型

### 3. 模型导出格式

最终提交的模型必须包含以下文件：

```text
model/
├── infer_cfg.yml
├── model.pdmodel
└── model.pdiparams
```

文件说明：

| 文件 | 说明 |
|---|---|
| `infer_cfg.yml` | 推理配置文件 |
| `model.pdmodel` | 模型结构文件 |
| `model.pdiparams` | 模型权重文件 |

### 4. PaddleDetection 导出命令参考

```bash
python tools/export_model.py \
    -c configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml \
    --output_dir=./output_inference \
    -o weights=best_model.pdparams
```

导出完成后，将导出目录中的以下文件放入 `model/` 目录：

```text
infer_cfg.yml
model.pdmodel
model.pdiparams
```

### 5. 模型大小限制

```text
model/ 目录总大小 ≤ 200 MB
```

如果模型大小超过限制，可以考虑：

- 使用轻量模型
- 减小输入尺寸
- 使用更小的 backbone
- 清理无关文件
- 只保留推理所需文件

---

## 六、提交包结构要求

### 1. 提交文件格式

参赛队伍需将推理代码和模型文件打包为：

```text
submission.zip
```

### 2. 标准目录结构

压缩包内部结构必须如下：

```text
submission.zip
├── predict.py
├── model/
│   ├── infer_cfg.yml
│   ├── model.pdmodel
│   └── model.pdiparams
└── PaddleDetection/
    └── deploy/
        └── python/
            ├── preprocess.py
            ├── utils.py
            └── keypoint_preprocess.py
```

### 3. 文件说明

| 文件或目录 | 是否必须 | 说明 |
|---|---|---|
| `predict.py` | 必须 | 推理入口脚本 |
| `model/` | 必须 | 模型文件目录，总大小不超过 200MB |
| `model/infer_cfg.yml` | 必须 | 模型推理配置文件 |
| `model/model.pdmodel` | 必须 | 模型结构文件 |
| `model/model.pdiparams` | 必须 | 模型权重文件 |
| `PaddleDetection/` | 可选 | 使用官方部署代码时需要携带 |

> 如果 `predict.py` 中不使用 PaddleDetection 官方部署代码，而是自行实现预处理和推理逻辑，可以不包含 `PaddleDetection/` 目录。

---

## 七、推理脚本规范

### 1. 评测系统调用方式

后台评测系统会使用如下命令调用参赛队伍的推理脚本：

```bash
python predict.py <data_txt> <result_json>
```

其中：

| 参数 | 说明 |
|---|---|
| `data_txt` | 测试图片路径列表文件，每行一条图片路径 |
| `result_json` | 推理结果输出路径 |

示例：

```bash
python predict.py data.txt result.json
```

### 2. data.txt 格式

`data.txt` 中每一行是一张测试图片路径，例如：

```text
test/Image/frame_00000.jpg
test/Image/frame_00001.jpg
test/Image/frame_00002.jpg
```

### 3. predict.py 必须满足的要求

`predict.py` 需要满足以下规范：

1. 从 `sys.argv[1]` 读取图片路径列表文件；
2. 从 `sys.argv[2]` 获取输出 JSON 文件路径；
3. 完成全部图片推理；
4. 将检测结果写入 `sys.argv[2]` 指定路径；
5. 输出文件必须为合法 JSON；
6. 脚本正常退出，返回码为 0。

---

## 八、输出结果格式

### 1. result.json 总体格式

推理结果必须保存为 JSON 文件，格式如下：

```json
{
  "result": [
    {
      "image_id": "frame_00000",
      "type": 3,
      "x": 120.5,
      "y": 45.0,
      "width": 200.0,
      "height": 180.0,
      "segmentation": []
    }
  ]
}
```

### 2. 字段说明

| 字段 | 类型 | 要求 |
|---|---|---|
| `image_id` | string | 图片文件名，不含扩展名，如 `frame_00000` |
| `type` | int | 类别编号，1-indexed，1=battery，2=board，3=fire |
| `x` | float | 检测框左上角横坐标，单位为像素 |
| `y` | float | 检测框左上角纵坐标，单位为像素 |
| `width` | float | 检测框宽度，单位为像素 |
| `height` | float | 检测框高度，单位为像素 |
| `segmentation` | list | 本赛题无需分割，填空列表 `[]` |

### 3. 无目标图片处理

如果某张图片中没有检测到目标，则不需要在 `result` 中为该图片添加条目。

如果所有图片都没有检测结果，可以输出：

```json
{
  "result": []
}
```

### 4. 类别编号要求

提交结果中的类别编号必须如下：

| type | 类别 |
|---|---|
| 1 | battery |
| 2 | board |
| 3 | fire |

注意：

- `type` 必须是整数；
- 类别编号从 1 开始；
- 如果模型输出类别从 0 开始，需要在生成结果时加 1。

---

## 九、评测指标

### 1. 综合得分

综合得分为 3 个类别 F1 Score 的均值，满分为 1.0。

计算公式为：

```math
score = \frac{1}{3} \sum_{c=1}^{3} F1_c
```

其中：

```math
F1_c = \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}
```

| 符号 | 含义 |
|---|---|
| `P_c` | 第 c 类的 Precision |
| `R_c` | 第 c 类的 Recall |
| `F1_c` | 第 c 类的 F1 Score |

### 2. IoU 匹配阈值

预测框与真实框的 IoU 需要满足：

```text
IoU ≥ 0.5
```

才算成功命中。

### 3. FPS 要求

模型推理速度必须满足：

```text
FPS ≥ 20
```

否则综合得分将被强制记为 0。

### 4. FPS 计算方式

```math
FPS = \frac{2026}{推理总耗时（秒）}
```

其中：

- 2026 表示评测时测试图片数量；
- 推理总耗时包括模型读取、预处理、推理、后处理和结果写入等整体耗时；
- 如果 FPS 小于 20，则即使检测精度较高，最终分数也会被置为 0。

---

## 十、限制条件汇总

| 项目 | 要求 |
|---|---|
| 提交文件格式 | `.zip` |
| 推理框架 | PaddlePaddle |
| 模型导出格式 | Paddle Inference 静态图 |
| 模型文件 | `model.pdmodel` + `model.pdiparams` |
| 配置文件 | `infer_cfg.yml` |
| `model/` 目录大小 | ≤ 200MB |
| 推理速度 | FPS ≥ 20 |
| 检测类别数 | 3 类 |
| 输出格式 | JSON |
| 分割字段 | `segmentation` 填 `[]` |

---

## 十一、常见失败原因与解决方法

| 错误信息或现象 | 可能原因 | 解决方法 |
|---|---|---|
| `required file missing: predict.py` | 压缩包内没有 `predict.py` | 检查 zip 根目录是否包含该文件 |
| `required file missing: model/` | 缺少 `model/` 目录 | 检查目录名是否正确 |
| `model/ directory is empty` | `model/` 内没有模型文件 | 放入 `infer_cfg.yml`、`model.pdmodel`、`model.pdiparams` |
| `model/ size exceeds 200MB` | 模型目录超过大小限制 | 更换轻量模型或清理无关文件 |
| `predict.py exited with non-zero code` | 推理脚本运行报错 | 本地先运行 `python predict.py data.txt result.json` 测试 |
| `predict.py did not produce result.json` | 没有生成输出文件 | 确认写入路径使用的是 `sys.argv[2]` |
| `result.json missing "result" key` | JSON 顶层缺少 `result` 字段 | 顶层必须是 `{"result": [...]}` |
| `result.json item missing keys` | 单条检测结果字段不完整 | 检查是否包含全部必要字段 |
| `type` 编号错误 | 类别编号没有从 1 开始 | battery=1，board=2，fire=3 |
| score = 0 | FPS 小于 20 | 使用 GPU 推理、降低输入尺寸、换轻量模型 |
| 检测框位置异常 | 坐标格式转换错误 | 确认输出为 `x, y, width, height` |
| 输出结果为空 | 阈值过高或模型未正确加载 | 降低置信度阈值，检查模型路径和配置文件 |

---

## 十二、本地测试建议

提交前建议在本地进行完整测试。

### 1. 准备测试图片列表

新建 `data.txt`：

```text
test/Image/frame_00000.jpg
test/Image/frame_00001.jpg
test/Image/frame_00002.jpg
```

### 2. 运行推理脚本

```bash
python predict.py data.txt result.json
```

### 3. 检查输出文件

确认生成：

```text
result.json
```

并检查其格式是否为：

```json
{
  "result": []
}
```

或：

```json
{
  "result": [
    {
      "image_id": "frame_00000",
      "type": 3,
      "x": 120.5,
      "y": 45.0,
      "width": 200.0,
      "height": 180.0,
      "segmentation": []
    }
  ]
}
```

### 4. 检查压缩包结构

打包后应确保 `submission.zip` 解压后的根目录结构为：

```text
submission/
├── predict.py
├── model/
│   ├── infer_cfg.yml
│   ├── model.pdmodel
│   └── model.pdiparams
└── PaddleDetection/
    └── deploy/
        └── python/
            ├── preprocess.py
            ├── utils.py
            └── keypoint_preprocess.py
```

如果不使用 PaddleDetection 官方部署代码，也可以是：

```text
submission/
├── predict.py
└── model/
    ├── infer_cfg.yml
    ├── model.pdmodel
    └── model.pdiparams
```

---

## 十三、提交前检查清单

提交前请逐项确认：

- [ ] `predict.py` 位于 zip 根目录；
- [ ] `model/` 位于 zip 根目录；
- [ ] `model/` 中包含 `infer_cfg.yml`；
- [ ] `model/` 中包含 `model.pdmodel`；
- [ ] `model/` 中包含 `model.pdiparams`；
- [ ] `model/` 目录总大小不超过 200MB；
- [ ] `predict.py` 能从 `sys.argv[1]` 读取图片列表；
- [ ] `predict.py` 能将结果写入 `sys.argv[2]`；
- [ ] 输出 JSON 顶层包含 `result` 字段；
- [ ] 每个检测结果包含 `image_id`、`type`、`x`、`y`、`width`、`height`、`segmentation`；
- [ ] `segmentation` 字段为 `[]`；
- [ ] 类别编号为 battery=1，board=2，fire=3；
- [ ] 检测框格式为 `x, y, width, height`；
- [ ] 本地运行 `python predict.py data.txt result.json` 不报错；
- [ ] 推理速度满足 FPS ≥ 20；
- [ ] 最终文件格式为 `.zip`。

---

## 十四、核心注意事项

1. 评测系统只关心 `predict.py` 是否能正确读取输入并生成指定格式的 `result.json`。
2. 模型必须是 PaddlePaddle 导出的静态图推理模型。
3. `model/` 目录大小不能超过 200MB。
4. `type` 字段必须使用 1、2、3 三个整数编号。
5. `segmentation` 字段虽然不用做分割，但必须保留，且填 `[]`。
6. 如果 FPS 小于 20，最终得分直接为 0。
7. A 榜可以多次尝试，适合调试；B 榜提交次数有限，需谨慎提交。
