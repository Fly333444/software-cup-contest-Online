# 火灾检测标注工具 🔥

用于对火灾检测数据集（电池起火、电路板、火焰）进行图像标注的 Web 工具。

## 目录结构

```
annotation_tool/
├── A_train/                         # 标注好的数据集
│   ├── Image/                       #   — 图片（405 张 .jpg）
│   ├── label/                       #   — LabelMe JSON 标注文件
│   ├── train.json                   #   — COCO 格式训练集（80%）
│   └── val.json                     #   — COCO 格式验证集（20%）
├── annotation_tool.py               # Flask Web 标注工具
├── convert_labelme_to_coco.py       # 标注 → COCO 格式转换
├── environment.yml                  # Conda 环境配置
├── start.bat                        # 一键启动（Windows）
├── start_convert.bat                # 一键格式转换（Windows）
├── .gitignore
└── README.md
```

## 环境要求

- Python 3.9+
- Flask
- OpenCV、numpy（仅转换脚本需要）

### 已有环境（推荐）

如果你在本机上已经有 `fire_env1` 环境，直接双击 `start.bat` 即可启动。

### 新建环境

```bash
# 用 environment.yml 创建（推荐）
conda env create -f environment.yml -n fire_env

# 或者手动创建
conda create -n fire_env python=3.9 -y
conda activate fire_env
pip install flask numpy opencv-python pyyaml pycocotools
```

## 快速开始

### 启动标注工具

**方式一（推荐）—— 双击 start.bat**

在文件管理器中双击 `start.bat`，会自动激活环境并启动服务，然后在浏览器打开 `http://localhost:5001`。

**方式二—— 命令行**

```bash
conda activate fire_env1       # 或 fire_env
python annotation_tool.py
# 打开 http://localhost:5001
```

### 标注操作

| 操作 | 功能 |
|------|------|
| **拖拽** | 用当前类别画框 |
| **右键** | 选中已有标注框 |
| **双击** | 修改选中框的类别 |
| **Delete** | 删除选中的框 |
| **Ctrl+S** | 保存到磁盘 |
| **1 / 2 / 3** | 切换类别（电池 / 电路板 / 火焰） |
| **← →** | 上一张 / 下一张图片 |

### 类别

| ID | 类别 | 颜色 | 说明 |
|----|------|------|------|
| 1 | battery | 绿色 | 电池（含锂电池、蓄电池） |
| 2 | board | 红色 | 电路板（PCB、线路板） |
| 3 | fire | 橙色 | 火焰 |

### 导出 COCO 格式

**方式一：** 双击 `start_convert.bat`

**方式二：** 命令行

```bash
conda activate fire_env1
python convert_labelme_to_coco.py
```

会在 `A_train/` 下生成：
- `train.json` — 训练集（80%，约 324 张）
- `val.json` — 验证集（20%，约 81 张）

## 标注流程

```
1. 启动工具 → 浏览器打开 http://localhost:5001
2. 左侧筛选 "no battery / no board / no fire" 查看缺失类别
3. 逐张检查并补充标注
4. 标注自动保存，底部状态栏显示 "saved"
5. 全部完成后关闭即可
```

## 已标注数据

`A_train/` 目录下包含 405 张图片的标注结果（LabelMe JSON 格式），是已经由人工标注完成并验证过的数据集。

- 图片总数：405 张
- 标注文件：405 个（每张图片对应一个 .json）
- 已生成的 COCO 格式：`train.json`（训练集）和 `val.json`（验证集）
- 数据用途：火灾检测模型训练（如 PaddleDetection、YOLO）

## 常见问题

**端口被占用？**
修改 `annotation_tool.py` 最后一行 `port=5001` 为其他端口，或在浏览器中输入 `http://localhost:其他端口`。

**图片不显示？**
确保 `A_train/Image/` 目录下有图片文件（.jpg / .png），文件名与标注文件匹配。

**想重新转换 COCO 格式？**
运行 `start_convert.bat` 或 `python convert_labelme_to_coco.py`，会重新按 80/20 随机拆分并生成 COCO JSON。

**没有 `fire_env1` 环境？**
见上方「新建环境」部分，也可以用 `environment.yml` 创建同名环境。
