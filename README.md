# Image Flow — 图像处理流式编辑器

基于 PyQt5 的节点式图像处理编辑器，通过拖拽算子、连线构建处理流程，支持传统图像处理与 ONNX 深度学习推理。

## 系统要求

- Python 3.7+
- Windows / Linux / macOS
- （可选）NVIDIA GPU + CUDA 11.x + cuDNN，用于深度学习 GPU 推理

## 安装

```bash
pip install -r requirements.txt
```

如需 GPU 推理，将 `onnxruntime` 替换为 `onnxruntime-gpu`。

## 启动

```bash
python main.py
```

## 界面布局

```
+--------+------------------------+----------+
|  左侧  |      中间（画布）       |   右侧   |
| 算子   |                        |  属性    |
| 列表   |------------------------|  面板    |
|        |    执行日志 (底部)      |          |
+--------+------------------------+----------+
```

- **左侧** — 算子列表，按大类折叠分组，拖拽或双击添加节点
- **中间** — 节点画布（上）与执行日志（下）
  - 鼠标左键拖拽连线，右键删除节点/连线
  - 中键拖拽平移画布，滚轮缩放
  - 日志区域实时显示算子运行信息和错误
- **右侧** — 属性面板，选中节点后配置算法与参数

## 操作流程

1. 从左侧算子列表**拖拽**（或双击）算子到画布
2. 点击算子**输出端口**（蓝色圆点）拖拽到另一算子的**输入端口**（红色圆点）建立连线
3. 选中节点，在右侧属性面板配置参数
4. 点击工具栏 **「▶ 执行」**（或按 `F5`）运行流程
5. 结果图像可预览或保存到文件

## 算子列表

### 输入

| 算子 | 说明 |
|------|------|
| 图像读取 | 加载图像文件（png/jpg/bmp/tiff/webp） |
| 区域读取 | 加载图像/视频，在预览窗口绘制矩形/圆形/多边形区域，支持拖拽编辑顶点 |

### 图像处理

| 算子 | 算法 |
|------|------|
| 图像平滑 | 高斯模糊、均值滤波、中值滤波、双边滤波、引导滤波 |
| 边缘检测 | Sobel、Canny、Laplacian、Roberts |
| 阈值二值化 | Binary阈值、Otsu阈值、自适应阈值 |
| 几何变换 | 仿射变换、透视变换、图像旋转、图像缩放 |
| 形态学处理 | 腐蚀、膨胀、开运算、闭运算、形态学梯度 |
| 图像增强 | 直方图均衡化、对比度拉伸、伽马校正、对数变换、锐化滤波 |
| 频域处理 | 傅里叶变换、高通滤波、低通滤波、带通滤波、同态滤波 |
| 图像分割 | 区域生长、分水岭算法、K-means聚类、GrabCut、阈值分割 |
| 特征检测 | Harris角点、FAST角点、SIFT特征、SURF特征、HOG特征 |
| 颜色空间转换 | RGB ↔ HSV / LAB / YCrCb / YUV / HLS / LUV / Gray / XYZ / BGR |

### 图像叠加

| 算子 | 算法 |
|------|------|
| 图像绘制 | 轮廓绘制、实心填充、半透明填充（接收多区域输入） |
| 图像裁剪 | 区域裁剪（包围矩形裁切）、区域掩膜（区域外置黑） |

### 深度学习

| 算子 | 说明 |
|------|------|
| 目标检测 | ONNX Runtime 推理，支持 YOLO 格式输出，输出叠加框图像 + 区域列表 |
| 目标分类 | ONNX Runtime 推理，输出左上角叠加分类标签的图像 |

深度学习算子主要参数：
- 模型文件（.onnx）
- 输入分辨率（W × H）
- 置信度阈值 / IoU 阈值（检测）
- 类别数量 / Top-K（分类）
- 推理设备（GPU / CPU，默认 GPU）

### 输出

| 算子 | 说明 |
|------|------|
| 结果输出 | 图像预览、自动显示、保存到文件 |

## 端口类型

端口带有 `data_type` 标记，连线时会校验类型兼容性：

| 类型 | 说明 |
|------|------|
| 图像 | RGB numpy 数组 |
| 区域 | `{"type": "矩形"|"圆形"|"多边形", "coordinates": {...}}` |

## 架构概要

```
main.py              — 应用入口、主窗口、日志面板、属性面板
node_base.py         — Node / Port / Connection / ExecutionEngine 基类
node_registry.py     — 单例节点注册表（@register_node 装饰器）
node_canvas.py       — QGraphicsView 画布、节点项、连线、拖拽交互
nodes/
  image_input.py     — 图像读取
  region_input.py    — 区域读取（含绘制对话框）
  smoothing.py       — 图像平滑
  edge_detect.py     — 边缘检测
  threshold.py       — 阈值二值化
  geometry.py        — 几何变换
  morphology.py      — 形态学处理
  enhancement.py     — 图像增强
  frequency.py       — 频域处理
  segmentation.py    — 图像分割
  feature_detection.py — 特征检测
  color_space.py     — 颜色空间转换
  overlay.py         — 图像绘制（多连线输入）
  crop.py            — 图像裁剪
  object_detection.py — 目标检测（ONNX）
  classification.py  — 目标分类（ONNX）
  image_output.py    — 结果输出
```

执行引擎按拓扑顺序调度节点，后台线程运行避免 UI 阻塞，错误信息同时输出到日志面板和弹窗。

## 依赖

| 包 | 用途 |
|----|------|
| PyQt5 | GUI 框架 |
| opencv-python | 图像处理 |
| numpy | 数组计算 |
| onnxruntime | ONNX 模型推理 |
