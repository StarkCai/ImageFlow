# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

No linter, type checker, or test runner is configured. Python 3.7+ required.

## Architecture

**Image Flow** is a PyQt5 node-graph editor for image processing pipelines. Users drag operators from a sidebar onto a QGraphicsView canvas, wire them together, and hit Execute (F5) to run the graph.

### Core classes (`node_base.py`)

- **`Node`** — base class for all operators. Subclasses declare ports in `_setup_ports()` via `add_input(name, data_type, multi_connect)` / `add_output(name, data_type)`, and implement `process(**inputs) -> dict[str, Any]`. Ports carry `data_type` tags ("图像", "区域", "类别映射") for type-safe wiring.
- **`Port`** — named input/output endpoint with `data_type`, `multi_connect` flag (for region ports that aggregate multiple incoming connections).
- **`Connection`** — links an output Port to an input Port.
- **`ExecutionEngine`** — owns the node/connection lists, runs Kahn topological sort, and executes nodes in order. The `_data` dict on each node stores intermediate results keyed by port name. Has a `_progress_callback` hook for UI updates.
- **`_extract_regions(data)`** / **`format_regions(regions)`** — utility functions for the standard region data format (`{"regions": [{"id", "type", "class", "ocr", "coordinates"}, ...]}`).

### Batch execution (`batch_queue.py`)

The engine supports two execution modes. `execute()` runs the graph once. `execute_batch()` detects folder-mode source nodes and runs the pipeline repeatedly:

- **`BatchItem`** — dataclass holding `image` (RGB uint8 HxWx3), `filename`, `filepath`, `index`.
- **`ImageQueue`** — bounded producer-consumer queue with progress tracking, cancellation, and error collection. `None` is the end-of-stream sentinel.
- Source nodes (`ImageInputNode`, `VideoInputNode`) implement `prepare_batch()` → `next_batch_item()` → `cleanup_batch()` lifecycle. A background thread reads files from disk and pushes them into the queue.
- Between iterations, all non-source nodes are reset (`_data` cleared, `_processed = False`). The source node's output is injected directly.
- Output nodes auto-save per iteration via `_post_batch_iteration()`.

### Node registry (`node_registry.py`)

Singleton `NodeRegistry` + `@register_node` decorator. Every node class is decorated and registered at import time. `nodes/__init__.py` triggers all imports. The class attributes `display_name` and `category` control sidebar grouping and labels.

### Canvas (`node_canvas.py`)

`NodeItem` (QGraphicsObject) renders a rounded-rect node with header, ports as circles (blue = output, red = input), and supports drag-to-move. `WireItem` draws cubic-Bezier connections between ports. Left-click a port starts a wire; right-click deletes. Middle-mouse pans, scroll wheel zooms. `TempWireItem` shows a dashed line during drag. `NodeScene` manages the scene graph, background grid, and port-connection logic. `NodeCanvas` (QGraphicsView) handles pan/zoom.

### Main window (`main.py`)

Three-panel layout: **left** (210px, `NodePanel` with collapsible category sections), **center** (`NodeCanvas` + `LogPanel` at 150px fixed height), **right** (240px, `PropertyPanel`). The `PropertyPanel` dynamically builds per-node-type parameter widgets using helper methods (`_add_spin`, `_add_combo`, `_add_double_spin`, etc.) and handles algorithm switching for multi-algorithm nodes. `FlowRunner` (QThread) executes the graph in a background thread, emitting signals back to the UI for log messages, batch progress, and completion handling.

### Project files (`project.py`)

JSON serialization with `_node_params()` extracting non-private, non-callable attributes (filtering via `_SKIP_ATTRS`). `apply_params()` restores them. Saved format: `{"version": "1.1", "nodes": [...], "connections": [...]}`.

### Node categories

- **输入** — `ImageInputNode` (single/folder), `VideoInputNode` (single/folder/frame-skip), `RegionInputNode` (draw-on-image), `ClassMappingNode` (JSON class-id→name)
- **图像处理** — Smoothing (5 algorithms), EdgeDetect (4), Threshold (3), GeometryTransform (4), Morphology (5), Enhancement (5), Frequency (5), Segmentation (5), FeatureDetection (5), ColorSpace
- **图像叠加** — `OverlayNode` (outline/fill/semi-transparent, multi-connect region input), `CropNode` (bbox crop/mask)
- **深度学习** — `ObjectDetectionNode` (YOLO ONNX, auto-detects output format), `ClassificationNode` (ImageNet-style ONNX), `OcrNode` (detection+recognition ONNX pipeline with dictionary)
- **逻辑** — `CoordinateTransformNode` (rescale region coordinates between image sizes)
- **输出** — `ImageOutputNode` (preview/save/auto-show), `RegionOutputNode` (JSON preview/detail/download/auto-save), `VideoOutputNode` (video/image output modes, frame accumulation)

### Adding a new node

1. Create a new file in the appropriate `nodes/` subdirectory.
2. Define a class inheriting from `Node`, set `display_name` and `category` as class attributes.
3. If the node has multiple algorithms, set `algorithms = [...]` and an `algorithm: str` instance attribute.
4. Override `_setup_ports()` and `process(**inputs)`.
5. Decorate with `@register_node`.
6. Add the import to `nodes/__init__.py`.
7. Add the corresponding `_add_<node>_params()` method to `PropertyPanel` in `main.py` and wire it up in `_add_node_params()`.

### Key patterns

- Node ports are declared in the subclass `__init__`; `super().__init__()` must be called **after** setting mutable defaults (since `_setup_ports()` runs from the base `__init__`).
- Image data flows as RGB numpy uint8 arrays (HxWx3). Region data uses the standard dict format. Class mapping data uses `{"mapping": {"0": "name", ...}}`.
- ONNX inference nodes auto-detect model output formats (normalized vs pixel coords, xyxy vs cxcywh, YOLOv5 vs YOLOv8) and handle coordinate normalization back to the original image space.
- The log panel is thread-safe via `pyqtSignal`-connected `_do_append`.
- Multi-connect input ports (e.g., `OverlayNode` region input) aggregate multiple upstream connections into a list passed to `process()`.
- Batch cancellation sets `_cancelled` on the engine and calls `cancel_batch()` on source nodes, which pushes a `None` sentinel into the ImageQueue.
