"""区域坐标系转换算子：将输入区域的坐标系缩放转换到目标坐标系。"""

from node_base import Node, format_regions, _extract_regions, _extract_region_dims
from node_registry import register_node


@register_node
class CoordinateTransformNode(Node):
    display_name = "区域坐标系转换"
    category = "逻辑功能"

    def __init__(self):
        self.target_width: int = 1920
        self.target_height: int = 1080
        super().__init__()

    def _setup_ports(self):
        self.add_input("区域")
        self.add_output("区域")

    def process(self, **inputs):
        data = inputs.get("区域")
        regions = _extract_regions(data)
        if not regions:
            raise ValueError("区域数据为空")

        src_w, src_h = _extract_region_dims(data)
        if src_w <= 0 or src_h <= 0:
            raise ValueError(
                "输入区域缺少坐标系尺寸 (width/height)，"
                "请确保上游节点携带坐标系信息"
            )

        if src_w == self.target_width and src_h == self.target_height:
            return {"区域": format_regions(regions, self.target_width, self.target_height)}

        sx = self.target_width / src_w
        sy = self.target_height / src_h

        transformed = []
        for r in regions:
            coords = r["coordinates"]
            rtype = r["type"]
            new_coords = self._scale_coords(coords, rtype, sx, sy)
            transformed.append({
                "type": rtype,
                "class": r.get("class"),
                "ocr": r.get("ocr", ""),
                "coordinates": new_coords,
            })

        return {"区域": format_regions(transformed, self.target_width, self.target_height)}

    def _scale_coords(self, coords: dict, rtype: str, sx: float, sy: float) -> dict:
        """按缩放因子变换坐标。"""
        if rtype == "矩形":
            return {
                "x1": int(round(coords["x1"] * sx)),
                "y1": int(round(coords["y1"] * sy)),
                "x2": int(round(coords["x2"] * sx)),
                "y2": int(round(coords["y2"] * sy)),
            }
        elif rtype == "圆形":
            return {
                "cx": int(round(coords["cx"] * sx)),
                "cy": int(round(coords["cy"] * sy)),
                "radius": int(round(coords["radius"] * max(sx, sy))),
            }
        elif rtype == "多边形":
            return {
                "points": [
                    [int(round(p[0] * sx)), int(round(p[1] * sy))]
                    for p in coords["points"]
                ]
            }
        return dict(coords)
