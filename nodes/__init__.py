"""节点包初始化，导入所有算子以触发注册。"""

from nodes.image_input import ImageInputNode
from nodes.grayscale import GrayscaleNode
from nodes.blur import BlurNode
from nodes.edge_detect import EdgeDetectNode
from nodes.threshold import ThresholdNode
from nodes.resize import ResizeNode
from nodes.image_output import ImageOutputNode
