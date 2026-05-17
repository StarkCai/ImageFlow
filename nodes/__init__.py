"""节点包初始化，导入所有算子以触发注册。"""

from nodes.image_input import ImageInputNode
from nodes.smoothing import SmoothingNode
from nodes.edge_detect import EdgeDetectNode
from nodes.threshold import ThresholdNode
from nodes.geometry import GeometryTransformNode
from nodes.morphology import MorphologyNode
from nodes.enhancement import EnhancementNode
from nodes.frequency import FrequencyNode
from nodes.segmentation import SegmentationNode
from nodes.feature_detection import FeatureDetectionNode
from nodes.color_space import ColorSpaceNode
from nodes.region_input import RegionInputNode
from nodes.overlay import OverlayNode
from nodes.crop import CropNode
from nodes.object_detection import ObjectDetectionNode
from nodes.classification import ClassificationNode
from nodes.class_mapping import ClassMappingNode
from nodes.ocr import OcrNode
from nodes.region_output import RegionOutputNode
from nodes.image_output import ImageOutputNode
