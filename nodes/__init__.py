"""节点包初始化，导入所有算子以触发注册。"""

from nodes.input.image_input import ImageInputNode
from nodes.input.video_input import VideoInputNode
from nodes.input.region_input import RegionInputNode
from nodes.input.class_mapping import ClassMappingNode
from nodes.image_processing.smoothing import SmoothingNode
from nodes.image_processing.edge_detect import EdgeDetectNode
from nodes.image_processing.threshold import ThresholdNode
from nodes.image_processing.geometry import GeometryTransformNode
from nodes.image_processing.morphology import MorphologyNode
from nodes.image_processing.enhancement import EnhancementNode
from nodes.image_processing.frequency import FrequencyNode
from nodes.image_processing.segmentation import SegmentationNode
from nodes.image_processing.feature_detection import FeatureDetectionNode
from nodes.image_processing.color_space import ColorSpaceNode
from nodes.overlay.overlay import OverlayNode
from nodes.overlay.crop import CropNode
from nodes.deep_learning.object_detection import ObjectDetectionNode
from nodes.deep_learning.classification import ClassificationNode
from nodes.deep_learning.ocr import OcrNode
from nodes.logic.coordinate_transform import CoordinateTransformNode
from nodes.output.region_output import RegionOutputNode
from nodes.output.image_output import ImageOutputNode
from nodes.output.video_output import VideoOutputNode
