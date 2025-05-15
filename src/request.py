from pydantic import BaseModel, Field, HttpUrl
from typing import List, Literal, Optional
from enum import Enum

class DetectionMethod(str, Enum):
    YOLOv8 = "yolov8"
   

class ImageSourceType(str, Enum):
    UPLOAD = "upload"
    URL = "url"

class ImageInput(BaseModel):
    """
    Base model for image input, can be either file upload or URL
    """
    source_type: ImageSourceType = Field(
        default=ImageSourceType.UPLOAD,
        description="Type of image source (upload or URL)"
    )
    image_url: Optional[HttpUrl] = Field(
        default=None,
        description="URL of the image if source_type is 'url'"
    )

class DetectionRequest(ImageInput):
    """
    Request model for object detection
    """
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.1,
        le=0.99,
        description="Minimum confidence threshold for detections (0.1-0.99)"
    )
    iou_threshold: float = Field(
        default=0.45,
        ge=0.1,
        le=0.99,
        description="Intersection over Union threshold for NMS (0.1-0.99)"
    )
    method: DetectionMethod = Field(
        default=DetectionMethod.YOLOv8,
        description="Detection method to use"
    )
    return_image: bool = Field(
        default=False,
        description="Whether to return the annotated image"
    )
    classes: Optional[List[int]] = Field(
        default=None,
        description="List of class IDs to detect (None for all classes)"
    )

class BBox(BaseModel):
    """
    Bounding box model
    """
    x1: float = Field(description="Top-left x coordinate")
    y1: float = Field(description="Top-left y coordinate")
    x2: float = Field(description="Bottom-right x coordinate")
    y2: float = Field(description="Bottom-right y coordinate")

class DetectionResult(BaseModel):
    """
    Single detection result
    """
    class_id: int = Field(description="Class ID of the detected object")
    class_name: str = Field(description="Class name of the detected object")
    confidence: float = Field(description="Confidence score (0-1)")
    bbox: BBox = Field(description="Bounding box coordinates")

class DetectionResponse(BaseModel):
    """
    Response model for detection results
    """
    success: bool = Field(description="Whether the detection was successful")
    detections: List[DetectionResult] = Field(description="List of detections")
    processing_time: float = Field(description="Time taken for processing (seconds)")
    image_size: Optional[dict] = Field(
        default=None,
        description="Original image dimensions (width, height)"
    )
    annotated_image: Optional[str] = Field(
        default=None,
        description="Base64 encoded annotated image if requested",
        example="data:image/jpeg;base64,..."
    )