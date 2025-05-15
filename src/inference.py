from ultralytics import YOLO
from PIL import Image
import numpy as np
from typing import List, Dict

def load_model(model_path: str):
    """
    Load YOLOv8 model from the specified path
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

def predict_image(image: Image.Image, model) -> List[Dict]:
    """
    Perform prediction on an image using the loaded model
    Returns a list of detections with bounding boxes, confidence scores, and class labels
    """
    try:
        # Perform prediction
        results = model.predict(image)
        
        # Process results
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    "class": result.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": {
                        "x1": float(box.xyxy[0][0]),
                        "y1": float(box.xyxy[0][1]),
                        "x2": float(box.xyxy[0][2]),
                        "y2": float(box.xyxy[0][3])
                    }
                })
        return detections
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")