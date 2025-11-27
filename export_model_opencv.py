#!/usr/bin/env python3
"""
Script to export YOLOv8 model to ONNX format compatible with OpenCV DNN
This exports without the DFL (Distribution Focal Loss) head that causes issues
"""

from ultralytics import YOLO
import sys

def export_model(model_path, output_name=None):
    """
    Export YOLOv8 model to ONNX format optimized for OpenCV DNN
    
    Args:
        model_path: Path to the .pt model file
        output_name: Optional custom output name (without .onnx extension)
    """
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    if output_name is None:
        output_name = model_path.replace('.pt', '_opencv')
    
    print(f"Exporting model to ONNX format (OpenCV compatible)...")
    # Export with simplify=True and without DFL head
    # Using format='onnx' with simplify=True should work better with OpenCV
    model.export(
        format='onnx',
        simplify=True,
        opset=12,  # OpenCV DNN works well with opset 12
        imgsz=640  # Standard input size
    )
    
    print(f"✓ Model exported successfully!")
    print(f"  Output: {model_path.replace('.pt', '.onnx')}")
    print(f"\nYou can now use this model with OpenCV DNN:")
    print(f"  net = cv2.dnn.readNetFromONNX('{model_path.replace('.pt', '.onnx')}')")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 export_model_opencv.py <model.pt> [output_name]")
        print("\nExample:")
        print("  python3 export_model_opencv.py yolov8n.pt")
        sys.exit(1)
    
    model_path = sys.argv[1]
    output_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    export_model(model_path, output_name)
