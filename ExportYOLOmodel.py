from ultralytics import YOLO

# 1. Load model
model = YOLO("yolo11n.pt") 

# 2. Export with C++ optimized settings
model.export(
    format="onnx", 
    imgsz=640,       # Fixed size is easier for initial C++ implementation
    dynamic=False,   # Keep it fixed for better optimization
    simplify=True,   # Removes unnecessary ONNX nodes
    opset=12,         # Best compatibility for ONNX Runtime + CUDA
    name="yolo11"
)