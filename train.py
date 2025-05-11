from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO("yolo11n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo11n.pt")

# Train the model
results = model.train(
    data="railwaytools.yaml",
    epochs=300,
    imgsz=640,
)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
# results = model("https://ultralytics.com/images/bus.jpg")

# Export the model to ONNX format
# success = model.export(format="onnx")
