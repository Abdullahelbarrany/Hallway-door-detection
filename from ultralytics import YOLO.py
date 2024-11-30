from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Define path to video file
source = "v.mp4"

# Run inference on the source
results = model(source, show=True)  # generator of Results objects