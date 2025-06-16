from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")  # Load a pretrained model
    results = model.train(data="data.yaml", epochs=5, imgsz=640)