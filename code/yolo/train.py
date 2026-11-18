from ultralytics import YOLO

model = YOLO("yolo11m.pt")
model.train(data="labeldFiled/data.yaml", epochs=50, imgsz=640)