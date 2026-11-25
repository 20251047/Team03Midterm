from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict(source=["front.jpg", "side.jpg"])
results[0].show()

for box in results[0].boxes:
    class_id = int(box.cls.item())
    print(model.names[class_id], box.conf.item())