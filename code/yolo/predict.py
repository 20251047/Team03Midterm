from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict(source=["front.jpg", "side.jpg"])
for result in results:
    result.show()

    for box in result.boxes:
        class_id = int(box.cls.item())
        print(model.names[class_id], box.conf.item())