from ultralytics import YOLO

model = YOLO("yolov8n.pt")



#le variable fram et e resultat de get_frame du local


def count(fram):
    results = model(fram)
    nb_personnes = 0

    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0:
                nb_personnes += 1
    return nb_personnes