from ultralytics import YOLO
from datetime import datetime
import requests
import numpy as np
import cv2
import csv
import time

model = YOLO("yolov8n.pt")


URL1 = "https://opencam.live/api/cctv/frame/bCqve_dO30m?_t=1794828553375"
URL2="https://opencam.live/api/cctv/stream/-z96xtGbkog/v7gGco0hHI42pT1kpWsVmW.ts"

with open("results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["date", "personnes"])

for i in range(20):   # 12 captures = 1 minute

    response = requests.get(URL2)

    img = np.asarray(
        bytearray(response.content),
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        img,
        cv2.IMREAD_COLOR
    )

    results = model(frame)

    nb_personnes = sum(
        1
        for r in results
        for box in r.boxes
        if int(box.cls) == 0
    )

    with open("results.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now(),
            nb_personnes
        ])

    print(
        f"Capture {i+1}: {nb_personnes}"
    )

    time.sleep(5)