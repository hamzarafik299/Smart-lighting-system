#extraire les frames
import time
import requests
import numpy as np
import cv2


def get_frame(url):
    with requests.get(url, timeout=10, stream=True) as response:
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()

        # Cas 1 : image fixe (jpeg, png...)
        if content_type.startswith("image/"):
            img = np.asarray(bytearray(response.content), dtype=np.uint8)
            return cv2.imdecode(img, cv2.IMREAD_COLOR)

        # Cas 2 : vidéo ou flux MJPEG
        if (content_type.startswith("video/")
                or "multipart/x-mixed-replace" in content_type
                or "mpegurl" in content_type):
            cap = cv2.VideoCapture(url)
            try:
                ok, frame = cap.read()   # on récupère une seule frame
                return frame if ok else None
            finally:
                cap.release()

        print(f"Content-Type non géré : {content_type}")
        return None
