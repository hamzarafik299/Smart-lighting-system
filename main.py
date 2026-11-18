 
import tkinter as tk
from tkinter import messagebox
import threading
import time
import csv
import os
from datetime import datetime

import requests
import numpy as np
import cv2

import pandas as pd
import matplotlib.pyplot as plt

from ultralytics import YOLO
from outils import get_frame 
from graph import show_graph
# =====================================================
# CONFIGURATION
# =====================================================

URL = "https://opencam.live/api/cctv/stream/-z96xtGbkog/obWGa41vFHZdRakpDaSdHX.ts"

model = YOLO("yolov8n.pt")

running = False

# =====================================================
# CREATION CSV
# =====================================================

if not os.path.exists("resultst.csv"):
    with open("resultst.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "personnes"])

# =====================================================
# SURVEILLANCE 
# =====================================================

def surveillance():

    global running

    while running:

        try:
            
            frame = get_frame(URL)
            results = model(frame)

            nb_personnes = 0

            for r in results:
                for box in r.boxes:

                    if int(box.cls) == 0:
                        nb_personnes += 1

            root.after(
                0,
                lambda n=nb_personnes:
                label_count.config(
                    text=f"Personnes détectées : {n}"
                )
            )

            with open(
                "resultst.csv",
                "a",
                newline=""
            ) as f:

                writer = csv.writer(f)

                writer.writerow([
                    datetime.now(),
                    nb_personnes
                ])

            print(
                f"{datetime.now()} -> {nb_personnes} personnes"
            )

        except Exception as e:

            print("Erreur :", e)

        time.sleep(5)
# =====================================================
# START
# =====================================================

def start():

    global running

    if running:
        return

    running = True

    thread = threading.Thread(
        target=surveillance,
        daemon=True
    )

    thread.start()

# =====================================================
# STOP
# =====================================================

def stop():

    global running

    running = False
# =====================================================
# OUVRIR CSV
# =====================================================

def open_csv():

    if os.path.exists("resultst.csv"):
        os.startfile("resultst.csv")
    else:
        messagebox.showwarning(
            "Attention",
            "Fichier introuvable"
        )
# =====================================================
# INTERFACE
# =====================================================
root = tk.Tk()

root.title(
    "Détection de personnes avec YOLOv8"
)

root.geometry("600x400")

title = tk.Label(
    root,
    text="Surveillance Webcam",
    font=("Arial", 18, "bold")
)

title.pack(pady=10)

label_count = tk.Label(
    root,
    text="Personnes détectées : 0",
    font=("Arial", 16)
)

label_count.pack(pady=20)

btn_start = tk.Button(
    root,
    text="▶ Démarrer",
    command=start,
    width=25,
    bg="lightgreen"
)

btn_start.pack(pady=5)

btn_stop = tk.Button(
    root,
    text="■ Arrêter",
    command=stop,
    width=25,
    bg="salmon"
)

btn_stop.pack(pady=5)

btn_graph = tk.Button(
    root,
    text="📈 Afficher graphique",
    command=show_graph,
    width=25
)

btn_graph.pack(pady=5)

btn_csv = tk.Button(
    root,
    text="📄 Ouvrir CSV",
    command=open_csv,
    width=25
)

btn_csv.pack(pady=5)

root.mainloop()