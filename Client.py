import os
import sys
import time
import socket
import struct
import threading
import subprocess
import requests
import numpy as np
import cv2
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

# Ajout du dossier Smart-lighting-system au path Python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "Smart-lighting-system"))
import graph

# Synchronise le chemin du CSV entre Client.py et graph.py
graph.CSV_PATH = os.path.join(BASE_DIR, "resultst.csv")

VM_IP = "9.205.17.7"
VM_PORT = 5000
DEFAULT_URL = "https://opencam.live/api/cctv/stream/s1o0W5Oq0v4/Wt2AxyiEGvwwMEHIIOztNi.m3u8"
INTERVALLE_SECONDES = 5


def get_frame(url):
    if ".m3u8" in url.lower():
        cap = cv2.VideoCapture(url)
        try:
            ok, frame = cap.read()
            return frame if ok else None
        finally:
            cap.release()

    with requests.get(url, timeout=10, stream=True) as response:
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()

        if content_type.startswith("image/"):
            img = np.asarray(bytearray(response.content), dtype=np.uint8)
            return cv2.imdecode(img, cv2.IMREAD_COLOR)

        if (content_type.startswith("video/")
                or "multipart/x-mixed-replace" in content_type
                or "mpegurl" in content_type):
            cap = cv2.VideoCapture(url)
            try:
                ok, frame = cap.read()
                return frame if ok else None
            finally:
                cap.release()

        return None


def recv_exact(sock, num_bytes):
    data = bytearray()
    while len(data) < num_bytes:
        chunk = sock.recv(num_bytes - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def send_frame_and_save_csv(frame, host=VM_IP, port=VM_PORT, output_csv=graph.CSV_PATH):
    if frame is None:
        return False, "Aucune image capturée depuis le flux."

    ok, encoded_img = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return False, "Erreur lors de l'encodage JPEG."

    img_bytes = encoded_img.tobytes()

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(30)
            s.connect((host, port))
            s.sendall(struct.pack(">Q", len(img_bytes)) + img_bytes)

            raw_size = recv_exact(s, 8)
            if not raw_size:
                return False, "Aucune réponse reçue de la VM."

            csv_size = struct.unpack(">Q", raw_size)[0]
            csv_bytes = recv_exact(s, csv_size)
            if not csv_bytes:
                return False, "Données CSV incomplètes."

            with open(output_csv, "wb") as f:
                f.write(csv_bytes)

            return True, "OK"

    except socket.error as e:
        return False, f"Erreur réseau : {e}"


class SmartLightingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Lighting System — Client")
        self.root.geometry("560x420")
        self.root.resizable(False, False)
        self.root.configure(bg="#f4f6f9")

        self.is_running = False
        self.stop_event = threading.Event()

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        # En-tête
        header = tk.Frame(self.root, bg="#1e293b", pady=12)
        header.pack(fill="x")
        tk.Label(
            header, text="Smart Lighting System — Contrôle Caméra",
            font=("Segoe UI", 13, "bold"), fg="white", bg="#1e293b"
        ).pack()

        # Zone URL du flux
        frame_url = tk.Frame(self.root, bg="#f4f6f9", padx=20, pady=10)
        frame_url.pack(fill="x")
        tk.Label(frame_url, text="URL du flux caméra :", font=("Segoe UI", 9, "bold"), bg="#f4f6f9").pack(anchor="w")
        self.url_var = tk.StringVar(value=DEFAULT_URL)
        tk.Entry(frame_url, textvariable=self.url_var, font=("Segoe UI", 9), width=75).pack(fill="x", pady=(4, 0))

        # Statut et compteur en direct
        frame_status = tk.Frame(self.root, bg="white", bd=1, relief="solid", padx=15, pady=10)
        frame_status.pack(fill="x", padx=20, pady=5)

        self.status_label = tk.Label(
            frame_status, text="● Statut : Arrêté", font=("Segoe UI", 10, "bold"), fg="#dc2626", bg="white"
        )
        self.status_label.pack(side="left")

        self.count_label = tk.Label(
            frame_status, text="Personnes détectées : --", font=("Segoe UI", 11, "bold"), fg="#1e293b", bg="white"
        )
        self.count_label.pack(side="right")

        # Boutons de contrôle (Démarrer / Stopper)
        frame_ctrl = tk.Frame(self.root, bg="#f4f6f9", padx=20, pady=8)
        frame_ctrl.pack(fill="x")

        self.btn_start = tk.Button(
            frame_ctrl, text="▶ Démarrer l'envoi (5s)", command=self.start_sending,
            font=("Segoe UI", 10, "bold"), bg="#16a34a", fg="white", relief="flat", padx=12, pady=8, cursor="hand2"
        )
        self.btn_start.pack(side="left", expand=True, fill="x", padx=(0, 5))

        self.btn_stop = tk.Button(
            frame_ctrl, text="⏹ Stopper l'envoi", command=self.stop_sending,
            font=("Segoe UI", 10, "bold"), bg="#dc2626", fg="white", relief="flat", padx=12, pady=8,
            state="disabled", cursor="hand2"
        )
        self.btn_stop.pack(side="right", expand=True, fill="x", padx=(5, 0))

        # Boutons de consultation (Ouvrir CSV / Ouvrir Graphique)
        frame_view = tk.Frame(self.root, bg="#f4f6f9", padx=20, pady=4)
        frame_view.pack(fill="x")

        tk.Button(
            frame_view, text="📄 Ouvrir le fichier CSV", command=self.open_csv,
            font=("Segoe UI", 9, "bold"), bg="#2563eb", fg="white", relief="flat", padx=10, pady=7, cursor="hand2"
        ).pack(side="left", expand=True, fill="x", padx=(0, 5))

        tk.Button(
            frame_view, text="📊 Ouvrir le graphique", command=self.open_graph,
            font=("Segoe UI", 9, "bold"), bg="#7c3aed", fg="white", relief="flat", padx=10, pady=7, cursor="hand2"
        ).pack(side="right", expand=True, fill="x", padx=(5, 0))

        # Journal d'activité (Logs)
        frame_log = tk.Frame(self.root, bg="#f4f6f9", padx=20, pady=8)
        frame_log.pack(fill="both", expand=True)
        tk.Label(frame_log, text="Journal en direct :", font=("Segoe UI", 9, "bold"), bg="#f4f6f9").pack(anchor="w")

        self.log_box = tk.Text(frame_log, height=7, font=("Consolas", 8), bg="#0f172a", fg="#e2e8f0", state="disabled")
        self.log_box.pack(fill="both", expand=True, pady=(4, 0))

    def log(self, message):
        def _append():
            self.log_box.configure(state="normal")
            self.log_box.insert("end", message + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        self.root.after(0, _append)

    def start_sending(self):
        if self.is_running:
            return
        self.is_running = True
        self.stop_event.clear()

        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.status_label.configure(text="● Statut : En cours (toutes les 5s)", fg="#16a34a")
        self.log("Démarrage de la capture et de l'envoi vers la VM Azure...")

        threading.Thread(target=self._worker_loop, daemon=True).start()

    def stop_sending(self):
        if not self.is_running:
            return
        self.is_running = False
        self.stop_event.set()

        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.status_label.configure(text="● Statut : Arrêté", fg="#dc2626")
        self.log("Envoi arrêté par l'utilisateur.")

    def _worker_loop(self):
        while not self.stop_event.is_set():
            debut = time.time()
            url = self.url_var.get().strip()

            try:
                frame = get_frame(url)
                ok, msg = send_frame_and_save_csv(frame)

                if ok and os.path.exists(graph.CSV_PATH):
                    df = pd.read_csv(graph.CSV_PATH, names=["Horodatage", "Personnes"])
                    horodatage = df["Horodatage"].iloc[-1]
                    nb = df["Personnes"].iloc[-1]

                    self.root.after(0, lambda n=nb: self.count_label.configure(text=f"Personnes détectées : {n}"))
                    self.log(f"[{horodatage}] Envoyé -> {nb} personne(s) détectée(s)")
                else:
                    self.log(f"[Erreur] {msg}")

            except Exception as e:
                self.log(f"[Exception] {e}")

            # Attente intelligente de 5s interruptible immédiatement par le bouton Stop
            temps_ecoule = time.time() - debut
            self.stop_event.wait(max(0, INTERVALLE_SECONDES - temps_ecoule))

    def open_csv(self):
        if not os.path.exists(graph.CSV_PATH):
            messagebox.showwarning("Fichier introuvable", f"Le fichier {graph.CSV_PATH} n'existe pas encore.\nLancez d'abord un envoi.")
            return
        os.startfile(graph.CSV_PATH)  # Ouvre directement dans Excel / Bloc-notes sur Windows

    def open_graph(self):
        if not os.path.exists(graph.CSV_PATH):
            messagebox.showwarning("Aucune donnée", "Aucune donnée enregistrée pour afficher le graphique.")
            return
        # Lance graph.py dans un processus séparé pour ne pas bloquer l'interface Tkinter
        graph_script = os.path.join(BASE_DIR, "Smart-lighting-system", "graph.py")
        if os.path.exists(graph_script):
            subprocess.Popen([sys.executable, "-c", f"import sys; sys.path.append(r'{os.path.dirname(graph_script)}'); import graph; graph.CSV_PATH = r'{graph.CSV_PATH}'; graph.show_graph()"])
        else:
            graph.show_graph()

    def on_close(self):
        self.stop_event.set()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SmartLightingApp(root)
    root.mainloop()