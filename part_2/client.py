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
from tkinter import messagebox

# Ajout du dossier Smart-lighting-system au path Python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "Smart-lighting-system"))
import graph

# Synchronise le chemin du CSV entre Client.py et graph.py
graph.CSV_PATH = os.path.join(BASE_DIR, "resultst.csv")

# VM_IP = "9.205.17.7"
VM_IP ="127.0.0.1" 
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
        return False, "Aucune image capturée depuis le flux.", 0.0

    ok, encoded_img = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return False, "Erreur lors de l'encodage JPEG.", 0.0

    img_bytes = encoded_img.tobytes()
    t_start_net = time.perf_counter()

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(30)
            s.connect((host, port))
            s.sendall(struct.pack(">Q", len(img_bytes)) + img_bytes)

            raw_size = recv_exact(s, 8)
            if not raw_size:
                return False, "Aucune réponse reçue de la VM.", 0.0

            csv_size = struct.unpack(">Q", raw_size)[0]
            csv_bytes = recv_exact(s, csv_size)
            if not csv_bytes:
                return False, "Données CSV incomplètes.", 0.0

            rtt_ms = (time.perf_counter() - t_start_net) * 1000.0

            with open(output_csv, "wb") as f:
                f.write(csv_bytes)

            return True, "OK", rtt_ms

    except socket.error as e:
        return False, f"Erreur réseau : {e}", 0.0


class SmartLightingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Lighting System — Client & Performances Cloud")
        self.root.geometry("640x560")
        self.root.resizable(False, False)
        self.root.configure(bg="#f4f6f9")

        self.is_running = False
        self.stop_event = threading.Event()

        # Cumuls de la session en cours
        self.session_total_bytes = 0
        self.session_total_cost = 0.0

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        # En-tête
        header = tk.Frame(self.root, bg="#1e293b", pady=12)
        header.pack(fill="x")
        tk.Label(
            header, text="Smart Lighting System — Contrôle Caméra & Métriques VM",
            font=("Segoe UI", 13, "bold"), fg="white", bg="#1e293b"
        ).pack()

        # Zone URL du flux
        frame_url = tk.Frame(self.root, bg="#f4f6f9", padx=20, pady=8)
        frame_url.pack(fill="x")
        tk.Label(frame_url, text="URL du flux caméra :", font=("Segoe UI", 9, "bold"), bg="#f4f6f9").pack(anchor="w")
        self.url_var = tk.StringVar(value=DEFAULT_URL)
        tk.Entry(frame_url, textvariable=self.url_var, font=("Segoe UI", 9)).pack(fill="x", pady=(4, 0))

        # Statut et compteur en direct
        frame_status = tk.Frame(self.root, bg="white", bd=1, relief="solid", padx=15, pady=8)
        frame_status.pack(fill="x", padx=20, pady=4)

        self.status_label = tk.Label(
            frame_status, text="● Statut : Arrêté", font=("Segoe UI", 10, "bold"), fg="#dc2626", bg="white"
        )
        self.status_label.pack(side="left")

        self.count_label = tk.Label(
            frame_status, text="Personnes détectées : --", font=("Segoe UI", 11, "bold"), fg="#1e293b", bg="white"
        )
        self.count_label.pack(side="right")

        # Tableau de bord des 6 performances Cloud
        frame_perf = tk.LabelFrame(
            self.root, text=" Performances de la VM Cloud (Dernière requête) ",
            font=("Segoe UI", 9, "bold"), fg="#1e293b", bg="#f4f6f9", padx=10, pady=8
        )
        frame_perf.pack(fill="x", padx=20, pady=6)

        for col in range(3):
            frame_perf.columnconfigure(col, weight=1, uniform="perf_col")

        self.lbl_efficiency = self._create_metric_card(
            frame_perf, row=0, col=0, title="Efficacité détection", value="-- % (YOLO: -- ms)", color="#0284c7"
        )
        self.lbl_response = self._create_metric_card(
            frame_perf, row=0, col=1, title="Temps de réponse", value="VM: -- ms | RTT: -- ms", color="#7c3aed"
        )
        self.lbl_cpu = self._create_metric_card(
            frame_perf, row=0, col=2, title="Processeur (CPU)", value="-- %", color="#ea580c"
        )
        self.lbl_ram = self._create_metric_card(
            frame_perf, row=1, col=0, title="Mémoire utilisée (RAM)", value="-- MB", color="#0d9488"
        )
        self.lbl_bandwidth = self._create_metric_card(
            frame_perf, row=1, col=1, title="Bande passante (In / Out)", value="-- KB / -- KB", color="#2563eb"
        )
        self.lbl_cost = self._create_metric_card(
            frame_perf, row=1, col=2, title="Coût monétaire (Req / Total)", value="$-- / $--", color="#16a34a"
        )

        # Boutons de contrôle (Démarrer / Stopper)
        frame_ctrl = tk.Frame(self.root, bg="#f4f6f9", padx=20, pady=6)
        frame_ctrl.pack(fill="x")

        self.btn_start = tk.Button(
            frame_ctrl, text="▶ Démarrer l'envoi (5s)", command=self.start_sending,
            font=("Segoe UI", 10, "bold"), bg="#16a34a", fg="white", relief="flat", padx=12, pady=7, cursor="hand2"
        )
        self.btn_start.pack(side="left", expand=True, fill="x", padx=(0, 5))

        self.btn_stop = tk.Button(
            frame_ctrl, text="⏹ Stopper l'envoi", command=self.stop_sending,
            font=("Segoe UI", 10, "bold"), bg="#dc2626", fg="white", relief="flat", padx=12, pady=7,
            state="disabled", cursor="hand2"
        )
        self.btn_stop.pack(side="right", expand=True, fill="x", padx=(5, 0))

        # Boutons de consultation (Ouvrir CSV / Ouvrir Graphique)
        frame_view = tk.Frame(self.root, bg="#f4f6f9", padx=20, pady=4)
        frame_view.pack(fill="x")

        tk.Button(
            frame_view, text="📄 Ouvrir le fichier CSV complet", command=self.open_csv,
            font=("Segoe UI", 9, "bold"), bg="#2563eb", fg="white", relief="flat", padx=10, pady=6, cursor="hand2"
        ).pack(side="left", expand=True, fill="x", padx=(0, 5))

        tk.Button(
            frame_view, text="📊 Ouvrir le graphique", command=self.open_graph,
            font=("Segoe UI", 9, "bold"), bg="#7c3aed", fg="white", relief="flat", padx=10, pady=6, cursor="hand2"
        ).pack(side="right", expand=True, fill="x", padx=(5, 0))

        # Journal d'activité (Logs)
        frame_log = tk.Frame(self.root, bg="#f4f6f9", padx=20, pady=6)
        frame_log.pack(fill="both", expand=True)
        tk.Label(frame_log, text="Journal en direct :", font=("Segoe UI", 9, "bold"), bg="#f4f6f9").pack(anchor="w")

        self.log_box = tk.Text(frame_log, height=6, font=("Consolas", 8), bg="#0f172a", fg="#e2e8f0", state="disabled")
        self.log_box.pack(fill="both", expand=True, pady=(4, 0))

    def _create_metric_card(self, parent, row, col, title, value, color):
        card = tk.Frame(parent, bg="white", bd=1, relief="solid", padx=8, pady=6)
        card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")

        tk.Label(card, text=title, font=("Segoe UI", 8), fg="#64748b", bg="white").pack(anchor="w")
        val_label = tk.Label(card, text=value, font=("Segoe UI", 9, "bold"), fg=color, bg="white")
        val_label.pack(anchor="w", pady=(2, 0))
        return val_label

    def log(self, message):
        def _append():
            self.log_box.configure(state="normal")
            self.log_box.insert("end", message + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        self.root.after(0, _append)

    def _update_dashboard(self, nb, avg_conf, yolo_ms, vm_ms, rtt_ms, cpu, ram, kb_in, kb_out, req_cost):
        self.count_label.configure(text=f"Personnes détectées : {nb}")
        self.lbl_efficiency.configure(text=f"{avg_conf * 100:.1f}% (YOLO: {yolo_ms:.1f} ms)")
        self.lbl_response.configure(text=f"VM: {vm_ms:.1f} ms | RTT: {rtt_ms:.0f} ms")
        self.lbl_cpu.configure(text=f"{cpu:.1f} %")
        self.lbl_ram.configure(text=f"{ram:.1f} MB")
        self.lbl_bandwidth.configure(text=f"{kb_in:.1f} KB / {kb_out:.1f} KB")
        self.lbl_cost.configure(text=f"${req_cost:.6f} / ${self.session_total_cost:.5f}")

    def start_sending(self):
        if self.is_running:
            return
        self.is_running = True
        self.stop_event.clear()

        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.status_label.configure(text="● Statut : En cours (toutes les 5s)", fg="#16a34a")
        self.log("Démarrage de la capture et de l'envoi vers la VM Cloud...")

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
                ok, msg, rtt_ms = send_frame_and_save_csv(frame)

                if ok and os.path.exists(graph.CSV_PATH):
                    df = pd.read_csv(graph.CSV_PATH)

                    if "timestamp" not in df.columns:
                        self.log("[Erreur] Format CSV incompatible. Supprimez l'ancien resultst.csv sur la VM.")
                        break

                    last = df.iloc[-1]
                    horodatage = last["timestamp"]
                    nb = int(last["persons_detected"])
                    avg_conf = float(last["avg_confidence"])
                    yolo_ms = float(last["yolo_inference_ms"])
                    vm_ms = float(last["response_time_ms"])
                    cpu = float(last["cpu_percent"])
                    ram = float(last["ram_used_mb"])
                    bytes_in = int(last["bytes_in"])
                    bytes_out = int(last["bytes_out"])
                    req_cost = float(last["estimated_cost_usd"])

                    kb_in = bytes_in / 1024.0
                    kb_out = bytes_out / 1024.0
                    self.session_total_bytes += (bytes_in + bytes_out)
                    self.session_total_cost += req_cost

                    self.root.after(
                        0,
                        lambda: self._update_dashboard(
                            nb, avg_conf, yolo_ms, vm_ms, rtt_ms, cpu, ram, kb_in, kb_out, req_cost
                        )
                    )
                    self.log(
                        f"[{horodatage}] {nb} pers. (conf {avg_conf*100:.0f}%) | "
                        f"VM: {vm_ms:.0f}ms (RTT: {rtt_ms:.0f}ms) | CPU: {cpu:.1f}% | "
                        f"RAM: {ram:.0f}MB | BW: {kb_in+kb_out:.1f}KB | ${req_cost:.6f}"
                    )
                else:
                    self.log(f"[Erreur] {msg}")

            except Exception as e:
                self.log(f"[Exception] {e}")

            temps_ecoule = time.time() - debut
            self.stop_event.wait(max(0, INTERVALLE_SECONDES - temps_ecoule))

    def open_csv(self):
        if not os.path.exists(graph.CSV_PATH):
            messagebox.showwarning("Fichier introuvable", f"Le fichier {graph.CSV_PATH} n'existe pas encore.\nLancez d'abord un envoi.")
            return
        os.startfile(graph.CSV_PATH)

    def open_graph(self):
        if not os.path.exists(graph.CSV_PATH):
            messagebox.showwarning("Aucune donnée", "Aucune donnée enregistrée pour afficher le graphique.")
            return

        # Génère un CSV temporaire à 2 colonnes sans en-tête pour rester 100 % compatible avec graph.py
        try:
            df = pd.read_csv(graph.CSV_PATH)
            graph_csv_path = os.path.join(BASE_DIR, "resultst_graph.csv")
            if "timestamp" in df.columns and "persons_detected" in df.columns:
                df[["timestamp", "persons_detected"]].to_csv(graph_csv_path, index=False, header=False)
            else:
                graph_csv_path = graph.CSV_PATH
        except Exception:
            graph_csv_path = graph.CSV_PATH

        graph_script = os.path.join(BASE_DIR, "Smart-lighting-system", "graph.py")
        if os.path.exists(graph_script):
            subprocess.Popen([
                sys.executable, "-c",
                f"import sys; sys.path.append(r'{os.path.dirname(graph_script)}'); "
                f"import graph; graph.CSV_PATH = r'{graph_csv_path}'; graph.show_graph()"
            ])
        else:
            graph.CSV_PATH = graph_csv_path
            graph.show_graph()

    def on_close(self):
        self.stop_event.set()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SmartLightingApp(root)
    root.mainloop()