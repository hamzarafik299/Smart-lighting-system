import io
import socket
import struct
from datetime import datetime
from PIL import Image
from ultralytics import YOLO

# Configuration
HOST = "0.0.0.0"
PORT = 5000
CSV_VM_PATH = "resultst.csv"

print("Chargement du modèle YOLOv8...")
model = YOLO("yolov8n.pt")
print("Modèle prêt.")


def recv_exact(conn, num_bytes):
    """Garantit la lecture complète d'un bloc de données sur le socket."""
    data = bytearray()
    while len(data) < num_bytes:
        packet = conn.recv(num_bytes - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(5)
        print(f"Serveur YOLOv8 en écoute sur {HOST}:{PORT}...")

        while True:
            conn, addr = server.accept()
            with conn:
                try:
                    # 1. Lire la taille de l'image (8 octets)
                    raw_size = recv_exact(conn, 8)
                    if not raw_size:
                        continue
                    img_size = struct.unpack(">Q", raw_size)[0]

                    # 2. Recevoir les octets JPEG et ouvrir l'image avec PIL
                    img_bytes = recv_exact(conn, img_size)
                    if not img_bytes:
                        continue
                    img = Image.open(io.BytesIO(img_bytes))

                    # 3. Détection des personnes avec YOLOv8 (classe 0 = person)
                    results = model(img, classes=[0], verbose=False)
                    nb_personnes = len(results[0].boxes)

                    # 4. Écriture dans le fichier CSV
                    horodatage = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(CSV_VM_PATH, "a", encoding="utf-8") as f:
                        f.write(f"{horodatage},{nb_personnes}\n")

                    # 5. Envoi du CSV mis à jour au client
                    with open(CSV_VM_PATH, "rb") as f:
                        csv_bytes = f.read()

                    conn.sendall(struct.pack(">Q", len(csv_bytes)) + csv_bytes)
                    print(f"[{horodatage}] {addr[0]} -> {nb_personnes} personne(s) détectée(s)")

                except Exception as e:
                    print(f"Erreur avec {addr} : {e}")


if __name__ == "__main__":
    start_server()