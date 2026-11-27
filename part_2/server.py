import io
import os
import socket
import struct
import time
from datetime import datetime
import psutil
from PIL import Image
from ultralytics import YOLO

# Configuration
HOST = "127.0.0.1"
# HOST = "0.0.0.0"
PORT = 5000
CSV_VM_PATH = "results_metrics.csv"

# Set your cloud VM's hourly rate in USD (e.g., $0.1942/hr for GCP n2-standard-4 or $0.2016/hr for AWS m7i.xlarge)
VM_HOURLY_COST_USD = 0.1942
EGRESS_COST_PER_GB_USD = 0.09  # Typical cloud outbound data transfer rate after free tier

print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")
process = psutil.Process(os.getpid())
print("Model ready.")

# Initialize CSV with headers if it doesn't exist
if not os.path.exists(CSV_VM_PATH):
    with open(CSV_VM_PATH, "w", encoding="utf-8") as f:
        f.write(
            "timestamp,persons_detected,avg_confidence,yolo_inference_ms,"
            "response_time_ms,cpu_percent,ram_used_mb,bytes_in,bytes_out,estimated_cost_usd\n"
        )


def recv_exact(conn, num_bytes):
    """Guarantees reading the exact number of bytes from the socket."""
    data = bytearray()
    while len(data) < num_bytes:
        packet = conn.recv(num_bytes - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


def start_server():
    # Prime psutil cpu_percent baseline
    psutil.cpu_percent(interval=None)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(5)
        print(f"YOLOv8 Server listening on {HOST}:{PORT}...")

        while True:
            conn, addr = server.accept()
            with conn:
                try:
                    req_start = time.perf_counter()

                    # 1. Read image size (8 bytes)
                    raw_size = recv_exact(conn, 8)
                    if not raw_size:
                        continue
                    img_size = struct.unpack(">Q", raw_size)[0]

                    # 2. Receive image bytes
                    img_bytes = recv_exact(conn, img_size)
                    if not img_bytes:
                        continue
                    bytes_in = 8 + len(img_bytes)

                    img = Image.open(io.BytesIO(img_bytes))

                    # 3. Run YOLOv8 detection (class 0 = person)
                    results = model(img, classes=[0], verbose=False)

                    boxes = results[0].boxes
                    nb_personnes = len(boxes)

                    # # Générer l'image avec les bounding boxes
                    # annotated_frame = results[0].plot()

                    # # Convertir en image PIL
                    # result_image = Image.fromarray(annotated_frame[:, :, ::-1])

                    # # Afficher l'image
                    # result_image.show()

                    # Detection efficiency metrics: mean confidence & pure model inference speed
                    avg_conf = float(boxes.conf.mean()) if nb_personnes > 0 else 0.0
                    yolo_inference_ms = results[0].speed["inference"]

                    # 4. System resource & cost metrics
                    cpu_pct = psutil.cpu_percent(interval=None)
                    ram_mb = process.memory_info().rss / (1024 * 1024)

                    # Prepare preliminary response time and bandwidth to log into CSV
                    # (We calculate final bytes_out right before sending)
                    horodatage = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Read existing CSV size to predict exact egress bytes
                    current_csv_size = os.path.getsize(CSV_VM_PATH)
                    estimated_line_len = 95  # Approximate byte length of the new row
                    bytes_out_est = 8 + current_csv_size + estimated_line_len

                    elapsed_sec = time.perf_counter() - req_start
                    response_time_ms = elapsed_sec * 1000

                    # Monetary cost = compute time cost + outbound bandwidth cost
                    compute_cost = (elapsed_sec / 3600.0) * VM_HOURLY_COST_USD
                    egress_cost = (bytes_out_est / (1024**3)) * EGRESS_COST_PER_GB_USD
                    total_cost_usd = compute_cost + egress_cost

                    # 5. Append metrics to CSV
                    row = (
                        f"{horodatage},{nb_personnes},{avg_conf:.4f},{yolo_inference_ms:.2f},"
                        f"{response_time_ms:.2f},{cpu_pct:.1f},{ram_mb:.2f},"
                        f"{bytes_in},{bytes_out_est},{total_cost_usd:.8f}\n"
                    )
                    with open(CSV_VM_PATH, "a", encoding="utf-8") as f:
                        f.write(row)

                    # 6. Send updated CSV back to client
                    with open(CSV_VM_PATH, "rb") as f:
                        csv_bytes = f.read()

                    conn.sendall(struct.pack(">Q", len(csv_bytes)) + csv_bytes)
                    print(
                        f"[{horodatage}] {addr[0]} | Persons: {nb_personnes} (conf: {avg_conf:.2f}) | "
                        f"Time: {response_time_ms:.1f}ms | CPU: {cpu_pct}% | RAM: {ram_mb:.0f}MB | "
                        f"Cost: ${total_cost_usd:.6f}"
                    )

                except Exception as e:
                    print(f"Error with {addr}: {e}")


if __name__ == "__main__":
    start_server()