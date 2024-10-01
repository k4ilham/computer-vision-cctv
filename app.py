import cv2
import torch
import os
import shutil
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
import csv
from datetime import datetime
import time

# Path untuk model YOLOv5
model_path = 'yolov5s.pt'

# Cek apakah model sudah ada, jika tidak ada, unduh model
if not os.path.exists(model_path):
    print("Model tidak ditemukan, mengunduh model YOLOv5...")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', force_reload=True)
else:
    print("Model ditemukan, memuat model YOLOv5 dari path lokal...")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Membaca data CCTV dari file CSV
cctv_data = pd.read_csv('cctv_data.csv')  # Pastikan file CSV ini ada di direktori yang sama
streams = dict(zip(cctv_data['Nama'], cctv_data['URL']))

# Variabel global untuk kontrol streaming
cap = None
streaming = False

# Variabel untuk menyimpan pilihan dropdown
stream_var = None

# Diksi untuk menyimpan jumlah deteksi
detection_counts = {"car": 0, "motorcycle": 0, "truck": 0}
detected_ids = {}
iou_threshold = 0.5  # Threshold IoU untuk mempertimbangkan deteksi

# Membuat folder 'dataset' jika belum ada
if not os.path.exists('dataset'):
    os.makedirs('dataset')
    for category in ['car', 'motorcycle', 'truck']:
        os.makedirs(f'dataset/{category}', exist_ok=True)

def clear_dataset_folder():
    # Menghapus semua file dalam folder dataset
    folder_path = 'dataset'
    for category in ['car', 'motorcycle', 'truck']:
        folder = os.path.join(folder_path, category)
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)  # Menghapus file
                except Exception as e:
                    print(f"Error: {e} ketika menghapus file {file_path}")

def calculate_iou(box1, box2):
    # Menghitung IoU antara dua kotak
    x1_intersection = max(box1[0], box2[0])
    y1_intersection = max(box1[1], box2[1])
    x2_intersection = min(box1[2], box2[2])
    y2_intersection = min(box1[3], box2[3])

    intersection_area = max(0, x2_intersection - x1_intersection) * max(0, y1_intersection - y1_intersection)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0

# Fungsi untuk memulai stream
def start_stream():
    global cap, streaming, stream_var, detection_counts, detected_ids
    selected_stream = stream_var.get()
    url = streams[selected_stream]

    # Hapus semua file dalam folder dataset
    clear_dataset_folder()

    # Reset jumlah deteksi ketika memulai stream
    detection_counts = {"car": 0, "motorcycle": 0, "truck": 0}
    detected_ids = {}

    # Membuka video stream yang dipilih
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("Error: Tidak dapat membuka stream.")
        return

    # Mengatur ukuran jendela OpenCV
    cv2.namedWindow('Stream', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Stream', 800, 600)  # Mengatur ukuran jendela menjadi 800x600
    streaming = True

    # Menulis header CSV jika file belum ada
    if not os.path.exists('detections.csv'):
        with open('detections.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Class'])  # Menulis header kolom

    while streaming:
        # Membaca frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Tidak dapat membaca frame.")
            break

        # Mengubah ukuran frame agar sesuai dengan model
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Deteksi objek
        results = model(img)

        # Mengambil hasil deteksi
        detections = results.xyxy[0].numpy()  # format [x1, y1, x2, y2, confidence, class]

        # Menampilkan hasil deteksi
        for *box, conf, cls in detections:
            label = model.names[int(cls)]
            if label in ["car", "motorcycle", "truck"] and conf > 0.5:  # Hanya pertimbangkan deteksi yang relevan
                x1, y1, x2, y2 = map(int, box)
                current_box = (x1, y1, x2, y2)
                current_time = time.time()

                # Filter deteksi dengan IoU
                detected = False
                for detected_id, (detected_box, last_time) in detected_ids.items():
                    if calculate_iou(current_box, detected_box) > iou_threshold:
                        detected = True
                        # Perbarui waktu deteksi terakhir
                        detected_ids[detected_id] = (detected_box, current_time)
                        break
                
                # Jika tidak terdeteksi sebelumnya, catat deteksi baru
                if not detected:
                    detection_id = len(detected_ids) + 1  # atau gunakan metode lain untuk menghasilkan ID unik
                    detected_ids[detection_id] = (current_box, current_time)

                    # Crop dan simpan gambar objek yang dideteksi
                    cropped_image = frame[y1:y2, x1:x2]  # Crop gambar berdasarkan bounding box
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    file_path = f'dataset/{label}/{label}_{timestamp}_{detection_id}.jpg'  # Buat path file
                    cv2.imwrite(file_path, cropped_image)  # Simpan gambar ke file

                    # Catat deteksi ke file CSV
                    with open('detections.csv', 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), label])  # Catat timestamp dan label

                    # Update jumlah deteksi
                    detection_counts[label] += 1

                    # Gambar bounding box
                    color = (0, 255, 0) if label == "car" else (255, 255, 0) if label == "motorcycle" else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

        # Menampilkan jumlah deteksi di bagian bawah frame
        text_display = f"Car: {detection_counts['car']}, Motorcycle: {detection_counts['motorcycle']}, Truck: {detection_counts['truck']}"
        cv2.putText(frame, text_display, (10, frame.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)  # Warna hijau

        # Menampilkan frame
        cv2.imshow('Stream', frame)

        # Keluar dari loop jika 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            streaming = False  # Set streaming ke False untuk menghentikan loop

    # Membersihkan dan menutup jendela
    cap.release()
    cv2.destroyWindow('Stream')

# Membuat antarmuka Tkinter
root = tk.Tk()
root.title("InerVision")  # Judul jendela
root.geometry("400x200")  # Mengatur ukuran jendela Tkinter

# Menambahkan gaya ke antarmuka
frame = tk.Frame(root, padx=10, pady=10)
frame.pack(pady=20)

# Inisialisasi stream_var sebelum digunakan
stream_var = tk.StringVar(value=list(streams.keys())[0])  # Pilihan awal

# Dropdown untuk memilih stream
dropdown_label = tk.Label(frame, text="Pilih CCTV:", font=("Arial", 12))
dropdown_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

dropdown = ttk.Combobox(frame, textvariable=stream_var, values=list(streams.keys()), state="readonly")
dropdown.grid(row=0, column=1, padx=10, pady=5)

# Tombol untuk memulai stream
start_button = tk.Button(frame, text="Mulai Stream", command=start_stream, font=("Arial", 12), bg="green", fg="white")
start_button.grid(row=1, columnspan=2, pady=10)

# Menjalankan antarmuka
root.mainloop()
