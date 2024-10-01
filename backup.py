import streamlit as st
import cv2
import torch
import numpy as np
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode
import subprocess

# Kelas untuk pemrosesan video menggunakan YOLOv5
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        # Memuat model YOLOv5
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        st.write("Model YOLOv5 dimuat.")

    def transform(self, frame):
        # Menampilkan informasi tentang frame
        st.write(f"Frame shape: {frame.shape}")

        # Mengonversi frame OpenCV menjadi tensor
        img = [frame[..., ::-1]]  # BGR ke RGB
        results = self.model(img)

        # Cek hasil deteksi
        if len(results.xyxy[0]) == 0:
            st.write("Tidak ada objek yang terdeteksi.")
        else:
            st.write(f"Jumlah objek terdeteksi: {len(results.xyxy[0])}")

        conf_threshold = 0.4  # Ubah sesuai kebutuhan

        for *xyxy, conf, cls in results.xyxy[0]:
            if conf >= conf_threshold:  # Filter berdasarkan threshold
                label = f'{self.model.names[int(cls)]}: {conf:.2f}'
                frame = self.draw_box(frame, int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), label)

        return frame

    def draw_box(self, img, x1, y1, x2, y2, label):
        # Menggambar bounding box dan label
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Gambar kotak
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Gambar label
        return img

# Fungsi untuk menjalankan streaming video dari URL HLS
def get_video_stream(url):
    command = ["streamlink", url, "best", "--stdout"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process

# URL stream HLS
url = "https://atcs.cianjurkab.go.id:5443/LiveApp/streams/SP4_Dishub_PTZ_1.m3u8"

# Streamlit UI
st.title("WebRTC Object Detection with YOLOv5")

# Mengambil video stream dari URL
video_process = get_video_stream(url)

webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=VideoTransformer().transform,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if __name__ == "__main__":
    st.write("Streaming video dengan deteksi objek menggunakan YOLOv5.")
