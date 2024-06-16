import cv2
import time

# Inisialisasi video capture dari webcam (ganti argumen dengan file video jika ingin menampilkan video dari file)
#cap = cv2.VideoCapture("rtsp://192.168.100.146:8080/h264_pcm.sdp")
cap = cv2.VideoCapture(1)

# Inisialisasi variabel untuk menghitung FPS
fps = 0
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Hitung FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        start_time = time.time()
        frame_count = 0

    # Tambahkan teks FPS pada frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Tampilkan frame
    cv2.imshow('Video', frame)

    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan resources
cap.release()
cv2.destroyAllWindows()
