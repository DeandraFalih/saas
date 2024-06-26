import tkinter as tk
from tkinter import ttk
from ultralytics import YOLO
import cv2
from PIL import Image, ImageTk
import pytesseract
import re

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Smart Parking System")
        self.geometry("1600x600")  # Adjusted to fit two video feeds side by side

        # Create main frame
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create left frame for program display
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create video label for first camera
        self.video_label1 = tk.Label(self.left_frame)
        self.video_label1.pack(padx=10, pady=10)

        # Create label for displaying detected license plate for first camera
        self.plate_label1 = tk.Label(self.left_frame, text="Detected License Plate (Cam 1): ", font=("Helvetica", 16))
        self.plate_label1.pack(padx=10, pady=10)

        # Create right frame for second camera display
        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create video label for second camera
        self.video_label2 = tk.Label(self.right_frame)
        self.video_label2.pack(padx=10, pady=10)

        # Create label for displaying detected license plate for second camera
        self.plate_label2 = tk.Label(self.right_frame, text="Detected License Plate (Cam 2): ", font=("Helvetica", 16))
        self.plate_label2.pack(padx=10, pady=10)

        # Initialize YOLO model
        self.model = YOLO('data/model/best.pt')

        # Initialize webcams
        self.cap1 = cv2.VideoCapture(1)
        self.cap2 = cv2.VideoCapture(2)
        self.update_frame()

    def update_frame(self):
        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()

        if ret1:
            print("Camera 1 frame captured.")
            detected_plate1 = self.process_frame(frame1, "Camera 1")
            
            # Resize frame to 320x240
            frame1 = cv2.resize(frame1, (560, 340))

            # Convert first frame to ImageTk format
            img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            img1 = Image.fromarray(img1)
            img_tk1 = ImageTk.PhotoImage(image=img1)
            
            # Update video label with new frame from first camera
            self.video_label1.img_tk = img_tk1
            self.video_label1.config(image=img_tk1)

            # Update plate label with detected license plate text from first camera
            self.plate_label1.config(text=f"Parking In: {detected_plate1}")
        else:
            print("Failed to capture frame from Camera 1.")

        if ret2:
            print("Camera 2 frame captured.")
            detected_plate2 = self.process_frame(frame2, "Camera 2")
            
            # Resize frame to 320x240
            frame2 = cv2.resize(frame2, (560, 340))

            # Convert second frame to ImageTk format
            img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            img2 = Image.fromarray(img2)
            img_tk2 = ImageTk.PhotoImage(image=img2)
            
            # Update video label with new frame from second camera
            self.video_label2.img_tk = img_tk2
            self.video_label2.config(image=img_tk2)

            # Update plate label with detected license plate text from second camera
            self.plate_label2.config(text=f"Parking Out: {detected_plate2}")
        else:
            print("Failed to capture frame from Camera 2.")

        self.after(10, self.update_frame)

    def process_frame(self, frame, camera_name):
        detected_plate = ""
        
        # Define the region of interest (ROI) - adjust coordinates as needed
        roi_points = [(50, 50), (frame.shape[1] - 50, 50), (frame.shape[1] - 50, frame.shape[0] - 50), (50, frame.shape[0] - 50)]
        
        # Draw the ROI on the frame
        cv2.polylines(frame, [np.array(roi_points, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Run YOLO model on the frame
        results = self.model(frame)

        # Extract bounding boxes and labels
        for result in results:
            for box in result.boxes:
                coords = box.xyxy[0]  # Extract the coordinates from the tensor
                x1, y1, x2, y2 = map(int, coords.tolist())  # Convert to list and then to int

                # Check if the detected box is within the ROI
                if self.is_within_roi((x1, y1, x2, y2), roi_points):
                    cropped_plate = frame[y1:y2, x1:x2]
                    
                    # Use Tesseract to read text from the license plate
                    plate_text = pytesseract.image_to_string(cropped_plate, config='--psm 8').strip()
                    # Match the plate text with Indonesian license plate format
                    match = re.search(r'[A-Z]{1,2}\s?\d{1,4}\s?[A-Z0-9]{1,3}', plate_text.upper())
                    if match:
                        plate_text = match.group(0).replace(' ', '')
                        plate_text = plate_text[:9]  # Limit to a maximum of 9 characters
                        detected_plate = plate_text
                        print(f"Detected License Plate ({camera_name}): {plate_text}")

                    # Draw bounding box on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return detected_plate

    def is_within_roi(self, box, roi_points):
        x1, y1, x2, y2 = box
        box_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        
        # Check if the box center is within the polygon defined by roi_points
        return cv2.pointPolygonTest(np.array(roi_points, dtype=np.int32), box_center, False) >= 0

    def close(self):
        self.cap1.release()
        self.cap2.release()
        self.destroy()

if __name__ == "__main__":
    import numpy as np
    app = Application()
    app.mainloop()
