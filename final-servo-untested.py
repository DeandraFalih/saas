import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pytesseract
import requests
import re
from model.tracker import *

pytesseract.pytesseract.tesseract_cmd = r'tesseract\tesseract.exe'

model_plate = YOLO('model/best.pt')
model_car = YOLO('model/yolov8s.pt')

# API endpoints for parking in and parking out
API_ENDPOINT_IN = "https://harezayoankristianto.online/api/parkings/in"
API_ENDPOINT_OUT = "https://harezayoankristianto.online/api/parkings/out"
API_ENDPOINT_OPENGATE = "https://harezayoankristianto.online/api/gates/in"
status_gate = False

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Smart Parking System")
        self.geometry("1580x820")

        # Create main frame
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create left frame for program display
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create right frame for plate crop display
        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Initialize tracker
        self.tracker_in = Tracker()
        self.tracker_out = Tracker()

        # Load COCO class names
        with open("coco/coco.txt", "r") as my_file:
            data = my_file.read()
            self.class_list_car = data.split("\n")
            
        self.existing_ids_in = set()  # to keep track of existing IDs for parking in
        self.existing_ids_out = set()  # to keep track of existing IDs for parking out
        self.captured_cars_in = {}  # to store captured car images for parking in
        self.captured_cars_out = {}  # to store captured car images for parking out
        
        # Open file for reading class list
        with open("coco/coco1.txt", "r") as file:
            data = file.read()
            self.class_list_plate = data.split("\n") 

        # Define areas for different functionalities
        self.area_photo_car_in = [(17, 203), (543, 217), (545, 247), (31, 242)]
        self.area_photo_car_out = [(38, 244), (38, 275), (789, 285), (791, 240)]
        self.area_plate_recognition_in = [(19, 278), (556, 283), (558, 324), (20, 325)]
        self.area_plate_recognition_out = [(60, 260), (60, 290), (800, 290), (800, 260)]

        self.processed_numbers_plate = set()

        self.source_in_var = tk.StringVar()
        self.source_out_var = tk.StringVar()

        self.initialize_gui()

    def initialize_cameras(self, source_in, source_out):
        self.cap_in = cv2.VideoCapture(source_in)  # Camera for parking_in
        self.cap_out = cv2.VideoCapture(source_out)  # Camera for parking_out
        
        # Set frame width and height for both cameras
        frame_width = 320
        frame_height = 240
        self.cap_in.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap_in.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.cap_out.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap_out.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    def initialize_gui(self):
        self.video_label_in_title = tk.Label(self.left_frame, text="Parking In", font=("Arial", 14))
        self.video_label_in_title.pack(side=tk.TOP, padx=10, pady=(0, 5))

        self.video_label_in = tk.Label(self.left_frame)
        self.video_label_in.pack(padx=10, pady=10)

        self.video_label_out_title = tk.Label(self.left_frame, text="Parking Out", font=("Arial", 14))
        self.video_label_out_title.pack(side=tk.TOP, padx=10, pady=(0, 5))

        self.video_label_out = tk.Label(self.left_frame)
        self.video_label_out.pack(padx=10, pady=10)

        self.crop_label_title = tk.Label(self.right_frame, text="Plat Nomor", font=("Arial", 14))
        self.crop_label_title.pack(side=tk.TOP, padx=10, pady=(0, 5))
        
        self.crop_label = tk.Label(self.right_frame)
        self.crop_label.pack(padx=10, pady=10)

        # Create result label
        self.result_label = tk.Label(self.right_frame, text="", font=("Arial", 14))
        self.result_label.pack(pady=5)
        
        self.success_label = tk.Label(self.right_frame, text="", font=("Arial", 12), fg="green")
        self.success_label.pack(padx=10, pady=(0, 5), after=self.result_label)
        
        self.error_label = tk.Label(self.right_frame, text="", font=("Arial", 12), fg="red")
        self.error_label.pack(padx=10, pady=(0, 5))

        # Create photo panel
        self.photo_panel_title = tk.Label(self.right_frame, text="Foto Mobil", font=("Arial", 14))
        self.photo_panel_title.pack(side=tk.TOP, padx=10, pady=30)

        self.photo_panel = tk.Label(self.right_frame)
        self.photo_panel.pack(padx=10, pady=(1, 5))

        # Create input fields for video sources
        self.source_in_label = tk.Label(self.right_frame, text="Parking In Source:", font=("Arial", 12))
        self.source_in_label.pack(pady=5)
        self.source_in_entry = tk.Entry(self.right_frame, textvariable=self.source_in_var, width=20)
        self.source_in_entry.pack(pady=5)
        
        self.source_out_label = tk.Label(self.right_frame, text="Parking Out Source:", font=("Arial", 12))
        self.source_out_label.pack(pady=5)
        self.source_out_entry = tk.Entry(self.right_frame, textvariable=self.source_out_var, width=20)
        self.source_out_entry.pack(pady=5)

        self.initialize_button = tk.Button(self.right_frame, text="Initialize Cameras", command=self.on_initialize_button)
        self.initialize_button.pack(pady=10)
        
        self.update_frame()

    def on_initialize_button(self):
        source_in = self.source_in_var.get()
        source_out = self.source_out_var.get()

        try:
            source_in = int(source_in) if source_in.isdigit() else source_in
        except ValueError:
            source_in = 0

        try:
            source_out = int(source_out) if source_out.isdigit() else source_out
        except ValueError:
            source_out = 1

        if hasattr(self, 'cap_in') and self.cap_in.isOpened():
            self.cap_in.release()
        if hasattr(self, 'cap_out') and self.cap_out.isOpened():
            self.cap_out.release()

        self.initialize_cameras(source_in, source_out)

    def update_frame(self):
        if hasattr(self, 'cap_in') and self.cap_in.isOpened():
            ret_in, frame_in = self.cap_in.read()
            if ret_in:
                frame_in = cv2.resize(frame_in, (560, 340))
                self.process_frame(frame_in, self.area_photo_car_in, self.area_plate_recognition_in, self.tracker_in, self.existing_ids_in, self.captured_cars_in, API_ENDPOINT_IN, self.video_label_in)

        if hasattr(self, 'cap_out') and self.cap_out.isOpened():
            ret_out, frame_out = self.cap_out.read()
            if ret_out:
                frame_out = cv2.resize(frame_out, (560, 340))
                self.process_frame(frame_out, self.area_photo_car_out, self.area_plate_recognition_out, self.tracker_out, self.existing_ids_out, self.captured_cars_out, API_ENDPOINT_OUT, self.video_label_out)

        self.video_label_in.after(10, self.update_frame)

    def process_frame(self, frame, area_photo_car, area_plate_recognition, tracker, existing_ids, captured_cars, api_endpoint, video_label):
        detected = self.plate_recognition(frame, area_plate_recognition, api_endpoint)
        if detected:
            status_gate = True
        else:
            status_gate = False
        try:
            response = requests.post(API_ENDPOINT_OPENGATE, json={'gateStatus': status_gate})
            if response.status_code == 201:
                print("GATE OPEN.")
            else:
                print("Failed to send gate status to API:", response.status_code)
        except Exception as e:
            print("Error occurred while sending gate status to API:", e)
        
        self.photo_car(frame, area_photo_car, tracker, existing_ids, captured_cars)
        cv2.polylines(frame, [np.array(area_photo_car, np.int32)], True, (255, 0, 0), 2)
        cv2.polylines(frame, [np.array(area_plate_recognition, np.int32)], True, (0, 255, 0), 2)

        # Convert frame to RGB format for tkinter display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_imgtk = ImageTk.PhotoImage(image=frame_pil)

        # Update video label with the new image
        video_label.imgtk = frame_imgtk
        video_label.configure(image=frame_imgtk)

    def plate_recognition(self, frame, area, api_endpoint):
        results = model_plate.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")
        
        detected = False  # Flag to track detection

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            
            d = int(row[5])
            c = self.class_list_plate[d]
            cx = int(x1 + x2) // 2
            cy = int(y1 + y2) // 2
            result = cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)
            if result >= 0:
                crop = frame[y1:y2, x1:x2]
                crop_resized = cv2.resize(crop, (260, 85))
                gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray, 10, 20, 20)
                
                #difokuskan plat indo
                text = pytesseract.image_to_string(gray).strip()
                match = re.search(r'[A-Z]{1,2}\s?\d{1,4}\s?[A-Z0-9]{1,3}', text.upper())
                if match:
                    text = match.group(0).replace(' ', '')
                    text = text[:9]
                else:
                    continue
                
                print(text)
                if text not in self.processed_numbers_plate:
                    self.processed_numbers_plate.add(text) 
                    detected = True  # Plate detected

                    # Convert cropped image to RGB format for tkinter display
                    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
                    crop_pil = Image.fromarray(crop_rgb)
                    crop_imgtk = ImageTk.PhotoImage(image=crop_pil)

                    # Update crop label with the new image
                    self.crop_label.imgtk = crop_imgtk
                    self.crop_label.configure(image=crop_imgtk)

                    # Update text label for plate number
                    result_text = f"Plate Number: {text}"
                    self.result_label.config(text=result_text)

                    # Send plate number to API endpoint
                    payload = {'code': text}
                    try:
                        response = requests.post(api_endpoint, json=payload)
                        if response.status_code == 201:
                            self.success_label.config(text="Plate number sent successfully to API.", fg="green")
                            print("Plate number sent successfully to API.")
                        else:
                            self.error_label.config(text=f"Failed to send plate number to API: {response.status_code}", fg="red")
                            print("Failed to send plate number to API:", response.status_code)
                    except Exception as e:
                        self.error_label.config(text=f"Error occurred while sending plate number to API: {e}", fg="red")
                        print("Error occurred while sending plate number to API:", e)    
        return detected

    def update_photo_frame(self, photo):
        photo = cv2.resize(photo, (480, 270))  # Resize photo for display 480, 270
        photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)  # Convert photo to RGB for tkinter
        img = Image.fromarray(photo)
        imgtk = ImageTk.PhotoImage(image=img)
        self.photo_panel.imgtk = imgtk
        self.photo_panel.configure(image=imgtk)

    def photo_car(self, frame, area, tracker, existing_ids, captured_cars):
        results = model_car.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        car_boxes = []
        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = self.class_list_car[d]
            if 'car' in c:
                car_boxes.append([x1, y1, x2, y2])

        bbox_idx = tracker.update(car_boxes)
        for bbox in bbox_idx:
            x3, y3, x4, y4, id1 = bbox
            if id1 not in existing_ids:  # Check if ID is new
                cx = int(x3 + x4) // 2
                cy = int(y3 + y4) // 2
                result = cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)
                if result >= 0:
                    # Draw bounding box and label
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 255), 2)
                    cvzone.putTextRect(frame, f'{id1}', (x3, y3), 1, 1)
                    existing_ids.add(id1)  # Add ID to existing IDs

                    # Capture car image
                    car_image = frame[y3:y4, x3:x4]
                    captured_cars[id1] = car_image
                    self.update_photo_frame(car_image)

    def close(self):
        if hasattr(self, 'cap_in') and self.cap_in.isOpened():
            self.cap_in.release()
        if hasattr(self, 'cap_out') and self.cap_out.isOpened():
            self.cap_out.release()
        self.destroy()

if __name__ == "__main__":
    app = Application()
    app.mainloop()
