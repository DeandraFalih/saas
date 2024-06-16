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
model_car = YOLO('model/yolov8n.pt')

API_ENDPOINT = "https://harezayoankristianto.online/api/parkings/in"
#API_ENDPOINT = "https://harezayoankristianto.online/api/parkings/out"
API_ENDPOINT_OPENGATE = "https://harezayoankristianto.online/api/gates/in"
status_gate = False

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Smart Parking System")
        self.geometry("1580x720")

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
        self.tracker = Tracker()

        # Load COCO class names
        with open("coco/coco.txt", "r") as my_file:
            data = my_file.read()
            self.class_list_car = data.split("\n")
            
        self.existing_ids = set()  # to keep track of existing IDs
        self.captured_cars = {}  # to store captured car images
        
        # Open file for reading class list
        with open("coco/coco1.txt", "r") as file:
            data = file.read()
            self.class_list_plate = data.split("\n") 

        self.area = [(27, 323), (27, 428), (798, 430), (795, 319)]
        self.area1 = [(38, 244), (38, 275), (789, 285), (791, 240)]
        self.processed_numbers_plate = set()
        self.processed_numbers_car = set()

        self.initialize_camera()
        self.initialize_gui()
 
    def initialize_camera(self):
        self.cap = cv2.VideoCapture(0)

    def initialize_gui(self):
        self.video_label_title = tk.Label(self.left_frame, text="Output Video", font=("Arial", 14))
        self.video_label_title.pack(side=tk.TOP, padx=10, pady=(0, 5))

        self.video_label = tk.Label(self.left_frame)
        self.video_label.pack(padx=10, pady=10)

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

        self.update_frame()

    def update_frame(self):
        global status_gate
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (860, 540))

            # Call plate_recognition function to process the frame
            plate_detected = self.plate_recognition(frame)
            self.photo_car(frame)

            # Draw area on the frame
            cv2.polylines(frame, [np.array(self.area, np.int32)], True, (255, 0, 0), 2)
            cv2.polylines(frame, [np.array(self.area1, np.int32)], True, (0, 255, 0), 2)

            # Convert frame to RGB format for tkinter display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_imgtk = ImageTk.PhotoImage(image=frame_pil)

            # Update video label with the new image
            self.video_label.imgtk = frame_imgtk
            self.video_label.configure(image=frame_imgtk)

            if plate_detected:
                status_gate = True
            else:
                status_gate = False

            # Send gate status to API without updating GUI labels
            try:
                response = requests.post(API_ENDPOINT_OPENGATE, json={'gateStatus': status_gate})
                if response.status_code == 201:
                    print("GATE OPEN.")
                else:
                    print("Failed to send gate status to API:", response.status_code)
            except Exception as e:
                print("Error occurred while sending gate status to API:", e)

        self.video_label.after(10, self.update_frame)

    def plate_recognition(self, frame):
        plate_detected = False
        results = model_plate.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            
            d = int(row[5])
            c = self.class_list_plate[d]
            cx = int(x1 + x2) // 2
            cy = int(y1 + y2) // 2
            result = cv2.pointPolygonTest(np.array(self.area, np.int32), ((cx, cy)), False)
            if result >= 0:
                plate_detected = True
                crop = frame[y1:y2, x1:x2]
                crop_resized = cv2.resize(crop, (260, 85))
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray, 10, 20, 20)

                #hanya alpabet dan angka, max 9 char
                # text = pytesseract.image_to_string(gray).strip()
                # text = re.sub(r'[^A-Z0-9]', '', text.upper())[:9]
                
                #difokuskan plat indo
                text = pytesseract.image_to_string(gray).strip()
                match = re.search(r'[A-Z]{1,2}\s?\d{1,4}\s?[A-Z0-9]{1,3}', text.upper())
                if match:
                    text = match.group(0).replace(' ', '')
                    text = text[:9]  # Limit to maximum of 9 characters
                else:
                    continue
                
                if text not in self.processed_numbers_plate:
                    self.processed_numbers_plate.add(text) 

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

                    # Send plate number to API and update status in GUI
                    payload = {'code': text}
                    try:
                        response = requests.post(API_ENDPOINT, json=payload)
                        if response.status_code == 201:
                            self.success_label.config(text="Plate number sent successfully to API.", fg="green")
                            print("Plate number sent successfully to API.")
                        else:
                            self.error_label.config(text=f"Failed to send plate number to API: {response.status_code}", fg="red")
                            print("Failed to send plate number to API:", response.status_code)
                    except Exception as e:
                        self.error_label.config(text=f"Error occurred while sending plate number to API: {e}", fg="red")
                        print("Error occurred while sending plate number to API:", e)                
        return plate_detected
    
    def update_photo_frame(self, photo):
        photo = cv2.resize(photo, (480, 270))  # Resize photo for display
        photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)  # Convert photo to RGB for tkinter
        img = Image.fromarray(photo)
        imgtk = ImageTk.PhotoImage(image=img)
        self.photo_panel.imgtk = imgtk
        self.photo_panel.configure(image=imgtk)

    def photo_car(self, frame):
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

        bbox_idx = self.tracker.update(car_boxes)
        for bbox in bbox_idx:
            x3, y3, x4, y4, id1 = bbox
            if id1 not in self.existing_ids:  # Check if ID is new
                cx = int(x3 + x4) // 2
                cy = int(y3 + y4) // 2
                result = cv2.pointPolygonTest(np.array(self.area1, np.int32), ((cx, cy)), False)
                if result >= 0:
                    # Draw bounding box and label
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 255), 2)
                    cvzone.putTextRect(frame, f'{id1}', (x3, y3), 1, 1)
                    self.existing_ids.add(id1)  # Add ID to existing IDs

                    # Capture car image
                    car_image = frame[y3:y4, x3:x4]
                    self.captured_cars[id1] = car_image
                    self.update_photo_frame(car_image)

        cv2.polylines(frame, [np.array(self.area1, np.int32)], True, (0, 255, 0), 2)

        # Update video frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for tkinter
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def close(self):
        self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = Application()
    app.mainloop()
