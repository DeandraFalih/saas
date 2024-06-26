import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, simpledialog
from PIL import Image
import numpy as np
import json
import os
import cv2
import re
import uuid
import cvzone
import threading
import pandas as pd
import requests
from ultralytics import YOLO
import pytesseract
import time

'''
modal
'''
def manual_book():
    modal = ctk.CTkToplevel(app)
    modal.title("")
    modal.geometry(geometry_center(300, 250))
    modal.resizable(False, False)
    
    frame_header = ctk.CTkFrame(modal, bg_color="#18181b", fg_color="#18181b")
    frame_header.pack(fill="x")
    
    frame_body = ctk.CTkFrame(modal, bg_color="#27272a", fg_color="#27272a")
    frame_body.pack(expand="true", fill="both")
    
    frame_footer = ctk.CTkFrame(modal, bg_color="#18181b", fg_color="#18181b")
    frame_footer.pack(fill="x")

    image_path = os.path.join(os.getcwd(), "data", "icons", "form.png")
    image = Image.open(image_path)
    resized_image = image.resize((20, 20))
    
    image_ctk = ctk.CTkImage(light_image=resized_image, size=(20, 20))

    image_header = ctk.CTkLabel(frame_header, image=image_ctk, text="")
    image_header.pack(side="left", fill="x", padx=(20,0), pady=(20))
    
    header = ctk.CTkLabel(frame_header, text="Manual Book", font=("Helvetica", 15, "bold"))
    header.pack(side="left", fill="x", padx=10, pady=20) 
    
    style.configure("Treeview.Heading", font=("Helvetica", 10, "bold"))
    
    tree = ttk.Treeview(frame_body, columns=("#", "Key", "Function"), show="headings", height=0)
    tree.heading("#", text="#")
    tree.heading("Key", text="Key")
    tree.heading("Function", text="Function")
    
    tree.column("#", width=30, stretch=False, anchor="center")
    tree.column("Key", width=80, stretch=True)
    tree.column("Function", stretch=True)
    
    style.configure("Treeview", rowheight=25, font=("Helvetica", 10), foreground="#1f2937")
    style.map("Treeview", background=[("selected", "#4ade80")], foreground=[("selected", "#000")])
    tree.tag_configure("evenrow", background="#d1d5db")
    tree.tag_configure("oddrow", background="#e5e7eb")

    datas = [
        {
            "key":"i",
            "function":"Show manual book"
        },
        {
            "key":"left click",
            "function":"Draw the polyline"
        },
        {
            "key":"right click",
            "function":"Delete the polyline"
        },
        {
            "key":"s",
            "function":"Save"
        },
        {
            "key":"q",
            "function":"Quit"
        },
    ]
    
    for i, data in enumerate(datas):
        index = i+1
        values = (index, data["key"], data["function"])
        if i % 2 == 0:
            tree.insert("", "end", values=values, tags="evenrow")
        else:
            tree.insert("", "end", values=values, tags="oddrow")
        
    tree.pack(expand="true", fill="both", padx=10, pady=10)
    
    def close():
        modal.destroy()
        
    modal.protocol("WM_DELETE_WINDOW", lambda: close())
    
    btn = ctk.CTkButton(frame_footer, text="Hide", width=60, fg_color="#4ade80", text_color="#000", hover_color="#16a34a",command=lambda: close())
    btn.pack(side="right", padx=(10), pady=10)
    
    modal.transient(app)
    modal.grab_set()
    modal.focus()
    app.wait_window(modal)

def modal_okcancel(body):
    result = [None]
    modal = ctk.CTkToplevel(app)
    modal.title("")
    modal.geometry(geometry_center(300, 180))
    modal.resizable(False, False)
    
    frame_header = ctk.CTkFrame(modal, bg_color="#18181b", fg_color="#18181b")
    frame_header.pack(fill="x")
    
    frame_body = ctk.CTkFrame(modal, bg_color="#27272a", fg_color="#27272a")
    frame_body.pack(expand="true", fill="both")
    
    frame_footer = ctk.CTkFrame(modal, bg_color="#18181b", fg_color="#18181b")
    frame_footer.pack(fill="x")
    frame_footer.grid_columnconfigure((0,1), weight=1)

    # Memuat dan mengubah ukuran gambar menggunakan PIL
    image_path = os.path.join(os.getcwd(), "data", "icons", "warning.png")
    image = Image.open(image_path)
    resized_image = image.resize((20, 20))
    
    # Menggunakan CTkImage untuk memastikan kompatibilitas HighDPI
    image_ctk = ctk.CTkImage(light_image=resized_image, size=(20, 20))

    image_header = ctk.CTkLabel(frame_header, image=image_ctk, text="")
    image_header.pack(side="left", fill="x", padx=(20,0), pady=(20))
    
    header = ctk.CTkLabel(frame_header, text="Warning", font=("Helvetica", 15, "bold"))
    header.pack(side="left", fill="x", padx=10, pady=20) 
    
    msg = ctk.CTkLabel(frame_body, text=body, font=("Helvetica", 15), anchor="w")
    msg.pack(expand="true", fill="both", padx=40)
    
    def close(value):
        result[0] = value
        modal.destroy()
        
    modal.protocol("WM_DELETE_WINDOW", lambda: close(None))
    
    btn = ctk.CTkButton(frame_footer, text="Cancel", width=60, fg_color="#4ade80", text_color="#000", hover_color="#16a34a",command=lambda: close(None))
    btn.pack(side="right", padx=(10), pady=10)
    
    btn = ctk.CTkButton(frame_footer, text="Ok", width=60, fg_color="#27272a", border_width=1, border_color="#f87171", hover_color="#dc2626", command=lambda: close("yes"))
    btn.pack(side="right", padx=0, pady=10)
    
    
    modal.transient(app)
    modal.grab_set()
    modal.focus()
    app.wait_window(modal)
    
    return result[0]
    
def modal_input(body):
    result = [None]

    modal = ctk.CTkToplevel(app)
    modal.title("")
    modal.geometry(geometry_center(300, 180))
    modal.resizable(False, False)
    
    frame_header = ctk.CTkFrame(modal, bg_color="#18181b", fg_color="#18181b")
    frame_header.pack(fill="x")
    
    frame_body = ctk.CTkFrame(modal, bg_color="#27272a", fg_color="#27272a")
    frame_body.pack(expand="true", fill="both")
    frame_body.grid_columnconfigure(1, weight=1)
    frame_body.grid_rowconfigure(0, weight=1)
    
    frame_footer = ctk.CTkFrame(modal, bg_color="#18181b", fg_color="#18181b")
    frame_footer.pack(fill="x")
    frame_footer.grid_columnconfigure((0,1), weight=1)

    # Memuat dan mengubah ukuran gambar menggunakan PIL
    image_path = os.path.join(os.getcwd(), "data", "icons", "form.png")
    image = Image.open(image_path)
    resized_image = image.resize((20, 20))
    
    # Menggunakan CTkImage untuk memastikan kompatibilitas HighDPI
    image_ctk = ctk.CTkImage(light_image=resized_image, size=(20, 20))

    image_header = ctk.CTkLabel(frame_header, image=image_ctk, text="")
    image_header.pack(side="left", fill="x", padx=(20,0), pady=(20))
    
    header = ctk.CTkLabel(frame_header, text="Form", font=("Helvetica", 15, "bold"))
    header.pack(side="left", fill="x", padx=10, pady=20) 
    
    input_label = ctk.CTkLabel(frame_body, text=body, font=("Helvetica", 12))
    input_label.grid(row=0, column=0, padx=(30,5), sticky="e")
    
    input_entry = ctk.CTkEntry(frame_body)
    input_entry.grid(row=0, column=1, padx=(0, 30), sticky="we")
    modal.after(100, input_entry.focus_set)
    
    def close(value):
        result[0] = value
        modal.destroy()
        
    modal.protocol("WM_DELETE_WINDOW", lambda: close(None))
    
    btn = ctk.CTkButton(frame_footer, text="Ok", width=60, fg_color="#4ade80", text_color="#000", hover_color="#16a34a", command=lambda: close(input_entry.get()))
    btn.pack(side="right", padx=10, pady=10)
    
    btn = ctk.CTkButton(frame_footer, text="Cancel", width=60, fg_color="#27272a", border_width=1, border_color="#f87171", hover_color="#dc2626", command=lambda: close(None))
    btn.pack(side="right", padx=(0), pady=10)
    
    modal.transient(app)
    modal.grab_set()
    modal.focus()
    app.wait_window(modal)
    
    return result[0]
    
def modal_alert(status, message):
    modal = ctk.CTkToplevel(app)
    modal.title("")
    modal.geometry(geometry_center(300, 180))
    modal.resizable(False, False)
    
    frame_header = ctk.CTkFrame(modal, bg_color="#18181b", fg_color="#18181b")
    frame_header.pack(fill="x")
    
    frame_body = ctk.CTkFrame(modal, bg_color="#27272a", fg_color="#27272a")
    frame_body.pack(expand="true", fill="both")
    
    frame_footer = ctk.CTkFrame(modal, bg_color="#18181b", fg_color="#18181b")
    frame_footer.pack(fill="x")

    # Memuat dan mengubah ukuran gambar menggunakan PIL
    image_path = os.path.join(os.getcwd(), "data", "icons", "warning.png")
    if status == "success":
        image_path = os.path.join(os.getcwd(), "data", "icons", "success.png")
    elif status == "failed":
        image_path = os.path.join(os.getcwd(), "data", "icons", "failed.png")
    image = Image.open(image_path)
    resized_image = image.resize((20, 20))
    
    # Menggunakan CTkImage untuk memastikan kompatibilitas HighDPI
    image_ctk = ctk.CTkImage(light_image=resized_image, size=(20, 20))

    image_header = ctk.CTkLabel(frame_header, image=image_ctk, text="")
    image_header.pack(side="left", fill="x", padx=(20,0), pady=(20))
    
    header = ctk.CTkLabel(frame_header, text=status.title(), font=("Helvetica", 15, "bold"))
    header.pack(side="left", fill="x", padx=10, pady=20) 
    
    msg = ctk.CTkLabel(frame_body, text=message, font=("Helvetica", 13), wraplength=300)
    msg.pack(expand="true", fill="both", padx=20)
    
    def close():
        modal.destroy()
    
    btn = ctk.CTkButton(frame_footer, text="Ok", width=60, fg_color="#4ade80", text_color="#000", hover_color="#16a34a",command=lambda: close())
    btn.pack(side="right", padx=(10), pady=10)
    
    modal.transient(app)
    modal.grab_set()
    modal.focus()
    app.wait_window(modal)


'''
file management
'''
def folder_check(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder not found : {folder_path}")
        try:
            os.mkdir(folder_path)
            print(f"Folder created : {folder_path}")
        except Exception as e:
            print(f"Folder failed to create : {e}")
        return False
    else:
        return True

def file_check(file_path):
    if not os.path.isfile(file_path):
        print(f"File not found : {file_path}")
        return False
    else:
        return True

def save_json(data):
    with open(data["file_path"], 'w') as f:
        json.dump(data, f)

def get_json(file_path):
    if file_check(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

def get_all_json(folder_path):
    json_files = []
    if folder_check(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".json"):
                file_path = os.path.join(folder_path, file_name)
                json_file = get_json(file_path)
                json_files.append(json_file)
    return json_files

def get_all_parking_names():
    parking_names = []
    file_path = os.path.join(os.getcwd(), "data", "json", "parking_detections")
    parkings = get_all_json(file_path)
    
    for parking in parkings:
        parking_names.append(parking["parking_name"])
    
    return parking_names

def get_all_space_names():
    space_names = []
    file_path = os.path.join(os.getcwd(), "data", "json", "parking_detections")
    parkings = get_all_json(file_path)
    
    for parking in parkings:
        if "space_names" in parking:
            space_names.extend(parking["space_names"])
    
    return space_names

'''
table management
'''
def select_rows():
    row_ids = []
    datas = []
    
    select_rows = tree.selection()
    
    if select_rows:
        for row_id in select_rows:
            row_ids.append(row_id)
        for row_id in row_ids:
            row = tree.item(row_id)
            values = row["values"]
            if values:
                i, parking_name, source, total_area, file_name, file_path = values
                datas.append(
                    {
                        "parking_name":parking_name,
                        "source":source,
                        "file_name":file_name,
                        "file_path":file_path
                    }
                )
            else:
                modal_alert("warning", "Values is null")
    else:
        modal_alert("warning", "No data selected")
    return datas

'''
api management
'''
def request_api(url, payload, success, failed):
    try:
        response = requests.post(f"https://harezayoankristianto.online/api{url}", json=payload, timeout=10)
        #response = requests.post(f"http://localhost:5000/api{url}", json=payload, timeout=10)
        response.raise_for_status()
        print(f"{success} : {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"{failed} : {e}")

'''
parking detection management
'''
def setup_parking_detection(parking_name, source):
    video = cv2.VideoCapture(source if not source.isdigit() else int(source))
    
    if not video.isOpened():
        modal_alert("failed", "Failed to start video")
        return

    app.withdraw()
    
    modal_active = True
    manual_book()
    modal_active = False
    
    area_id = str(uuid.uuid4())
    file_name = f'{parking_name.replace(" ", "_")}.json'
    folder_path = os.path.join(os.getcwd(), "data", "json", "parking_detections")
    file_path = os.path.join(folder_path, file_name)
    space_names = []
    polylines = []
    
    coordinates = []
    drawing = False

    def draw_parking_mapping(event, x, y, flags, param):
        nonlocal polylines, space_names, coordinates, drawing, modal_active
        
        if modal_active:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            coordinates = [(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                coordinates.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if coordinates:
                modal_active = True
                current_name = modal_input("Space name")
                modal_active = False
                if current_name:
                    current_name = current_name.upper()
                    if current_name in get_all_space_names() or current_name in space_names:
                        modal_active = True
                        modal_alert("warning", f"{current_name} already exist")
                        modal_active = False
                    else:
                        space_names.append(current_name)
                        polylines.append(np.array(coordinates, np.int32))
                coordinates = []
        elif event == cv2.EVENT_RBUTTONDOWN:
            for i, polyline in enumerate(polylines):
                polyline_target = cv2.pointPolygonTest(polyline, (x, y), False)
                if polyline_target >= 0:
                    del polylines[i]
                    del space_names[i]
                    break

    while True:
        ret, frame = video.read()
        if not ret:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame = cv2.resize(frame, (720, 480))

        for i, polyline in enumerate(polylines):
            cv2.polylines(frame, [polyline], True, (0, 0, 255), 2)
            cvzone.putTextRect(frame, f'{space_names[i]}', tuple(polyline[0]), 1, 1)

        cv2.imshow('FRAME', frame)
        cv2.setMouseCallback('FRAME', draw_parking_mapping)

        key = cv2.waitKey(100) & 0xFF

        if key == 115 or key == 83:
            data = {
                'area_id': area_id,
                'parking_name': parking_name,
                'source': source,
                'file_name': file_name,
                'folder_path': folder_path,
                'file_path': file_path,
                'space_names': space_names,
                'polylines': [polyline.tolist() for polyline in polylines],
            }
            folder_check(folder_path)
            save_json(data)
            modal_active = True
            modal_alert("success", f"{parking_name} success to save")
            modal_active = False
            break
        elif key == 113 or key == 81:
            modal_active = True
            decision = modal_okcancel(f"Cancel create {parking_name}")
            modal_active = False
            if decision == "yes":
                break
        elif key == 105 or key == 73:
            modal_active = True
            manual_book()
            modal_active = False

    video.release()
    cv2.destroyAllWindows()
    create_data()
    app.deiconify()
    
def setup_parking_detection_update(parking_name, source, old_file_path):
    video = cv2.VideoCapture(source if not source.isdigit() else int(source))
    
    if not video.isOpened():
        modal_alert("failed", "Failed to start video")
        return

    app.withdraw()
    
    modal_active = True
    manual_book()
    modal_active = False
    
    try:
        parking = get_json(old_file_path)
        area_id = parking["area_id"]
        space_names = parking["space_names"]
        polylines = [np.array(polyline, np.int32) for polyline in parking['polylines']]
    except FileNotFoundError:
        area_id = str(uuid.uuid4())
        space_names = []
        polylines = []
    
    file_name = f'{parking_name.replace(" ", "_")}.json'
    folder_path = os.path.join(os.getcwd(), "data", "json", "parking_detections")
    file_path = os.path.join(folder_path, file_name)
    
    coordinates = []
    drawing = False

    def draw_parking_mapping(event, x, y, flags, param):
        nonlocal polylines, space_names, coordinates, drawing, modal_active
        
        if modal_active:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            coordinates = [(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                coordinates.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if coordinates:
                modal_active = True
                current_name = modal_input("Space name")
                modal_active = False
                if current_name:
                    current_name = current_name.upper()
                    if current_name in get_all_space_names() or current_name in space_names:
                        modal_active = True
                        modal_alert("warning", f"{current_name} already exist")
                        modal_active = False
                    else:
                        space_names.append(current_name)
                        polylines.append(np.array(coordinates, np.int32))
                coordinates = []
        elif event == cv2.EVENT_RBUTTONDOWN:
            for i, polyline in enumerate(polylines):
                polyline_target = cv2.pointPolygonTest(polyline, (x, y), False)
                if polyline_target >= 0:
                    del polylines[i]
                    del space_names[i]
                    break

    while True:
        ret, frame = video.read()
        if not ret:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame = cv2.resize(frame, (720, 480))

        for i, polyline in enumerate(polylines):
            cv2.polylines(frame, [polyline], True, (0, 0, 255), 2)
            cvzone.putTextRect(frame, f'{space_names[i]}', tuple(polyline[0]), 1, 1)

        cv2.imshow('FRAME', frame)
        cv2.setMouseCallback('FRAME', draw_parking_mapping)

        key = cv2.waitKey(100) & 0xFF

        if key == 115 or key == 83:
            try:
                os.remove(old_file_path)
                print("Old file success to delete")
            except Exception as e:
                print(f"Old file failed to delete : {e}")
            data = {
                'area_id': area_id,
                'parking_name': parking_name,
                'source': source,
                'file_name': file_name,
                'folder_path': folder_path,
                'file_path': file_path,
                'space_names': space_names,
                'polylines': [polyline.tolist() for polyline in polylines],
            }
            folder_check(folder_path)
            save_json(data)
            modal_active = True
            modal_alert("success", f"{parking_name} success to update")
            modal_active = False
            break
        elif key == 113 or key == 81:
            modal_active = True
            decision = modal_okcancel(f"Cancel update {parking_name}")
            modal_active = False
            if decision == "yes":
                break
        elif key == 105 or key == 73:
            modal_active = True
            manual_book()
            modal_active = False
    
    updated_parking = [
        {
            "parking_name":parking_name,
            "source":source,
            "file_name":file_name,
            "folder_path":folder_path,
            "file_path":file_path
        }
    ]

    video.release()
    cv2.destroyAllWindows()
    update_data(updated_parking)
    app.deiconify()
    
def detection_parking(file_path, stop_event):
    parking = get_json(file_path)
    if not parking:
        return
    
    parking_name = parking["parking_name"]
    polylines = [np.array(polyline, np.int32) for polyline in parking['polylines']]
    space_names = parking['space_names']
    source = parking['source']

    with open(os.path.join(os.getcwd(), "data", "coco", "coco.txt"), "r") as my_file:
        class_list = my_file.read().split("\n")

    model = YOLO(os.path.join(os.getcwd(), "data", "model", "yolov8s.pt"))

    video = cv2.VideoCapture(source if not source.isdigit() else int(source))
    
    if not video.isOpened():
        modal_alert("failed", "Failed to start video")
        return

    while not stop_event.is_set():
        ret, frame = video.read()
        if not ret:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (720, 480))
        result = model.predict(frame)
        bounding_boxes = result[0].boxes.data.cpu().numpy()
        df = pd.DataFrame(bounding_boxes).astype("float")

        list2 = []
        car = []
        filled_space = []
        empty_space = []
        parking_area = []

        for i, row in df.iterrows():
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            d = int(row[5])
            c = class_list[d]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if 'car' in c:
                list2.append([cx, cy])

        for i, polyline in enumerate(polylines):
            parking_area.append(i)
            for cx1, cy1 in list2:
                result = cv2.pointPolygonTest(polyline, (cx1, cy1), False)
                if result >= 0:
                    car.append(cx1)
                    filled_space.append(space_names[i])
                    empty_space = [x for x in space_names if x not in filled_space]

        # datas = f"Parking : {len(parking_area) - len(car)}/{len(parking_area)}\nFilled Space : {len(filled_space)} : {filled_space}\nEmpty Space : {len(empty_space)} : {empty_space}"
        # print(datas)
        
        payload = []
        for space in space_names:
            payload.append({
                "space_name":space,
                "status": True if space in filled_space else False
            })
        # print(parking_name)
        request_api("/parking_spaces", payload, f"{parking_name} | Space success to update", f"{parking_name} | Space failed to update")

    video.release()
    cv2.destroyAllWindows()

def check_parking(file_path):
    parking = get_json(file_path)
    if not parking:
        return
    
    polylines = [np.array(polyline, np.int32) for polyline in parking['polylines']]
    space_names = parking['space_names']
    source = parking['source']

    with open(os.path.join(os.getcwd(), "data", "coco", "coco.txt"), "r") as my_file:
        clast_list = my_file.read().split("\n")

    model = YOLO(os.path.join(os.getcwd(), "data", "model", "yolov8s.pt"))

    video = cv2.VideoCapture(source if not source.isdigit() else int(source))
    
    if not video.isOpened():
        modal_alert("failed", "Failed to start video")
        return

    app.withdraw()
    
    manual_book()

    while True:
        ret, frame = video.read()
        if not ret:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (720, 480))
        result = model.predict(frame)
        bounding_boxes = result[0].boxes.data.cpu().numpy()
        df = pd.DataFrame(bounding_boxes).astype("float")
        
        list2 = []
        car = []
        filled_space = []
        empty_space = []
        parking_area = []

        for i, row in df.iterrows():
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            d = int(row[5])
            c = clast_list[d]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if 'car' in c:
                list2.append([cx, cy])

        for i, polyline in enumerate(polylines):
            parking_area.append(i)
            cv2.polylines(frame, [polyline], True, (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'{space_names[i]}', tuple(polyline[0]), 1, 1)

            for i1 in list2:
                cx1 = i1[0]
                cy1 = i1[1]
                result = cv2.pointPolygonTest(polyline, ((cx1, cy1)), False)
                if result >= 0:
                    cv2.circle(frame, (cx1, cy1), 5, (255, 0, 0), 1)
                    cv2.polylines(frame, [polyline], True, (0, 0, 255), 2)
                    car.append(cx1)
                    filled_space.append(space_names[i])
                    empty_space = [x for x in space_names if x not in filled_space]

        empty_total = len(parking_area) - len(car)
        cvzone.putTextRect(frame, f'Parking : {empty_total}/{len(parking_area)}', (20, 50), 1, 1)
        cvzone.putTextRect(frame, f'Filled space : {filled_space}', (20, 100), 1, 1)
        cvzone.putTextRect(frame, f'Empty space : {empty_space}', (20, 150), 1, 1)

        cv2.imshow('FRAME', frame)
        key = cv2.waitKey(100) & 0xFF
        
        if key == 113 or key == 81:
            break
        elif key == 105 or key == 73:
            manual_book()

    video.release()
    cv2.destroyAllWindows()
    app.deiconify()

def start_check_parking(select_row):
    if not select_row:
        return
    
    if len(select_row) > 1:
        modal_alert("warning", "Can only select one data")
        return
    
    file_path = select_row[0]["file_path"]

    check_parking(file_path)

'''
gate detection management
'''
def setup_gate(gate):
    source = gate["source"]
    video = cv2.VideoCapture(source if not source.isdigit() else int(source))
    
    if not video.isOpened():
        modal_alert("failed", "Failed to start video")
        return

    app.withdraw()

    gate_name = gate["gate_name"]
    file_name = gate["file_name"]
    file_path = gate["file_path"]
    folder_path = gate["folder_path"]
    polylines = []
    area_json = get_json(file_path)
    if area_json:
        if "polylines" in area_json:
            polylines = [np.array(polyline, np.int32) for polyline in area_json['polylines']]
    coordinates = []
    drawing = False
    modal_active = False

    def draw_area(event, x, y, flags, param):
        nonlocal polylines, coordinates, drawing, modal_active
        
        if modal_active:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            coordinates = [(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                coordinates.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if coordinates:
                if len(polylines):
                    return
                polylines.append(np.array(coordinates, np.int32))
                coordinates = []
        elif event == cv2.EVENT_RBUTTONDOWN:
            for i, polyline in enumerate(polylines):
                polyline_target = cv2.pointPolygonTest(polyline, (x, y), False)
                if polyline_target >= 0:
                    del polylines[i]
                    break

    while True:
        ret, frame = video.read()
        if not ret:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame = cv2.resize(frame, (720, 480))

        for i, polyline in enumerate(polylines):
            cv2.polylines(frame, [polyline], True, (0, 0, 255), 2)

        cv2.imshow('FRAME', frame)
        cv2.setMouseCallback('FRAME', draw_area)

        key = cv2.waitKey(100) & 0xFF

        if key == 115 or key == 83:
            if not len(polylines):
                modal_active = True
                modal_alert("warning", "No Polyline")
                modal_active = False
            else:
                data = {
                        "gate_name":gate_name,
                        "file_name":file_name,
                        'file_path': file_path,
                        "folder_path":folder_path,
                        'source': source,
                        'polylines': [polyline.tolist() for polyline in polylines],
                    }
                folder_check(folder_path)
                save_json(data)
                modal_alert("success", f"{gate_name} success save")
                break
        elif key == 113 or key == 81:
            modal_active = True
            decision = modal_okcancel(f"Cancel create {gate_name}")
            modal_active = False
            if decision:
                break

    video.release()
    cv2.destroyAllWindows()
    entry_gate()
    app.deiconify()

def detection_plat(file_path, stop_event):
    data = get_json(file_path)
    if not data:
        return
    
    pytesseract.pytesseract.tesseract_cmd = os.path.join(os.getcwd(), "data", "tesseract", "tesseract.exe")
    
    polylines = [np.array(polyline, np.int32) for polyline in data['polylines']]
    source = data['source']
    file_name = data["file_name"]
    gate_name = data["gate_name"]
    url = "/parkings/in" if file_name == "gate_in.json" else "/parkings/out"
    print(url)
    url_gate = "/gates/in"
    

    with open(os.path.join(os.getcwd(), "data", "coco", "coco1.txt"), "r") as my_file:
        class_list = my_file.read().split("\n")
        
    # Load the YOLO model
    model = YOLO(os.path.join(os.getcwd(), 'data', 'model', 'best.pt'))

    video = cv2.VideoCapture(source if not source.isdigit() else int(source))
    
    if not video.isOpened():
        print("Failed to start video")
        return

    while not stop_event.is_set():    
        #time.sleep(1)
        
        ret, frame = video.read()
        if not ret:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (720, 480))
        hasil = model.predict(frame)
        a = hasil[0].boxes.data.cpu().numpy()
        px = pd.DataFrame(a).astype("float")
        
        payload = {}
        gate_status = False
        detection_in_area = "default"

        for index, row in px.iterrows():
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            d = int(row[5])
            c = class_list[d]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            result = cv2.pointPolygonTest(np.array(polylines[0], np.int32), (cx, cy), False)
            if result >= 0:
                detection_in_area = "detected"
                
                # Crop and preprocess the detected region for OCR
                crop = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray, 10, 20, 20)

                # Extract and clean text from the cropped image
                text = pytesseract.image_to_string(gray).strip()
                match = re.search(r'[A-Z]{1,2}\s?\d{1,4}\s?[A-Z0-9]{1,3}', text.upper())
                if match:
                    text = match.group(0).replace(' ', '')
                    text = text[:9]  # Limit to a maximum of 9 characters
                    # Print recognized text in the terminal
                    detection_in_area = "validated"
                    payload = {"code": text}
                    print(f"Tervalidasi : {gate_name} : {text}")
                print(f"Terdeteksi : {gate_name}")
                
        if detection_in_area == "validated":
            request_api(url, payload, f"{gate_name} | success to send | {text}", f"{gate_name} | failed send to | {text}")

        if detection_in_area == "default":
            gate_status = False
        elif detection_in_area == "detected":
            gate_status = False
        elif detection_in_area == "validated":
            gate_status = True
        print(detection_in_area)
            
        if gate_name == "Gate in":
            request_api(url_gate, {"gateStatus":gate_status}, f"Gate open", f"Gate close")
            
        
        

    video.release()
    cv2.destroyAllWindows()
    
def check_detection_plat(file_path):
    data = get_json(file_path)
    if not data:
        return
    
    pytesseract.pytesseract.tesseract_cmd = os.path.join(os.getcwd(), "data", "tesseract", "tesseract.exe")
    
    polylines = [np.array(polyline, np.int32) for polyline in data['polylines']]
    source = data['source']
    file_name = data["file_name"]
    gate_name = data["gate_name"]
    

    with open(os.path.join(os.getcwd(), "data", "coco", "coco1.txt"), "r") as my_file:
        class_list = my_file.read().split("\n")
        
    # Load the YOLO model
    model = YOLO(os.path.join(os.getcwd(), 'data', 'model', 'best.pt'))

    video = cv2.VideoCapture(source if not source.isdigit() else int(source))
    
    if not video.isOpened():
        print("Failed to start video")
        return

    while True:    
        ret, frame = video.read()
        if not ret:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (720, 480))
        hasil = model.predict(frame)
        a = hasil[0].boxes.data.cpu().numpy()
        px = pd.DataFrame(a).astype("float")
        
        detection_in_area = "default"

        for index, row in px.iterrows():
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            d = int(row[5])
            c = class_list[d]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            result = cv2.pointPolygonTest(np.array(polylines[0], np.int32), (cx, cy), False)
            if result >= 0:
                detection_in_area = "detected"
                
                # Crop and preprocess the detected region for OCR
                crop = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray, 10, 20, 20)

                # Extract and clean text from the cropped image
                text = pytesseract.image_to_string(gray).strip()
                match = re.search(r'[A-Z]{1,2}\s?\d{1,4}\s?[A-Z0-9]{1,3}', text.upper())
                if match:
                    text = match.group(0).replace(' ', '')
                    text = text[:9]  # Limit to a maximum of 9 characters
                    # Print recognized text in the terminal
                    detection_in_area = "validated"
                    print(f"Tervalidasi : {gate_name} : {text}")
                print(f"Terdeteksi : {gate_name}")
        
        if detection_in_area == "default":
            area_color = (255, 255, 255)
        elif detection_in_area == "detected":
            area_color = (255, 0, 0)
        elif detection_in_area == "validated":
            area_color = (0, 255, 0)
        
        cv2.polylines(frame, [polylines[0]], True, area_color, 2)
        cv2.imshow("RGB", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


'''
start system
'''
def start_system(select_rows):
    parkings = []
    gates = get_all_json(os.path.join(os.getcwd(), "data", "json", "anpr"))
    space_names = []
    payload = []    
    
    if not select_rows:
        return
    
    if len(gates) <= 1:
        modal_alert("warning", "Please complete gate setup")
        entry_gate()
        return
    
    for gate in gates:
        gate_name = gate["gate_name"]
        if not "polylines" in gate or not "source" in gate:
            modal_alert("warning", f"Please complete {gate_name} setup")
            entry_gate()
            return
    
    for select_row in select_rows:
        parkings.append(get_json(select_row["file_path"]))
        space_names.extend(get_json(select_row["file_path"])["space_names"])
    
    for space_name in space_names:
        payload.append({
            "space_name":space_name,
            "status": True
        })
        
    for parking in parkings:
        if "polylines" in parking:
            del parking["polylines"]
    request_api("/parking_areas", parkings, "Parkings success to update", "Parking failed to update")
    
    # Membuat stop event
    stop_event = threading.Event()

    # Membuat dan memulai thread untuk setiap set data
    threads = []
    for parking in parkings:
        thread = threading.Thread(target=detection_parking, args=(parking['file_path'], stop_event))
        threads.append(thread)
        thread.start()
        
    for gate in gates:
        thread = threading.Thread(target=detection_plat, args=(gate["file_path"], stop_event))
        threads.append(thread)
        thread.start()

    def close(payload):
        stop_event.set()
        for thread in threads:
            thread.join()
        request_api("/parking_areas/delete", None, "Parking detection success to sh", "Parking detection failed to sh")
        request_api("/gates/in", {"gateStatus":False}, f"Gate In success to sh", f"Gate In failed to sh")
        modal_alert("success", "System is stoped")
        control_window.destroy()
    
    control_window = ctk.CTkToplevel(app)
    control_window.title("")
    control_window.geometry(geometry_center(300, 180))
    control_window.resizable(False, False)
    
    frame_header = ctk.CTkFrame(control_window, bg_color="#18181b", fg_color="#18181b")
    frame_header.pack(fill="x")
    
    frame_msg = ctk.CTkFrame(control_window, bg_color="#27272a", fg_color="#27272a")
    frame_msg.pack(expand="true", fill="both")
    
    frame_btn = ctk.CTkFrame(control_window, bg_color="#18181b", fg_color="#18181b")
    frame_btn.pack(fill="x")

    # Memuat dan mengubah ukuran gambar menggunakan PIL
    image_path = os.path.join(os.getcwd(), "data", "icons", "success.png")
    image = Image.open(image_path)
    resized_image = image.resize((20, 20))
    
    # Menggunakan CTkImage untuk memastikan kompatibilitas HighDPI
    image_ctk = ctk.CTkImage(light_image=resized_image, size=(20, 20))

    image_header = ctk.CTkLabel(frame_header, image=image_ctk, text="")
    image_header.pack(side="left", fill="x", padx=(20,0), pady=(20))
    
    header = ctk.CTkLabel(frame_header, text="Smart Parking System", font=("Helvetica", 15, "bold"))
    header.pack(side="left", fill="x", padx=10, pady=20) 
    
    msg = ctk.CTkLabel(frame_msg, text="System is running...", font=("Helvetica", 13), wraplength=300)
    msg.pack(expand="true", fill="both", padx=20)
    
    control_window.protocol("WM_DELETE_WINDOW", lambda: close(payload))
    
    btn = ctk.CTkButton(frame_btn, text="Stop", width=60, fg_color="#f87171", text_color="#000", hover_color="#dc2626",command=lambda: close(payload))
    btn.pack(side="right", padx=(10), pady=10)
    
    control_window.transient(app)
    control_window.grab_set()
    control_window.focus()
    app.wait_window(control_window)


'''
ctk management
'''
def geometry_center(app_width, app_height):    
    screen_width = app.winfo_screenwidth()
    screen_height = app.winfo_screenheight()
    
    x = int((screen_width - app_width)/2)
    y = int((screen_height - app_height)/2)
    
    center = f"{app_width}x{app_height}+{x}+{y}"
    
    return center

def clear_widget(frame):
    for widget in frame.winfo_children():
        widget.destroy()

def tabel():
    global tree
    
    clear_widget(frame_content)
    clear_widget(frame_action)
    
    buttons = ["Create", "Update", "Delete", "Check", "Setup Gate", "Start"]
    
    btn = ctk.CTkButton(frame_action, text=buttons[0], width=0, fg_color="#27272a",hover_color="#16a34a", border_width=1, border_color="#4ade80", command=lambda: button_handle_click(buttons[0]))
    btn.pack(side="left", expand="true", fill="x", padx=(10,0), pady=(10))
    
    for i in range(1, 4):
        btn = ctk.CTkButton(frame_action, text=buttons[i], width=0, fg_color="#27272a",hover_color="#16a34a", border_width=1, border_color="#4ade80", command=lambda b=buttons[i]: button_handle_click(b))
        btn.pack(side="left", expand="true", fill="x", padx=(10, 0), pady=(10))
        
    btn = ctk.CTkButton(frame_action, text=buttons[4], width=0, fg_color="#60a5fa", text_color="#000",hover_color="#2563eb", command=lambda: button_handle_click(buttons[4]))
    btn.pack(side="left", expand="true", fill="x", padx=(10, 0), pady=(10))
    
    btn = ctk.CTkButton(frame_action, text=buttons[5], width=0, fg_color="#4ade80", text_color="#000",hover_color="#16a34a", command=lambda: button_handle_click(buttons[5]))
    btn.pack(side="left", expand="true", fill="x", padx=(10), pady=(10))
    
    style.configure("Treeview.Heading", font=("Helvetica", 10, "bold"))
    
    tree = ttk.Treeview(frame_content, columns=("#", "Parking Name", "Source", "Total Area"), show="headings")
    tree.heading("#", text="#")
    tree.heading("Parking Name", text="Parking Name")
    tree.heading("Source", text="Source")
    tree.heading("Total Area", text="Total Area")
    
    tree.column("#", width=30, stretch=False, anchor="center")
    tree.column("Parking Name", width=150, stretch=True)
    tree.column("Source", width=150, stretch=True)
    tree.column("Total Area", width=50, stretch=True, anchor="center")
    
    style.configure("Treeview", rowheight=25, font=("Helvetica", 10), foreground="#1f2937")
    style.map("Treeview", background=[("selected", "#4ade80")], foreground=[("selected", "#000")])
    tree.tag_configure("evenrow", background="#d1d5db")
    tree.tag_configure("oddrow", background="#e5e7eb")

    
    file_path = os.path.join(os.getcwd(), "data", "json", "parking_detections")
    parkings = get_all_json(file_path)
    
    for i, parking in enumerate(parkings):
        index = i+1
        values = (index, parking["parking_name"], parking["source"], len(parking["space_names"]), parking["file_name"], parking["file_path"])
        if i % 2 == 0:
            tree.insert("", "end", values=values, tags="evenrow")
        else:
            tree.insert("", "end", values=values, tags="oddrow")
        
    tree.pack(side="top", expand="true", fill="both")

def create_data():
    clear_widget(frame_content)
    clear_widget(frame_action)
    
    parking_name_label = ctk.CTkLabel(frame_content, text="Parking name", font=("Helvetica", 13))
    parking_name_label.grid(row=0, column=0, padx=(40,10), pady=10, sticky="e")
    
    parking_name_entry = ctk.CTkEntry(frame_content)
    parking_name_entry.grid(row=0, column=1, padx=(0,40), pady=10, sticky="we")
    parking_name_entry.focus_set()
    
    source_label = ctk.CTkLabel(frame_content, text="Source", font=("Helvetica", 13))
    source_label.grid(row=1, column=0, padx=(40,10), pady=10, sticky="e")
    
    source_entry = ctk.CTkEntry(frame_content)
    source_entry.grid(row=1, column=1, padx=(0,40), pady=10, sticky="we")
    
    def next():
        parking_name = " ".join(parking_name_entry.get().split()).upper()
        source = source_entry.get()
        
        if not parking_name or not source:
            modal_alert("warning", "Parking name and Source is required")
        else:
            if re.findall(r'[^A-Za-z0-9\s]', parking_name):
                modal_alert("warning", "Parking name cannot use unique charecters")
            else:
                if parking_name in get_all_parking_names():
                    modal_alert("warning", f"{parking_name} already exists")
                else:
                    setup_parking_detection(parking_name, source)
    
    btn = ctk.CTkButton(frame_action, text="Next", width=100, fg_color="#4ade80", text_color="#000", hover_color="#16a34a", command=next)
    btn.pack(side="right", padx=(0, 10), pady=(10))
    
    btn = ctk.CTkButton(frame_action, text="Cancel", width=100, fg_color="#27272a", border_color="#f87171", hover_color="#dc2626", border_width=1, command=lambda: tabel())
    btn.pack(side="right", padx=(0, 10), pady=(10))
    
def update_data(parking):
    if not parking:
        return
    
    if len(parking) > 1:
        modal_alert("warning", "Can only select one data")
        return
    
    parking = parking[0]
    
    clear_widget(frame_content)
    clear_widget(frame_action)
    
    parking_name_label = ctk.CTkLabel(frame_content, text="Parking name", font=("Helvetica", 12))
    parking_name_label.grid(row=0, column=0, padx=(40,10), pady=10, sticky="e")
    
    parking_name_entry = ctk.CTkEntry(frame_content)
    parking_name_entry.grid(row=0, column=1, padx=(0,40), pady=10, sticky="we")
    parking_name_entry.insert(0, parking["parking_name"])
    parking_name_entry.focus_set()
    
    source_label = ctk.CTkLabel(frame_content, text="Source", font=("Helvetica", 12))
    source_label.grid(row=1, column=0, padx=(40,10), pady=10, sticky="e")
    
    source_entry = ctk.CTkEntry(frame_content)
    source_entry.grid(row=1, column=1, padx=(0,40), pady=10, sticky="we")
    source_entry.insert(0, parking["source"])
    
    def next():
        parking_name = " ".join(parking_name_entry.get().split()).upper()
        source = source_entry.get()
        all_parking_names = get_all_parking_names()
        
        while parking["parking_name"] in all_parking_names:
            all_parking_names.remove(parking["parking_name"])
        
        if not parking_name or not source:
            modal_alert("warning", "Name area and Source is required")
        else:
            if re.findall(r'[^A-Za-z0-9\s]', parking_name):
                modal_alert("warning", "Name cannot use unique characters")
            else:
                if parking_name in all_parking_names:
                    modal_alert("warning", f"{parking_name} already exists")
                else:
                    setup_parking_detection_update(parking_name, source, parking["file_path"])
    
    btn = ctk.CTkButton(frame_action, text="Next", width=100, fg_color="#4ade80", text_color="#000", hover_color="#16a34a", command=next)
    btn.pack(side="right", padx=(0, 10), pady=(10))
    
    btn = ctk.CTkButton(frame_action, text="Cancel", width=100, fg_color="#27272a", border_color="#f87171", hover_color="#dc2626", border_width=1, command=lambda: tabel())
    btn.pack(side="right", padx=(0, 10), pady=(10))

def delete_data():
    parkings = select_rows()
    
    if not parkings:
        return

    if modal_okcancel("Delete selected items?") == "yes":
        for parking in parkings:
            file_path = parking["file_path"]
            if file_check(file_path):
                os.remove(file_path)
        modal_alert("success", "Data success to delete")
        tabel()

def entry_gate():
    datas = get_all_json(os.path.join(os.getcwd(), "data", "json", "anpr"))
    gate_in = {
        "gate_name":"Gate in",
        "file_name":"gate_in.json",
        "file_path":os.path.join(os.getcwd(), "data", "json", "anpr", "gate_in.json"),
        "folder_path":os.path.join(os.getcwd(), "data", "json", "anpr"),
    }
    gate_out = {
        "gate_name":"Gate out",
        "file_name":"gate_out.json",
        "file_path":os.path.join(os.getcwd(), "data", "json", "anpr", "gate_out.json"),
        "folder_path":os.path.join(os.getcwd(), "data", "json", "anpr"),
    }
    
    if len(datas):
        for data in datas:
            if "file_name" in data:
                gate_in.update(data) if data["file_name"] == "gate_in.json" else gate_out.update(data)
    
    clear_widget(frame_content)
    clear_widget(frame_action)
    
    gate_in_label = ctk.CTkLabel(frame_content, text="Gate In", font=("Helvetica", 12))
    gate_in_label.grid(row=0, column=0, padx=(20,10), pady=10, sticky="e")
    
    gate_in_entry = ctk.CTkEntry(frame_content)
    gate_in_entry.grid(row=0, column=1, padx=(0,10), pady=10, sticky="we")
    gate_in_entry.insert(0, gate_in["source"] if "source" in gate_in else "")
    gate_in_entry.focus_set()
    
    gate_in_btn = ctk.CTkButton(frame_content, text="Setup", width=100, fg_color="#4ade80", text_color="#000", hover_color="#16a34a", command=lambda: next(gate_in, gate_in_entry.get()))
    gate_in_btn.grid(row=0, column=2, padx=(0,10), pady=10, sticky="we")
    
    if file_check(gate_in["file_path"]):
        gate_in_check_btn = ctk.CTkButton(frame_content, text="Check", width=100, fg_color="#4ade80", text_color="#000", hover_color="#16a34a", command=lambda: check_detection_plat(gate_in["file_path"]))
        gate_in_check_btn.grid(row=0, column=3, padx=(0,20), pady=10, sticky="we")
    
    gate_out_label = ctk.CTkLabel(frame_content, text="Gate Out", font=("Helvetica", 12))
    gate_out_label.grid(row=1, column=0, padx=(20,10), pady=10, sticky="e")
    
    gate_out_entry = ctk.CTkEntry(frame_content)
    gate_out_entry.grid(row=1, column=1, padx=(0,10), pady=10, sticky="we")
    gate_out_entry.insert(0, gate_out["source"] if "source" in gate_out else "")
    
    gate_out_btn = ctk.CTkButton(frame_content, text="Setup", width=100, fg_color="#4ade80", text_color="#000", hover_color="#16a34a", command=lambda: next(gate_out, gate_out_entry.get()))
    gate_out_btn.grid(row=1, column=2, padx=(0,10), pady=10, sticky="we")
    
    if file_check(gate_out["file_path"]):
        gate_out_check_btn = ctk.CTkButton(frame_content, text="Check", width=100, fg_color="#4ade80", text_color="#000", hover_color="#16a34a", command=lambda: check_detection_plat(gate_out["file_path"]))
        gate_out_check_btn.grid(row=1, column=3, padx=(0,20), pady=10, sticky="we")

    
    def next(data, source):
        gate = {}
        gate.update(data)
        gate.update({"source":source})
    
        if not source:
            modal_alert("warning", "Source is required")
        else:
            setup_gate(gate)
    
    btn = ctk.CTkButton(frame_action, text="Back", width=100, fg_color="#27272a", border_color="#f87171", hover_color="#dc2626", border_width=1, command=lambda: tabel())
    btn.pack(side="right", padx=(0, 10), pady=(10))

def button_handle_click(button):
    if button == "Create":
        create_data()
    elif button == "Update":
        update_data(select_rows())
    elif button == "Delete":
        delete_data()
    elif button == "Check":
        start_check_parking(select_rows())
    elif button == "Setup Gate":
        entry_gate()
    elif button == "Start":
        start_system(select_rows())
        # start_detection_parking(select_rows())
    else:
        print(button)
        tabel()

'''
main frame
'''
app = ctk.CTk()
app.title("Parking Area Detection")

ctk.set_appearance_mode("dark")

style = ttk.Style()

frame_header = ctk.CTkFrame(app, bg_color="#09090b", fg_color="#09090b")
frame_header.pack(fill="x")
frame_header.grid_columnconfigure(0, weight=1)

header_title = ctk.CTkLabel(frame_header, text="Smart Parking System", font=("Helvetica", 20, "bold"))
header_title.pack(side="top", ipady=20, fill="x")

frame_body = ctk.CTkFrame(app, bg_color="#27272a", fg_color="#27272a")
frame_body.pack(expand="true", fill="both")

frame_action = ctk.CTkFrame(frame_body, bg_color="#27272a", fg_color="#27272a")
frame_action.pack(fill="x")

frame_content = ctk.CTkFrame(frame_body, bg_color="#27272a", fg_color="#27272a")
frame_content.pack(expand="true", fill="both", padx=10, pady=10)
frame_content.grid_columnconfigure(1, weight=1)

tabel()

app.geometry(geometry_center(480,300))
app.mainloop()