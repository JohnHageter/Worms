import os
import time
import threading
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from Module.io.IDSCamera import Camera


class TimelapseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Timelapse Capture")

        self.cam = None
        self.running = False
        self.worker_thread = None

        self.save_dir = tk.StringVar(value="timelapse_images")
        self.interval_sec = tk.IntVar(value=120)
        self.duration_days = tk.DoubleVar(value=14)

        self.gain = tk.DoubleVar(value=3)
        self.exposure = tk.DoubleVar(value=10)
        self.framerate = tk.DoubleVar(value=30)

        self.build_ui()

    def build_ui(self):
        pad = {"padx": 5, "pady": 5}

        # Camera controls
        cam_frame = tk.LabelFrame(self.root, text="Camera")
        cam_frame.pack(fill="x", **pad)

        tk.Button(cam_frame, text="Open Camera", command=self.open_camera).grid(row=0, column=0, **pad)
        tk.Button(cam_frame, text="Close Camera", command=self.close_camera).grid(row=0, column=1, **pad)

        tk.Label(cam_frame, text="Gain").grid(row=1, column=0, sticky="e")
        tk.Entry(cam_frame, textvariable=self.gain, width=8).grid(row=1, column=1)

        tk.Label(cam_frame, text="Exposure").grid(row=2, column=0, sticky="e")
        tk.Entry(cam_frame, textvariable=self.exposure, width=8).grid(row=2, column=1)

        tk.Label(cam_frame, text="Frame rate").grid(row=3, column=0, sticky="e")
        tk.Entry(cam_frame, textvariable=self.framerate, width=8).grid(row=3, column=1)

        tk.Button(cam_frame, text="Apply Settings", command=self.apply_camera_settings).grid(
            row=4, column=0, columnspan=2, **pad
        )

        # Timelapse controls
        tl_frame = tk.LabelFrame(self.root, text="Timelapse")
        tl_frame.pack(fill="x", **pad)

        tk.Label(tl_frame, text="Interval (seconds)").grid(row=0, column=0, sticky="e")
        tk.Entry(tl_frame, textvariable=self.interval_sec, width=10).grid(row=0, column=1)

        tk.Label(tl_frame, text="Duration (days)").grid(row=1, column=0, sticky="e")
        tk.Entry(tl_frame, textvariable=self.duration_days, width=10).grid(row=1, column=1)

        # Save directory
        save_frame = tk.Frame(self.root)
        save_frame.pack(fill="x", **pad)

        tk.Entry(save_frame, textvariable=self.save_dir).pack(side="left", fill="x", expand=True, padx=5)
        tk.Button(save_frame, text="Browse", command=self.choose_directory).pack(side="right")

        # Run controls
        run_frame = tk.Frame(self.root)
        run_frame.pack(fill="x", **pad)

        tk.Button(run_frame, text="Start Timelapse", command=self.start_timelapse).pack(side="left", padx=5)
        tk.Button(run_frame, text="Stop", command=self.stop_timelapse).pack(side="right", padx=5)

    def open_camera(self):
        if self.cam is not None:
            return

        try:
            self.cam = Camera(index=0)
            self.cam.set_watch_window(0, 1280, 0, 1024)
            self.apply_camera_settings()
            messagebox.showinfo("Camera", "Camera opened successfully.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.cam = None

    def close_camera(self):
        if self.cam:
            self.cam.close()
            self.cam = None
            messagebox.showinfo("Camera", "Camera closed.")

    def apply_camera_settings(self):
        if not self.cam:
            return

        try:
            self.cam.set_attr("gain", self.gain.get())
            self.cam.set_attr("exposure", self.exposure.get())
            self.cam.set_attr("framerate", self.framerate.get())
        except Exception as e:
            messagebox.showerror("Camera Error", str(e))


    def start_timelapse(self):
        if self.running:
            return

        if not self.cam:
            messagebox.showwarning("Warning", "Open the camera first.")
            return

        os.makedirs(self.save_dir.get(), exist_ok=True)

        self.running = True
        self.worker_thread = threading.Thread(target=self.timelapse_loop, daemon=True)
        self.worker_thread.start()

    def stop_timelapse(self):
        self.running = False

    def timelapse_loop(self):
        start_time = datetime.now()
        end_time = start_time + timedelta(days=self.duration_days.get())

        print(f"Starting run at: {start_time}")
        print(f"Run will end at: {end_time}")

        while self.running and datetime.now() < end_time:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.save_dir.get(), f"image_{timestamp}.png")

            img, ok = self.cam.grab_next_frame(to_bgr=False)
            if ok and img is not None:
                img = np.ascontiguousarray(img.copy(), dtype=np.uint8)
                cv2.imwrite(filename, img)
                print(f"[{datetime.now()}] Saved image: {filename}")
            else:
                print(f"[{datetime.now()}] Failed to grab image.")

            for _ in range(self.interval_sec.get()):
                if not self.running:
                    break
                time.sleep(1)

        self.running = False
        print("Timelapse finished.")


    def choose_directory(self):
        path = filedialog.askdirectory()
        if path:
            self.save_dir.set(path)

    def ui_error(self, title, message):
        self.root.after(0, lambda: messagebox.showerror(title, message))




