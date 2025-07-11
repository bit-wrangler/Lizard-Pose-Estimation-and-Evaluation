import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
from pathlib import Path
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
import sys
import yaml
import numpy as np
import traceback
import torchvision.transforms.v2 as T
import pandas as pd
import bisect

# --- Dependency Check ---
try:
    # Check for libraries required for the app and for creating dummy files
    import h5py
except ImportError as e:
    print(f"Error: A required library is not installed. Please install it.")
    print(f"Missing library: {e.name}")
    print("You can likely install it using: pip install h5py pandas PyYAML")
    sys.exit(1)

# --- Configuration ---
VIDEO_DIRECTORY = "/home/alek/ml_data/lizard-movement"
RESULTS_DIRECTORY = "/home/alek/projects/cdl-test1/data/cdl-projects/test1-haag-2025-05-21/videos"
ANNOTATIONS_DIRECTORY = "annotations" # Directory to save output YAML files
SUBJECT_MODEL_CHECKPOINT_PATH = "fine_tuned_detector.pth"

# --- Model & Display Parameters ---
SUBJECT_CONFIDENCE_THRESHOLD = 0.90 # Confidence for finding the subject to crop
CROP_PADDING = 50                   # Padding around the detected subject bbox
MIN_LANDMARK_CONFIDENCE = 0.5       # Min confidence to display a DLC landmark
LANDMARK_RADIUS = 4                 # Pixel radius for drawing landmark circles

class VideoEventAnnotatorApp:
    def __init__(self, root, video_dir_path, results_dir_path):
        self.root = root
        self.root.title("Video Event Annotator")

        # --- File and Data Paths ---
        self.video_dir = Path(video_dir_path)
        self.results_dir = Path(results_dir_path)
        self.annotations_dir = Path(ANNOTATIONS_DIRECTORY)
        self.annotations_dir.mkdir(exist_ok=True) # Ensure annotation dir exists

        if not self.video_dir.is_dir() or not self.results_dir.is_dir():
            messagebox.showerror("Error", f"Video or Results directory not found.\nVideo: {self.video_dir}\nResults: {self.results_dir}")
            self.root.destroy()
            return

        # --- Data Structures ---
        self.video_to_h5_map = {}
        self.current_annotations = {} # {frame_num: ["event1", "event2"], ...}
        self.current_h5_data = None   # To store landmark data for the current video

        # --- Defined Events ---
        self.events = {
            "left ankle placed": "1",
            "right ankle placed": "2",
            "left wrist placed": "3",
            "right wrist placed": "4",
        }
        self.key_to_event_map = {v: k for k, v in self.events.items()}

        # --- State Variables ---
        self.current_video_filepath = None
        self.current_video_capture = None
        self.current_frame_number = -1
        self.total_frames_in_video = 0
        self.photo_image = None
        self.current_cropped_cv2_frame = None
        self.crop_x_offset_orig = 0
        self.crop_y_offset_orig = 0
        self.drawn_landmark_items = []
        self.show_landmarks = False
        self.show_confidence = False

        # --- UI Elements ---
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_pane = tk.Frame(main_frame, width=250)
        left_pane.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_pane.pack_propagate(False)

        list_label = tk.Label(left_pane, text="Videos (with results)")
        list_label.pack(fill=tk.X)
        self.video_listbox = tk.Listbox(left_pane, exportselection=False)
        self.video_listbox.pack(fill=tk.BOTH, expand=True)

        right_pane = tk.Frame(main_frame)
        right_pane.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(right_pane, cursor="arrow", bg="lightgray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        events_frame = tk.Frame(right_pane, relief=tk.GROOVE, borderwidth=2)
        events_frame.pack(fill=tk.X, pady=5)
        
        info_text = "Events (1-4) | Overlay (q) | Confidence (w) | Navigate Events (Home, End, PgUp, PgDn)"
        events_label = tk.Label(events_frame, text=info_text)
        events_label.pack()

        self.event_checkbox_vars = {}
        for event_name, key in self.events.items():
            var = tk.BooleanVar()
            chk = tk.Checkbutton(events_frame, text=f"{event_name} ({key})", variable=var, command=lambda e=event_name: self.toggle_event(e))
            chk.pack(anchor='w', padx=10)
            self.event_checkbox_vars[event_name] = var

        self.status_label = tk.Label(root, text="Status: Initializing...", anchor='w', relief=tk.SUNKEN)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.subject_detection_model = None
        self.image_transform_for_model = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

        # --- Bindings ---
        self.video_listbox.bind('<<ListboxSelect>>', self.on_video_select)
        self.root.bind('<Up>', self.navigate_listbox)
        self.root.bind('<Down>', self.navigate_listbox)
        self.root.bind('<Left>', lambda e: self.navigate_frame(-1))
        self.root.bind('<Right>', lambda e: self.navigate_frame(1))
        for key in self.events.values():
            self.root.bind(key, self.on_key_press_event)
        self.root.bind('q', self.toggle_landmark_display)
        self.root.bind('w', self.toggle_confidence_display)
        # Event navigation bindings
        self.root.bind('<Next>', lambda e: self.navigate_to_event('next'))      # Page Down
        self.root.bind('<Prior>', lambda e: self.navigate_to_event('prev'))     # Page Up
        self.root.bind('<Home>', lambda e: self.navigate_to_event('first'))
        self.root.bind('<End>', lambda e: self.navigate_to_event('last'))
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- Initial Load ---
        self.load_subject_detection_model()
        self.load_and_map_files()
        self._populate_video_listbox()

        if not self.video_to_h5_map:
            messagebox.showwarning("No Data", "No matching video and .h5 result files were found.")
        else:
            self.video_listbox.selection_set(0)
            self.video_listbox.event_generate("<<ListboxSelect>>")

    def get_object_detection_model(self, num_classes):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def load_subject_detection_model(self):
        self.status_label.config(text="Loading subject detection model..."); self.root.update_idletasks()
        if not os.path.exists(SUBJECT_MODEL_CHECKPOINT_PATH):
            messagebox.showerror("Model Missing", f"Subject model not found: {SUBJECT_MODEL_CHECKPOINT_PATH}")
            self.subject_detection_model = None
            return
        try:
            self.subject_detection_model = self.get_object_detection_model(num_classes=2)
            self.subject_detection_model.load_state_dict(torch.load(SUBJECT_MODEL_CHECKPOINT_PATH, map_location=self.device))
            self.subject_detection_model.to(self.device).eval()
            print(f"Loaded subject model from {SUBJECT_MODEL_CHECKPOINT_PATH}")
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Failed to load subject detection model: {e}")
            self.subject_detection_model = None
            print(traceback.format_exc())

    def load_and_map_files(self):
        video_files = list(self.video_dir.rglob("*.MP4")) + list(self.video_dir.rglob("*.mp4"))
        h5_files = list(self.results_dir.rglob("*.h5"))
        video_map_by_name = {v.name: v for v in video_files}

        for h5_path in h5_files:
            base_name = h5_path.name.split('DLC')[0].rstrip('_')
            potential_video_name_upper = f"{base_name}.MP4"
            potential_video_name_lower = f"{base_name}.mp4"

            if potential_video_name_upper in video_map_by_name:
                video_path = video_map_by_name[potential_video_name_upper]
                self.video_to_h5_map[str(video_path)] = str(h5_path)
            elif potential_video_name_lower in video_map_by_name:
                video_path = video_map_by_name[potential_video_name_lower]
                self.video_to_h5_map[str(video_path)] = str(h5_path)
        print(f"Found {len(self.video_to_h5_map)} video-result pairs.")

    def _populate_video_listbox(self):
        self.video_listbox.delete(0, tk.END)
        sorted_videos = sorted(self.video_to_h5_map.keys())
        for video_path_str in sorted_videos:
            self.video_listbox.insert(tk.END, Path(video_path_str).name)

    def navigate_listbox(self, event):
        if not self.video_listbox.size(): return
        current_selection = self.video_listbox.curselection()
        current_idx = current_selection[0] if current_selection else -1
        
        if event.keysym == 'Down': next_idx = min(current_idx + 1, self.video_listbox.size() - 1)
        elif event.keysym == 'Up': next_idx = max(current_idx - 1, 0)
        else: return

        if next_idx != current_idx:
            self.video_listbox.selection_clear(0, tk.END)
            self.video_listbox.selection_set(next_idx)
            self.video_listbox.activate(next_idx)
            self.video_listbox.see(next_idx)
            self.on_video_select(event)
        return "break"

    def on_video_select(self, event):
        selection = self.video_listbox.curselection()
        if not selection: return
        video_name = self.video_listbox.get(selection[0])
        video_path = Path(sorted(self.video_to_h5_map.keys())[selection[0]])
        self.load_video(video_path)

    def load_video(self, video_path):
        if self.current_video_capture and self.current_video_capture.isOpened():
            self.current_video_capture.release()

        self.current_video_filepath = video_path
        self.load_annotations()

        h5_path_str = self.video_to_h5_map.get(str(video_path))
        if not h5_path_str:
            messagebox.showerror("Error", f"No result file found for {video_path.name}")
            return
        try:
            with pd.HDFStore(h5_path_str, 'r') as store:
                df = store['df_with_missing']
                self.current_h5_data = df.to_numpy().reshape(len(df), -1, 3)
                print(f"Loaded H5 data from {Path(h5_path_str).name} with shape {self.current_h5_data.shape}")
        except Exception as e:
            messagebox.showerror("H5 Load Error", f"Could not read data from {Path(h5_path_str).name}: {e}")
            self.current_h5_data = None
            return

        self.current_video_capture = cv2.VideoCapture(str(video_path))
        if not self.current_video_capture.isOpened():
            messagebox.showerror("Video Error", f"Could not open video file: {video_path.name}")
            return

        self.total_frames_in_video = int(self.current_video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_number = -1
        self.navigate_frame(1)

    def navigate_frame(self, delta):
        if not self.current_video_capture or not self.subject_detection_model: return

        start_frame = self.current_frame_number
        target_frame_num = start_frame + delta

        while 0 <= target_frame_num < self.total_frames_in_video:
            self.status_label.config(text=f"Searching frame {target_frame_num} for subject..."); self.root.update_idletasks()
            self.current_video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame_num)
            ret, frame = self.current_video_capture.read()
            if not ret: break

            subject_bbox = self.run_subject_detection_on_frame(frame)
            if subject_bbox is not None:
                self.current_frame_number = target_frame_num
                self.process_and_display_frame(frame, subject_bbox)
                return
            target_frame_num += delta

        self.status_label.config(text=f"End of video or no subject found from frame {start_frame}.")
        if start_frame == -1:
            messagebox.showinfo("Info", f"No subject detected in the entirety of {self.current_video_filepath.name}")
            self.canvas.delete("all")

    def go_to_frame(self, target_frame_num):
        if not self.current_video_capture or not self.current_video_capture.isOpened(): return
        if not (0 <= target_frame_num < self.total_frames_in_video): return

        self.current_video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame_num)
        ret, frame = self.current_video_capture.read()
        if not ret:
            self.status_label.config(text=f"Error reading frame {target_frame_num}")
            return
        
        self.current_frame_number = target_frame_num
        subject_bbox = self.run_subject_detection_on_frame(frame)

        if subject_bbox is not None:
            self.process_and_display_frame(frame, subject_bbox)
        else:
            # Subject not found, display the full, uncropped frame
            self.current_cropped_cv2_frame = frame
            self.crop_x_offset_orig = 0
            self.crop_y_offset_orig = 0

            frame_rgb = cv2.cvtColor(self.current_cropped_cv2_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
            if canvas_w < 2 or canvas_h < 2: canvas_w, canvas_h = 800, 600
            
            ratio = min(canvas_w / pil_img.width, canvas_h / pil_img.height)
            new_size = (int(pil_img.width * ratio), int(pil_img.height * ratio))
            self.photo_image = ImageTk.PhotoImage(pil_img.resize(new_size, Image.LANCZOS))

            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
            self.canvas.image = self.photo_image
            
            # Add warning text on canvas
            self.canvas.create_text(canvas_w / 2, 20, text="Subject Not Detected", 
                                    fill="orange", font=("TkDefaultFont", 14, "bold"))

            self.draw_landmarks()
            self.update_status()
            self.update_event_checkboxes()

    def run_subject_detection_on_frame(self, frame_cv2):
        if not self.subject_detection_model: return None
        img_rgb = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB)
        img_tensor = self.image_transform_for_model(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.subject_detection_model([img_tensor])

        best_score = 0; best_box = None
        for i in range(len(prediction[0]['scores'])):
            score = prediction[0]['scores'][i].item()
            if score > SUBJECT_CONFIDENCE_THRESHOLD and score > best_score:
                best_score, best_box = score, prediction[0]['boxes'][i].cpu().numpy().astype(int)
        return best_box

    def process_and_display_frame(self, frame, subject_bbox):
        h, w = frame.shape[:2]
        x1s, y1s, x2s, y2s = subject_bbox
        self.crop_x_offset_orig = max(0, x1s - CROP_PADDING)
        self.crop_y_offset_orig = max(0, y1s - CROP_PADDING)
        crop_x2_abs = min(w, x2s + CROP_PADDING)
        crop_y2_abs = min(h, y2s + CROP_PADDING)
        self.current_cropped_cv2_frame = frame[self.crop_y_offset_orig:crop_y2_abs, self.crop_x_offset_orig:crop_x2_abs]

        if self.current_cropped_cv2_frame.size == 0: return

        frame_rgb = cv2.cvtColor(self.current_cropped_cv2_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if canvas_w < 2 or canvas_h < 2: canvas_w, canvas_h = 800, 600
        ratio = min(canvas_w / pil_img.width, canvas_h / pil_img.height)
        new_size = (int(pil_img.width * ratio), int(pil_img.height * ratio))
        
        self.photo_image = ImageTk.PhotoImage(pil_img.resize(new_size, Image.LANCZOS))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
        self.canvas.image = self.photo_image

        self.draw_landmarks()
        self.update_status()
        self.update_event_checkboxes()
        
    def draw_landmarks(self):
        for item in self.drawn_landmark_items:
            self.canvas.delete(item)
        self.drawn_landmark_items = []

        if not self.show_landmarks or self.current_h5_data is None or self.current_frame_number >= len(self.current_h5_data):
            return
            
        landmarks_for_frame = self.current_h5_data[self.current_frame_number]
        crop_h, crop_w = self.current_cropped_cv2_frame.shape[:2]
        disp_w, disp_h = self.photo_image.width(), self.photo_image.height()
        
        scale_x = disp_w / crop_w
        scale_y = disp_h / crop_h

        for i, (x_orig, y_orig, conf) in enumerate(landmarks_for_frame):
            if conf >= MIN_LANDMARK_CONFIDENCE:
                x_crop = x_orig - self.crop_x_offset_orig
                y_crop = y_orig - self.crop_y_offset_orig
                
                if 0 <= x_crop < crop_w and 0 <= y_crop < crop_h:
                    x_disp, y_disp = x_crop * scale_x, y_crop * scale_y
                    
                    x1, y1 = (x_disp - LANDMARK_RADIUS), (y_disp - LANDMARK_RADIUS)
                    x2, y2 = (x_disp + LANDMARK_RADIUS), (y_disp + LANDMARK_RADIUS)
                    color = "red" if i < 4 else "cyan"
                    
                    oval = self.canvas.create_oval(x1, y1, x2, y2, fill=color, outline=color)
                    self.drawn_landmark_items.append(oval)

                    if self.show_confidence:
                        text = self.canvas.create_text(x_disp + 5, y_disp - 5, text=f"{conf:.2f}", anchor="w", fill="yellow")
                        self.drawn_landmark_items.append(text)

    def toggle_landmark_display(self, event=None):
        self.show_landmarks = not self.show_landmarks
        if not self.show_landmarks:
            self.show_confidence = False
        self.draw_landmarks()
        self.update_status()

    def toggle_confidence_display(self, event=None):
        if self.show_landmarks:
            self.show_confidence = not self.show_confidence
            self.draw_landmarks()
            self.update_status()

    def navigate_to_event(self, direction):
        if not self.current_annotations:
            self.status_label.config(text="Status: No events to navigate in this video.")
            return

        event_frames = sorted(self.current_annotations.keys())
        if not event_frames:
            self.status_label.config(text="Status: No events to navigate in this video.")
            return

        current_frame = self.current_frame_number
        target_frame = -1

        if direction == 'first':
            target_frame = event_frames[0]
        elif direction == 'last':
            target_frame = event_frames[-1]
        elif direction == 'next':
            idx = bisect.bisect_right(event_frames, current_frame)
            target_frame = event_frames[idx] if idx < len(event_frames) else event_frames[0]
        elif direction == 'prev':
            idx = bisect.bisect_left(event_frames, current_frame)
            target_frame = event_frames[idx - 1] if idx > 0 else event_frames[-1]
        
        if target_frame != -1 and target_frame != self.current_frame_number:
            self.go_to_frame(target_frame)

    def on_key_press_event(self, event):
        event_name = self.key_to_event_map.get(event.keysym)
        if event_name: self.toggle_event(event_name)
        return "break"

    def toggle_event(self, event_name):
        if self.current_video_filepath is None or self.current_frame_number < 0: return
        
        frame_events = self.current_annotations.get(self.current_frame_number, [])
        if event_name in frame_events:
            frame_events.remove(event_name)
            if not frame_events:
                del self.current_annotations[self.current_frame_number]
        else:
            if self.current_frame_number not in self.current_annotations:
                self.current_annotations[self.current_frame_number] = []
            self.current_annotations[self.current_frame_number].append(event_name)
        
        self.save_annotations()
        self.update_event_checkboxes()
        self.update_status()

    def update_event_checkboxes(self):
        if self.current_frame_number < 0: return
        frame_events = self.current_annotations.get(self.current_frame_number, [])
        for event_name, var in self.event_checkbox_vars.items():
            var.set(event_name in frame_events)

    def update_status(self):
        if self.current_video_filepath:
            video_name = self.current_video_filepath.name
            status_text = f"Video: {video_name} | Frame: {self.current_frame_number}/{self.total_frames_in_video-1}"
            
            frame_events = self.current_annotations.get(self.current_frame_number, [])
            if frame_events:
                status_text += f" | Events: {', '.join(sorted(frame_events))}"
            
            overlay_status = []
            if self.show_landmarks: overlay_status.append("Landmarks ON")
            if self.show_confidence: overlay_status.append("Confidence ON")
            if overlay_status: status_text += f" | [{', '.join(overlay_status)}]"

            self.status_label.config(text=status_text)
        else:
            self.status_label.config(text="No video loaded.")

    def get_annotation_filepath(self):
        if not self.current_video_filepath: return None
        return self.annotations_dir / f"{self.current_video_filepath.stem}.yaml"

    def load_annotations(self):
        self.current_annotations = {}
        filepath = self.get_annotation_filepath()
        if filepath and filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    data = yaml.safe_load(f) or {}
                    # Ensure keys are integers
                    self.current_annotations = {int(k): v for k, v in data.items()}
                print(f"Loaded annotations from {filepath}")
            except Exception as e:
                print(f"Warning: Could not load annotation file {filepath}: {e}")

    def save_annotations(self):
        filepath = self.get_annotation_filepath()
        if not filepath: return
        try:
            sorted_annotations = dict(sorted(self.current_annotations.items()))
            with open(filepath, 'w') as f:
                yaml.dump(sorted_annotations, f, sort_keys=True, default_flow_style=False)
            print(f"Saved annotations to {filepath}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save annotations to {filepath}:\n{e}")

    def on_closing(self):
        print("Closing application...")
        self.save_annotations()
        if self.current_video_capture and self.current_video_capture.isOpened():
            self.current_video_capture.release()
        self.root.destroy()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    for dir_path in [VIDEO_DIRECTORY, RESULTS_DIRECTORY, ANNOTATIONS_DIRECTORY]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    try:
        import subprocess
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ffmpeg_exists = True
    except (FileNotFoundError, subprocess.CalledProcessError):
        ffmpeg_exists = False
        print("Warning: `ffmpeg` not found. Cannot create dummy video files.")
    if not any(Path(VIDEO_DIRECTORY).glob('*.mp4')) and not any(Path(VIDEO_DIRECTORY).glob('*.MP4')) and ffmpeg_exists:
        print(f"Creating dummy MP4 file in {VIDEO_DIRECTORY} for testing...")
        dummy_file = Path(VIDEO_DIRECTORY) / "0125_1.mp4"
        command = ['ffmpeg', '-y', '-f', 'lavfi', '-i', 'testsrc=duration=5:size=640x480:rate=10', str(dummy_file)]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Dummy video created: {dummy_file.name}")
    if not any(Path(RESULTS_DIRECTORY).glob('*.h5')):
        print(f"Creating dummy .h5 file in {RESULTS_DIRECTORY} for testing...")
        dummy_h5_file = Path(RESULTS_DIRECTORY) / "0125_1DLC_resnet50_testMay25shuffle1_100.h5"
        try:
            with h5py.File(dummy_h5_file, 'w') as f:
                scorer = "DLC_resnet50_testMay25shuffle1_100"
                bodyparts = ['snout', 'head', 'l_wrist', 'r_wrist', 'l_ankle', 'r_ankle', 'tail_base']
                coords = ['x', 'y', 'likelihood']
                columns = pd.MultiIndex.from_product([[scorer], bodyparts, coords], names=['scorer', 'bodyparts', 'coords'])
                data = np.random.rand(50, len(columns)) * 640
                data[:, 2::3] = np.random.rand(50, len(bodyparts))
                df = pd.DataFrame(data, columns=columns)
                df.to_hdf(str(dummy_h5_file), 'df_with_missing', format='table', mode='w')
            print(f"Dummy H5 file created: {dummy_h5_file.name}")
        except Exception as e:
            if 'pandas' in str(e).lower():
                 print("Could not create dummy H5 file: `pandas` is required. Please 'pip install pandas'.")
            else:
                print(f"Could not create dummy H5 file: {e}")
    root = tk.Tk()
    root.minsize(1024, 768)
    root.geometry("1440x900")
    app = VideoEventAnnotatorApp(root, VIDEO_DIRECTORY, RESULTS_DIRECTORY)
    if 'winfo_exists' in dir(root) and root.winfo_exists():
        app.run()