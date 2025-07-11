import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import random
from pathlib import Path
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T # Updated transforms
import threading # To run training in a separate thread
import os
import yaml

# --- Configuration ---
VIDEO_DIRECTORY = "/home/alek/ml_data/lizard-movement"  # <<<< CHANGE THIS TO YOUR VIDEO DIRECTORY PATH
                            # Create this directory and put some .mp4 files in it for testing.
TEMP_TRAINING_DATA_DIR = "data/temp_training_data"
HANDLE_SIZE = 8  # Size of the resize handles
MIN_BBOX_SIZE = 10 # Minimum size (width or height) for a bbox to be valid

class PatienceEarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class AnnotationDataset(Dataset):
    def __init__(self, image_annotations, transform=None):
        """
        Args:
            image_annotations (list): List of tuples (image_file_path, bbox_coords_orig).
                                      bbox_coords_orig is (x1, y1, x2, y2).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # No need to filter for ann[2] here as we only pass valid ones
        self.image_annotations = image_annotations
        self.transform = transform

    def __len__(self):
        return len(self.image_annotations)

    def __getitem__(self, idx):
        img_path_str, bbox_coords_orig = self.image_annotations[idx]
        img_path = Path(img_path_str)
        img_pil = None

        try:
            if not img_path.is_file():
                raise IOError(f"Image file not found: {img_path}")
            img_pil = Image.open(img_path).convert("RGB") # Ensure RGB

        except Exception as e:
            print(f"Dataset Warning: Error loading image {img_path}: {e}")
            img_pil = Image.new('RGB', (224, 224), color='black') # Dummy PIL
            bbox_coords_orig = (0, 0, 1, 1) # Dummy bbox

        # Apply the transform
        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            # Fallback (should not happen with current setup)
            temp_transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
            img_tensor = temp_transform(img_pil)

        # --- Target Preparation ---
        boxes = torch.as_tensor([bbox_coords_orig], dtype=torch.float32)
        labels = torch.ones((1,), dtype=torch.int64) # Label 1 for our single class
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.clamp(area, min=0.01)
        iscrowd = torch.zeros((1,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        return img_tensor, target

class VideoAnnotatorApp:
    def __init__(self, root, video_dir_path):
        self.root = root
        self.root.title("Video Frame Annotator")

        self.video_dir = Path(video_dir_path)
        if not self.video_dir.is_dir():
            messagebox.showerror("Error", f"Video directory not found: {self.video_dir}")
            self.root.destroy()
            return

        self.video_files = []
        self.annotated_data = []

        if os.path.exists("annotations.yaml"):
            with open("annotations.yaml", "r") as f:
                self.annotated_data = yaml.safe_load(f)

        # --- UI Elements ---
        self.canvas = tk.Canvas(root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        self.confirm_button = tk.Button(button_frame, text="Confirm Annotation (Spacebar)", command=self.confirm_annotation)
        self.confirm_button.pack(side=tk.LEFT, padx=5)

        self.retrain_button = tk.Button(button_frame, text="Retrain Model", command=self.start_retraining_thread)
        self.retrain_button.pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(root, text="Status: Initializing...")
        self.status_label.pack(pady=5)

        # --- State Variables ---
        self.current_filepath = None
        self.current_frame_number = None
        self.current_cv2_frame = None
        self.display_image = None 
        self.photo_image = None   

        self.bbox_coords = None  
        self.bbox_item = None    
        self.handles = {}        

        self.start_x = None
        self.start_y = None
        self.is_drawing = False
        self.is_resizing = False
        self.active_handle = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.detection_model = None
        self.model_checkpoint_path = "fine_tuned_detector.pth"
        self.num_classes = 2  
        self.confidence_threshold = 0.85

        # --- CORRECTED TRANSFORM ---
        self.image_transform_for_model = T.Compose([
            T.ToImage(),  # Convert PIL Image to torch.Tensor (unnormalized, C, H, W)
            T.ToDtype(torch.float32, scale=True), # Converts to float32 and scales to [0.0, 1.0]
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Optional
        ])
        # This transform is also used for inference in run_inference_on_current_frame

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.root.bind("<space>", self.on_spacebar_press) 

        self.load_video_files()
        if not self.video_files:
            messagebox.showerror("Error", f"No .MP4 files found in {self.video_dir} or its subdirectories.")
            self.root.destroy()
            return
        
        self.load_fine_tuned_model() 
        self.load_random_frame()

    def prepare_training_images_and_annotations(self, all_annotations, temp_dir_path_str):
        temp_dir = Path(temp_dir_path_str)
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Clean up old files in the temp directory (optional, but good practice)
        for item in temp_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir(): # If you create subdirs, handle them too
                import shutil
                shutil.rmtree(item)
        
        prepared_image_annotations = []
        image_counter = 0

        self.status_label.config(text="Preparing training images...")
        self.root.update_idletasks()
        
        valid_annotations = [ann for ann in all_annotations if ann[2] is not None]

        for i, (video_filepath_str, frame_num, bbox_coords_orig) in enumerate(valid_annotations):
            if i % 10 == 0 or i == len(valid_annotations) -1 : # Update status periodically
                self.status_label.config(text=f"Extracting frame {i+1}/{len(valid_annotations)}...")
                self.root.update_idletasks()

            video_filepath = Path(video_filepath_str)
            try:
                cap = cv2.VideoCapture(str(video_filepath))
                if not cap.isOpened():
                    print(f"Warning: Could not open video {video_filepath} during data prep.")
                    continue
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame_cv2 = cap.read()
                cap.release()

                if not ret or frame_cv2 is None:
                    print(f"Warning: Could not read frame {frame_num} from {video_filepath} during data prep.")
                    continue

                # Save the frame as an image
                img_filename = f"frame_{image_counter:05d}.png" # Using PNG for lossless
                img_save_path = temp_dir / img_filename
                cv2.imwrite(str(img_save_path), frame_cv2) # Save BGR frame directly
                
                prepared_image_annotations.append((str(img_save_path), bbox_coords_orig))
                image_counter += 1

            except Exception as e:
                print(f"Error processing video {video_filepath} frame {frame_num}: {e}")
        
        if not prepared_image_annotations:
            messagebox.showerror("Data Prep Error", "No valid images could be prepared for training.")
            return None

        print(f"Prepared {len(prepared_image_annotations)} images for training in {temp_dir}")
        return prepared_image_annotations

    def get_object_detection_model(self, num_classes, pretrained=True):
        # Load a pre-trained model (e.g., Faster R-CNN)
        # torchvision.models.detection.fasterrcnn_resnet50_fpn
        # torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn for a lighter model
        try:
            if pretrained:
                model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            else: # For loading a checkpoint that wasn't from default weights initially
                model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)


            # Get the number of input features for the classifier
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            # Replace the pre-trained head with a new one
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            return model
        except Exception as e:
            messagebox.showerror("Model Download Error", f"Failed to download/initialize pre-trained model. Check internet connection or torchvision version.\nError: {e}")
            print(f"Model loading/download error: {e}")
            return None


    def load_fine_tuned_model(self):
        if os.path.exists(self.model_checkpoint_path):
            try:
                self.status_label.config(text=f"Loading fine-tuned model from {self.model_checkpoint_path}...")
                self.root.update_idletasks()
                # Initialize model structure first (important if num_classes changed or not using default pretrained for base)
                self.detection_model = self.get_object_detection_model(self.num_classes, pretrained=False) # Pretrained=False here because we load full state_dict
                if not self.detection_model: return

                self.detection_model.load_state_dict(torch.load(self.model_checkpoint_path, map_location=self.device))
                self.detection_model.to(self.device)
                self.detection_model.eval()
                print(f"Loaded fine-tuned model from {self.model_checkpoint_path}")
                self.status_label.config(text="Fine-tuned model loaded.")
            except Exception as e:
                messagebox.showerror("Load Error", f"Could not load model checkpoint: {e}")
                print(f"Error loading checkpoint: {e}")
                self.detection_model = None # Ensure model is None if loading failed
                self.status_label.config(text="Failed to load fine-tuned model.")
        else:
            print(f"No fine-tuned model checkpoint found at {self.model_checkpoint_path}.")
            self.status_label.config(text="No pre-trained checkpoint found. Annotate and Retrain.")
            self.detection_model = None


    def start_retraining_thread(self):
        if not self.annotated_data:
            messagebox.showinfo("Retrain", "No annotations available to train the model.")
            return

        annotations_with_boxes = [ann for ann in self.annotated_data if ann[2] is not None]
        if len(annotations_with_boxes) < 5: # Minimum for train/val split to make sense
            messagebox.showinfo("Retrain", f"Not enough annotations with bounding boxes for a train/val split. Need at least 5, found {len(annotations_with_boxes)}.")
            return

        self.confirm_button.config(state=tk.DISABLED)
        self.retrain_button.config(state=tk.DISABLED)
        
        # Prepare images and get list of (image_path, bbox)
        # Pass TEMP_TRAINING_DATA_DIR to the preparation function
        prepared_data = self.prepare_training_images_and_annotations(self.annotated_data, TEMP_TRAINING_DATA_DIR)

        if not prepared_data:
            self.status_label.config(text="Data preparation failed. Cannot retrain.")
            self.confirm_button.config(state=tk.NORMAL)
            self.retrain_button.config(state=tk.NORMAL)
            return

        # Shuffle and split data (80/20)
        random.shuffle(prepared_data)
        split_idx = int(0.8 * len(prepared_data))
        train_image_annotations = prepared_data[:split_idx]
        val_image_annotations = prepared_data[split_idx:]

        if not train_image_annotations:
            messagebox.showwarning("Retrain", "Not enough data for a training set after split.")
            self.confirm_button.config(state=tk.NORMAL)
            self.retrain_button.config(state=tk.NORMAL)
            return
        if not val_image_annotations: # Can proceed without val, but good to have
            print("Warning: No validation data after split. Proceeding with training only.")


        self.status_label.config(text="Retraining model... This may take a while.")
        self.root.update_idletasks()

        thread = threading.Thread(target=self.retrain_model, args=(train_image_annotations, val_image_annotations))
        thread.daemon = True
        thread.start()

    def retrain_model(self, train_image_annotations, val_image_annotations): # Now takes prepared image annotations
        error_message_for_status = None
        try:
            print("Starting model retraining...")
            
            # 1. Prepare Datasets and DataLoaders
            train_dataset = AnnotationDataset(train_image_annotations, transform=self.image_transform_for_model)
            
            val_dataset = None
            if val_image_annotations:
                val_dataset = AnnotationDataset(val_image_annotations, transform=self.image_transform_for_model)

            def collate_fn(batch):
                return tuple(zip(*batch))
            
            batch_size_train = 1 if len(train_dataset) < 4 else 2
            train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0, collate_fn=collate_fn)
            
            val_loader = None
            if val_dataset:
                batch_size_val = 1 if len(val_dataset) < 4 else 2
                val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, num_workers=0, collate_fn=collate_fn)

            # 2. Initialize or load model
            if os.path.exists(self.model_checkpoint_path) and self.detection_model:
                print("Continuing training from existing fine-tuned model.")
                model = self.detection_model
            else:
                print("Starting training from a fresh pre-trained model.")
                model = self.get_object_detection_model(self.num_classes, pretrained=True)
            
            if not model:
                error_message_for_status = "Retraining failed: Could not get model."
                return # Finally block will handle UI

            model.to(self.device)

            # 3. Optimizer
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

            # reduce LR by factor of 10 every 3 epochs
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

            early_stopping = PatienceEarlyStopping(patience=3, min_delta=0.001)
            
            num_epochs = 20
            print(f"Training for {num_epochs} epochs...")

            best_val_loss = float('inf') # For saving the best model based on validation

            for epoch in range(num_epochs):
                # --- Training Phase ---
                model.train()
                train_epoch_loss = 0
                print(f"--- Starting Epoch {epoch+1}/{num_epochs} (Training) ---")
                for i, (images_batch, targets_batch) in enumerate(train_loader):
                    images = list(img.to(self.device) for img in images_batch)
                    targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets_batch]

                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()

                    current_loss = losses.item()
                    train_epoch_loss += current_loss
                    if (i + 1) % 1 == 0 or i == len(train_loader) - 1:
                        print(f"  Epoch {epoch+1} (Train), Batch {i+1}/{len(train_loader)}, Loss: {current_loss:.4f}")
                
                avg_train_loss = train_epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
                print(f"--- Epoch {epoch+1} Average Training Loss: {avg_train_loss:.4f} ---")

                # --- Validation Phase ---
                if val_loader:
                    model.eval() # Set model to evaluation mode for things like BatchNorm, Dropout
                    val_epoch_loss = 0
                    print(f"--- Starting Epoch {epoch+1}/{num_epochs} (Validation) ---")
                    
                    # Store current training state of the model for the forward pass to get losses
                    # This is a common pattern if a model's forward pass for loss calculation
                    # is tied to its training state but you don't want to update parameters.
                    is_training_before_val_loss_calc = model.training
                    model.train() # Temporarily set to train mode to get loss dict

                    with torch.no_grad(): # Crucial: ensure no gradients are computed/stored
                        for i, (images_batch, targets_batch) in enumerate(val_loader):
                            images = list(img.to(self.device) for img in images_batch)
                            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets_batch]
                            
                            # Now, model(images, targets) inside model.train() but with no_grad()
                            # should return the loss dictionary.
                            loss_dict_val = model(images, targets) 
                            losses_val = sum(loss for loss in loss_dict_val.values()) # This should work now
                            
                            val_epoch_loss += losses_val.item()
                            if (i + 1) % 1 == 0 or i == len(val_loader) - 1:
                                print(f"  Epoch {epoch+1} (Val), Batch {i+1}/{len(val_loader)}, Val Loss: {losses_val.item():.4f}")
                    
                    # Restore the original evaluation mode after loss calculation
                    if not is_training_before_val_loss_calc: # if it was eval before this block
                        model.eval() 
                    # Or simply: model.eval() # as we want it in eval mode for actual inference later

                    avg_val_loss = val_epoch_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
                    print(f"--- Epoch {epoch+1} Average Validation Loss: {avg_val_loss:.4f} ---")

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        torch.save(model.state_dict(), self.model_checkpoint_path)
                        print(f"New best model saved with validation loss: {best_val_loss:.4f} to {self.model_checkpoint_path}")
                else:
                    torch.save(model.state_dict(), self.model_checkpoint_path)
                    print(f"Model saved after epoch {epoch+1} (no validation) to {self.model_checkpoint_path}")

                lr_scheduler.step()
                early_stopping(avg_val_loss)
                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch+1} due to no validation loss improvement.")
                    break
            
            if not val_loader: 
                 print(f"Retraining complete (no validation). Final model saved to {self.model_checkpoint_path}")
            else:
                 print(f"Retraining complete. Best model (Val Loss: {best_val_loss:.4f}) saved to {self.model_checkpoint_path}")
            
        except Exception as exc: 
            print(f"Error during retraining: {exc}")
            import traceback
            traceback.print_exc()
            error_message_for_status = f"Retraining failed: {exc}"
        finally:
            if error_message_for_status:
                self.root.after(0, lambda msg=error_message_for_status: self.status_label.config(text=msg))
            else:
                self.root.after(0, lambda: self.status_label.config(text="Retraining finished. Check console for details."))
                self.root.after(0, self.load_fine_tuned_model) # Load the (potentially best) saved model

            self.root.after(0, lambda: self.confirm_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.retrain_button.config(state=tk.NORMAL))


    def run_inference_on_current_frame(self):
        if not self.detection_model or self.current_cv2_frame is None:
            return

        # self.status_label.config(text="Running inference...") # Can be too flashy
        # self.root.update_idletasks()

        try:
            img_rgb = cv2.cvtColor(self.current_cv2_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # Transform for model input
            img_tensor = self.image_transform_for_model(img_pil)
            img_tensor = img_tensor.to(self.device)

            self.detection_model.eval()
            with torch.no_grad():
                prediction = self.detection_model([img_tensor]) # Model expects a list of images

            # Process prediction
            # prediction is a list of dicts, one per image. We sent one image.
            pred_boxes = prediction[0]['boxes']
            pred_scores = prediction[0]['scores']
            # pred_labels = prediction[0]['labels'] # We only have one class + background

            best_score = 0
            best_box = None

            for i in range(len(pred_boxes)):
                score = pred_scores[i].item()
                if score > self.confidence_threshold and score > best_score:
                    best_score = score
                    box = pred_boxes[i].cpu().numpy() # [x1, y1, x2, y2]
                    best_box = box

            if best_box is not None:
                print(f"Inference: Found object with score {best_score:.2f} at {best_box}")
                # Bounding box from model is in original image coordinates.
                # We need to scale it to the DISPLAYED image coordinates for drawing.
                
                orig_width = self.current_cv2_frame.shape[1]
                orig_height = self.current_cv2_frame.shape[0]
                
                disp_width = self.photo_image.width()
                disp_height = self.photo_image.height()

                scale_x = disp_width / orig_width if orig_width > 0 else 1
                scale_y = disp_height / orig_height if orig_height > 0 else 1

                x1_disp = int(best_box[0] * scale_x)
                y1_disp = int(best_box[1] * scale_y)
                x2_disp = int(best_box[2] * scale_x)
                y2_disp = int(best_box[3] * scale_y)
                
                # Clear any manually drawn box before drawing inferred one
                self.clear_bbox_data()
                self.bbox_coords = [x1_disp, y1_disp, x2_disp, y2_disp]
                self.draw_bbox_and_handles()
                self.status_label.config(text=f"Inferred box (Score: {best_score:.2f}). Adjust if needed.")
            # else:
            #    self.status_label.config(text=f"File: {self.current_filepath.name} | Frame: {self.current_frame_number}")


        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            # self.status_label.config(text=f"Inference error: {e}")

    def on_spacebar_press(self, event): # ADDED: Handler for spacebar
        """Handles the spacebar press event to confirm annotation."""
        # We check event.widget to ensure the spacebar isn't pressed while
        # a text entry field has focus (if you were to add any in the future).
        # For this simple app, it's less critical but good practice.
        if isinstance(event.widget, tk.Entry) or isinstance(event.widget, tk.Text):
            return # Don't trigger if typing in an entry/text widget
        self.confirm_annotation()

    def load_video_files(self):
        self.status_label.config(text="Scanning for MP4 files...")
        self.root.update_idletasks()
        self.video_files = list(self.video_dir.rglob("*.MP4")) + list(self.video_dir.rglob("*.mp4"))
        if not self.video_files:
             self.status_label.config(text=f"No .MP4 files found in {self.video_dir}")
        else:
            self.status_label.config(text=f"Found {len(self.video_files)} MP4 files.")
        print(f"Found videos: {self.video_files}")


    def select_random_video_and_frame(self):
        if not self.video_files:
            self.current_filepath = None
            self.current_cv2_frame = None
            self.current_frame_number = -1
            messagebox.showerror("Error", "No video files available.")
            return None, -1, None

        random_video_path = random.choice(self.video_files)
        cap = cv2.VideoCapture(str(random_video_path))

        if not cap.isOpened():
            messagebox.showerror("Error", f"Could not open video: {random_video_path}")
            # Try to remove problematic file and retry, or just error out
            self.video_files.remove(random_video_path)
            if not self.video_files:
                self.root.destroy() # No more videos to try
                return None, -1, None
            return self.select_random_video_and_frame() # Recursive call to try another file

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            messagebox.showwarning("Warning", f"Video has no frames or invalid frame count: {random_video_path}")
            cap.release()
            self.video_files.remove(random_video_path)
            if not self.video_files:
                self.root.destroy()
                return None, -1, None
            return self.select_random_video_and_frame()

        random_frame_num = random.randint(0, frame_count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_num)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            messagebox.showerror("Error", f"Could not read frame {random_frame_num} from {random_video_path}")
            self.video_files.remove(random_video_path)
            if not self.video_files:
                self.root.destroy()
                return None, -1, None
            return self.select_random_video_and_frame()

        self.current_filepath = random_video_path
        self.current_frame_number = random_frame_num
        self.current_cv2_frame = frame
        
        self.status_label.config(text=f"File: {self.current_filepath.name} | Frame: {self.current_frame_number}")
        return random_video_path, random_frame_num, frame

    def display_frame(self):
        if self.current_cv2_frame is None:
            self.canvas.delete("all") # Clear canvas if no frame
            # Potentially display a placeholder image or message
            self.canvas.create_text(self.canvas.winfo_width()/2, self.canvas.winfo_height()/2, text="No Frame Loaded", anchor="center")
            return

        frame_rgb = cv2.cvtColor(self.current_cv2_frame, cv2.COLOR_BGR2RGB)
        self.display_image = Image.fromarray(frame_rgb)
        
        # Resize image to fit canvas while maintaining aspect ratio (optional, but good for large videos)
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width < 2 or canvas_height < 2 : # Canvas not yet sized
            canvas_width = 800 # default
            canvas_height = 600

        img_width, img_height = self.display_image.size
        
        ratio = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)

        if new_width > 0 and new_height > 0:
            resized_image = self.display_image.resize((new_width, new_height), Image.LANCZOS)
            self.photo_image = ImageTk.PhotoImage(resized_image)
        else: # Fallback if resizing fails or results in zero dimensions
            self.photo_image = ImageTk.PhotoImage(self.display_image)


        self.canvas.delete("all") # Clear previous image and bbox
        self.canvas.config(width=self.photo_image.width(), height=self.photo_image.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
        # Keep a reference to avoid garbage collection!
        self.canvas.image = self.photo_image 
        
        self.clear_bbox_data() # Clear previous bbox for the new image

        # ADDED: Run inference on the newly displayed frame
        if self.detection_model and self.current_cv2_frame is not None:
             self.root.after(50, self.run_inference_on_current_frame) # Short delay for canvas to settle
        else:
            # Reset status label if no model or no frame
            if self.current_filepath: # To avoid resetting "No pre-trained checkpoint found"
                 self.status_label.config(text=f"File: {self.current_filepath.name} | Frame: {self.current_frame_number}")

    def load_random_frame(self):
        self.select_random_video_and_frame()
        self.root.after(50, self.display_frame) # Give canvas time to get its size

    def clear_bbox_visuals(self):
        if self.bbox_item:
            self.canvas.delete(self.bbox_item)
            self.bbox_item = None
        for handle_id in self.handles.values():
            self.canvas.delete(handle_id)
        self.handles = {}
    
    def clear_bbox_data(self):
        self.clear_bbox_visuals()
        self.bbox_coords = None

    def draw_bbox_and_handles(self):
        self.clear_bbox_visuals() # Clear old ones first

        if self.bbox_coords:
            x1, y1, x2, y2 = self.bbox_coords
            # Ensure x1 < x2 and y1 < y2
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            self.bbox_coords = [x1, y1, x2, y2]

            self.bbox_item = self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline="red", width=2, tags="bbox"
            )
            
            # Define handle positions
            hs = HANDLE_SIZE // 2
            handle_positions = {
                'nw': (x1 - hs, y1 - hs, x1 + hs, y1 + hs),
                'ne': (x2 - hs, y1 - hs, x2 + hs, y1 + hs),
                'sw': (x1 - hs, y2 - hs, x1 + hs, y2 + hs),
                'se': (x2 - hs, y2 - hs, x2 + hs, y2 + hs),
                'n':  ((x1 + x2) // 2 - hs, y1 - hs, (x1 + x2) // 2 + hs, y1 + hs),
                's':  ((x1 + x2) // 2 - hs, y2 - hs, (x1 + x2) // 2 + hs, y2 + hs),
                'w':  (x1 - hs, (y1 + y2) // 2 - hs, x1 + hs, (y1 + y2) // 2 + hs),
                'e':  (x2 - hs, (y1 + y2) // 2 - hs, x2 + hs, (y1 + y2) // 2 + hs),
            }
            for name, coords in handle_positions.items():
                self.handles[name] = self.canvas.create_rectangle(
                    coords, fill="red", outline="black", tags="handle"
                )

    def get_handle_at_pos(self, x, y):
        for name, handle_id in self.handles.items():
            coords = self.canvas.coords(handle_id)
            if coords and coords[0] <= x <= coords[2] and coords[1] <= y <= coords[3]:
                return name
        return None

    def on_mouse_press(self, event):
        self.start_x = event.x
        self.start_y = event.y

        self.active_handle = self.get_handle_at_pos(event.x, event.y)

        if self.active_handle:
            self.is_resizing = True
            self.is_drawing = False
        elif self.bbox_coords:
            # Clicked outside existing bbox (and not on a handle)
            self.clear_bbox_data() 
            self.is_drawing = True # Start drawing new one on drag
            self.is_resizing = False
        else:
            # No bbox, start drawing
            self.is_drawing = True
            self.is_resizing = False
            self.bbox_coords = [self.start_x, self.start_y, self.start_x, self.start_y]


    def on_mouse_drag(self, event):
        if not (self.is_drawing or self.is_resizing):
            return

        cur_x, cur_y = event.x, event.y
        
        # Clamp coordinates to be within canvas boundaries
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        cur_x = max(0, min(cur_x, canvas_width))
        cur_y = max(0, min(cur_y, canvas_height))


        if self.is_drawing and self.bbox_coords:
            self.bbox_coords[2] = cur_x
            self.bbox_coords[3] = cur_y
            self.draw_bbox_and_handles()

        elif self.is_resizing and self.active_handle and self.bbox_coords:
            x1, y1, x2, y2 = self.bbox_coords
            
            if 'n' in self.active_handle: y1 = cur_y
            if 's' in self.active_handle: y2 = cur_y
            if 'w' in self.active_handle: x1 = cur_x
            if 'e' in self.active_handle: x2 = cur_x
            
            # Ensure x1 < x2 and y1 < y2 for handles, then actual draw will sort it
            self.bbox_coords = [min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)]

            self.draw_bbox_and_handles()


    def on_mouse_release(self, event):
        if self.is_drawing or self.is_resizing:
            self.is_drawing = False
            self.is_resizing = False
            self.active_handle = None

            if self.bbox_coords:
                x1, y1, x2, y2 = self.bbox_coords
                # Normalize: x1 < x2, y1 < y2
                self.bbox_coords = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                
                # Check for minimum size
                if abs(self.bbox_coords[2] - self.bbox_coords[0]) < MIN_BBOX_SIZE or \
                   abs(self.bbox_coords[3] - self.bbox_coords[1]) < MIN_BBOX_SIZE:
                    self.clear_bbox_data() # Remove if too small
                else:
                    self.draw_bbox_and_handles() # Redraw to finalize and show handles correctly

    def confirm_annotation(self):
        if self.current_filepath is None:
            messagebox.showwarning("Warning", "No video/frame loaded to annotate.")
            return

        final_bbox_coords_orig = None # This will be in original image coordinates
        if self.bbox_coords: # self.bbox_coords are in DISPLAYED image space
            x1_disp, y1_disp, x2_disp, y2_disp = self.bbox_coords
            
            # self.display_image is the PIL version of the original frame (before display resize)
            # self.photo_image is the TkInter photo image (after display resize)
            if self.display_image and self.photo_image:
                orig_width = self.display_image.width  # Original frame width
                orig_height = self.display_image.height # Original frame height
                
                disp_width = self.photo_image.width()
                disp_height = self.photo_image.height()

                scale_x_to_orig = orig_width / disp_width if disp_width > 0 else 1
                scale_y_to_orig = orig_height / disp_height if disp_height > 0 else 1
                
                x1_orig = int(x1_disp * scale_x_to_orig)
                y1_orig = int(y1_disp * scale_y_to_orig)
                x2_orig = int(x2_disp * scale_x_to_orig)
                y2_orig = int(y2_disp * scale_y_to_orig)
                final_bbox_coords_orig = (x1_orig, y1_orig, x2_orig, y2_orig)
            else: # Fallback if images aren't ready, though unlikely here
                print("Warning: display_image or photo_image not available for coordinate scaling.")


        annotation = [
            str(self.current_filepath),
            self.current_frame_number,
            list(final_bbox_coords_orig) if final_bbox_coords_orig else None # Store original coordinates
        ]
        self.annotated_data.append(annotation)
        print(f"Annotation Confirmed: {annotation}")
        print(f"Total annotations: {len(self.annotated_data)}")

        # For demonstration, you might want to save this list to a file periodically or on exit
        # For now, it's just in memory.

        # Load next random frame
        self.load_random_frame()

    def run(self):
        self.root.mainloop()
        # save annotations to yaml
        with open("annotations.yaml", "w") as f:
            yaml.dump(self.annotated_data, f)
        # You could save data here on close:
        # with open("annotations.txt", "w") as f:
        #    for item in self.annotated_data:
        #        f.write(str(item) + "\n")


if __name__ == "__main__":
    # --- Create dummy video files for testing if they don't exist ---
    # This is just for making the example runnable without manual setup.
    # You should replace VIDEO_DIRECTORY with your actual path.
    
    video_test_dir = Path(VIDEO_DIRECTORY)
    video_test_dir.mkdir(parents=True, exist_ok=True)

    if not any(video_test_dir.rglob("*.mp4")) and not any(video_test_dir.rglob("*.MP4")):
        print(f"Creating dummy MP4 files in {video_test_dir} for testing...")
        try:
            for i in range(2):
                # Create a very short, small dummy mp4 file
                # Requires ffmpeg to be installed and in PATH for this dummy creation
                # If you don't have ffmpeg, this part will fail, but the app will still
                # run if you manually place MP4s in the VIDEO_DIRECTORY.
                import subprocess
                dummy_file = video_test_dir / f"dummy_video_{i+1}.mp4"
                # ffmpeg command to create a 1-second black video
                command = [
                    'ffmpeg', '-y', '-f', 'lavfi', '-i', 'color=c=black:s=128x72:r=1', 
                    '-t', '1', '-pix_fmt', 'yuv420p', str(dummy_file)
                ]
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("Dummy files created.")
        except Exception as e:
            print(f"Could not create dummy MP4 files (ffmpeg might be missing or failed): {e}")
            print(f"Please manually place some .MP4 files in the '{VIDEO_DIRECTORY}' directory.")
    # --- End dummy file creation ---

    root = tk.Tk()
    app = VideoAnnotatorApp(root, VIDEO_DIRECTORY)
    
    # Set a minimum size for the window to ensure canvas is initially visible
    root.minsize(600, 400) 

    initial_width, initial_height = 1280, 900
    root.geometry(f"{initial_width}x{initial_height}")
    
    # Give the canvas a moment to initialize its size before the first display_frame call
    # that depends on canvas.winfo_width/height
    def delayed_start():
        if root.winfo_width() > 1 and root.winfo_height() > 1: # Check if window is mapped
            if not app.video_files and VIDEO_DIRECTORY == "videos": # If still no files and using default path
                 messagebox.showinfo("Setup Note", f"No MP4s found. Please put .MP4 files in the '{VIDEO_DIRECTORY}' folder and restart, or change the VIDEO_DIRECTORY variable in the script.")
            elif app.video_files: # if files were found during init or by dummy creation
                app.display_frame() # Call this again to ensure correct sizing
        else:
            root.after(100, delayed_start) # Wait longer

    if app.video_files: # Only schedule if app initialized correctly and found videos
        root.after(100, delayed_start) # Initial delay to allow window to draw and get dimensions
    
    app.run()