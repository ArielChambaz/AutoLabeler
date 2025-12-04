# auto_annotate_yolo.py
# pip install ultralytics opencv-python tkinter torch transformers pillow rfdetr supervision
import os, glob, shutil, cv2
import tkinter as tk
from tkinter import filedialog, messagebox

# --- IMPORTS ---
from ultralytics import YOLO
import torch
from transformers import DFineForObjectDetection, AutoImageProcessor
from PIL import Image

# IMPORTS SPÉCIFIQUES À VOTRE MODÈLE RFDETR
try:
    from rfdetr import RFDETRBase
    import supervision as sv
except ImportError:
    # Laissez l'erreur se manifester si les dépendances ne sont pas là
    pass 

# Default model (if path field is left empty)
DEFAULT_HF_MODEL = "ustc-community/dfine-xlarge-obj365"

# ------------------------
# CUSTOM MODEL DEFINITIONS (RFDETR)
# ------------------------

# Définition des classes pour le modèle CUSTOM
# Votre seule classe est 'person' avec l'ID 0
CUSTOM_CLASS_NAMES = {
    0: "person", 
}

# ------------------------
# MAIN FUNCTIONS
# ------------------------

def extract_frames_from_video(video_path, output_dir, frame_interval=30, time_interval=None):
    """Extract frames from a video at a given interval."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {os.path.basename(video_path)} | FPS: {fps} | Total frames: {total_frames}")

    # Determine extraction interval
    if time_interval is not None:
        interval_frames = int(fps * time_interval)
        print(f"Extracting 1 frame every {time_interval}s ({interval_frames} frames)")
    else:
        interval_frames = frame_interval
        print(f"Extracting 1 frame every {frame_interval} frames")
    
    os.makedirs(output_dir, exist_ok=True)
    extracted_paths = []
    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval_frames == 0:
            timestamp = frame_count / fps
            minutes, seconds = int(timestamp // 60), int(timestamp % 60)
            frame_filename = f"frame_{frame_count:06d}_{minutes:02d}m{seconds:02d}s.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            extracted_paths.append(frame_path)
            extracted_count += 1
            if extracted_count % 10 == 0:
                print(f"Extracted {extracted_count} frames...")
        frame_count += 1

    cap.release()
    print(f"✅ Extraction complete: {extracted_count} frames saved")
    return extracted_paths


def create_output_directory(video_path=None):
    """Create an output directory based on the video name or a default one."""
    if video_path:
        name = os.path.splitext(os.path.basename(video_path))[0]
        out_root = name
    else:
        out_root = "images_dataset"
    os.makedirs(out_root, exist_ok=True)
    return out_root


def process_images(model_path, video_path, images_dir, frame_interval, time_interval, conf, iou):
    """Full pipeline: video frame extraction → Prediction → Label generation."""
    
    # 0. Détermination automatique du type de modèle
    model_type = "UNKNOWN"
    target_model_path = ""
    
    if not model_path or model_path.strip() == "":
        model_type = "DFINE_ONLINE"
    else:
        ext = os.path.splitext(model_path)[-1].lower()
        if ext == ".pt":
            model_type = "YOLO"
            target_model_path = model_path
        elif ext in [".pth", ".safetensors"]:
            # NOUVELLE VÉRIFICATION : Teste si le nom de fichier contient 'rfdetr'
            if "rfdetr" in os.path.basename(model_path).lower():
                model_type = "RFDETR"
                target_model_path = model_path
            else:
                 messagebox.showerror("Model Error", 
                    "Ce script ne sait traiter que les modèles .pth/.safetensors s'ils sont de type RFDETR (nom de fichier contenant 'rfdetr').\n"
                    "Veuillez choisir un modèle YOLO (.pt) ou D-FINE, ou renommer votre fichier.")
                 return
        elif os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
             model_type = "DFINE_LOCAL"
             target_model_path = model_path
        else:
            messagebox.showerror("Error", f"Unsupported file or folder structure: {model_path}")
            return

    print(f"Model Path: {model_path} | Detected Type: {model_type}")

    # 1. Image preparation (identique)
    if video_path and os.path.exists(video_path):
        print("\n🎥 Video mode enabled")
        OUT_ROOT = create_output_directory(video_path)
        IMG_DIR = os.path.join(OUT_ROOT, "images")
        os.makedirs(IMG_DIR, exist_ok=True)
        print("Extracting frames...")
        image_paths = extract_frames_from_video(video_path, IMG_DIR, frame_interval, time_interval)
    else:
        print("\nImage mode enabled")
        OUT_ROOT = create_output_directory()
        IMG_DIR = os.path.join(OUT_ROOT, "images")
        os.makedirs(IMG_DIR, exist_ok=True)

        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        image_paths = []
        for e in exts:
            image_paths += glob.glob(os.path.join(images_dir, e))

        copied_paths = []
        for img_path in image_paths:
            fn = os.path.basename(img_path)
            dst = os.path.join(IMG_DIR, fn)
            shutil.copy2(img_path, dst)
            copied_paths.append(dst)
        image_paths = copied_paths

    print(f"{len(image_paths)} images to process")
    LBL_DIR = os.path.join(OUT_ROOT, "labels")
    os.makedirs(LBL_DIR, exist_ok=True)


    # ==========================================
    # BRANCH 1 : YOLO (Ultralytics - .pt)
    # ==========================================
    if model_type == "YOLO":
        print(f"🚀 Loading YOLO model from file: {target_model_path}")
        try:
            model = YOLO(target_model_path)
            names = model.names 
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load YOLO model: {e}")
            return

        with open(os.path.join(OUT_ROOT, "data.yaml"), "w") as f:
            f.write("names:\n")
            for cid, cname in names.items():
                f.write(f"  {cid}: {cname}\n")

        for img_path in image_paths:
            fn = os.path.basename(img_path)
            stem, _ = os.path.splitext(fn)
            results = model.predict(img_path, conf=conf, iou=iou, verbose=False)[0]
            txt_path = os.path.join(LBL_DIR, f"{stem}.txt")
            with open(txt_path, "w") as f:
                if results.boxes is not None and len(results.boxes) > 0:
                    for b in results.boxes:
                        cls = int(b.cls.item())
                        xcn, ycn, wn, hn = b.xywhn[0].tolist()
                        f.write(f"{cls} {xcn:.6f} {ycn:.6f} {wn:.6f} {hn:.6f}\n")
            print(f"OK (YOLO) {fn} -> {txt_path}")

    # ==========================================
    # BRANCH 2 : D-FINE (Hugging Face - Online ou Local avec config.json)
    # ==========================================
    elif model_type in ["DFINE_ONLINE", "DFINE_LOCAL"]:
        
        # --- LOADING LOGIC ---
        target_hf_model = DEFAULT_HF_MODEL if model_type == "DFINE_ONLINE" else target_model_path
        print(f"🌐 Loading D-FINE model from: {target_hf_model}")
        
        try:
            image_processor = AutoImageProcessor.from_pretrained(target_hf_model)
            model = DFineForObjectDetection.from_pretrained(target_hf_model)
            names = model.config.id2label
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load D-FINE model.\nSource: {target_hf_model}\n\nError: {e}")
            return

        with open(os.path.join(OUT_ROOT, "data.yaml"), "w") as f:
            f.write("names:\n")
            for cid, cname in names.items():
                f.write(f"  {cid}: {cname}\n")

        for img_path in image_paths:
            fn = os.path.basename(img_path)
            stem, _ = os.path.splitext(fn)

            try:
                image_pil = Image.open(img_path).convert("RGB")
            except Exception:
                print(f"❌ Skipped invalid image: {img_path}")
                continue

            width, height = image_pil.size
            inputs = image_processor(images=image_pil, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            target_sizes = torch.tensor([(height, width)]) 
            
            results = image_processor.post_process_object_detection(
                outputs, 
                threshold=conf, 
                target_sizes=target_sizes
            )[0]

            txt_path = os.path.join(LBL_DIR, f"{stem}.txt")
            with open(txt_path, "w") as f:
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    if score >= conf:
                        xmin, ymin, xmax, ymax = box.tolist()
                        
                        # Conversion to normalized YOLO format
                        x_center = ((xmin + xmax) / 2) / width
                        y_center = ((ymin + ymax) / 2) / height
                        box_width = (xmax - xmin) / width
                        box_height = (ymax - ymin) / height
                        
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        box_width = max(0, min(1, box_width))
                        box_height = max(0, min(1, box_height))

                        cls = label.item()
                        f.write(f"{cls} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
            print(f"OK (D-FINE) {fn} -> {txt_path}")

    # ==========================================
    # BRANCH 3 : VOTRE MODÈLE CUSTOM (RFDETR, .pth/.safetensors)
    # ==========================================
    elif model_type == "RFDETR":
        print(f"🛠️ Loading RFDETR model from: {target_model_path}")
        
        # --- (A) Initialisation du Modèle et Chargement des Poids ---
        try:
            # INSTANCIATION SIMPLE : Utilise l'API RFDETRBase pour charger le modèle
            model = RFDETRBase(pretrain_weights=target_model_path)
            
            names = CUSTOM_CLASS_NAMES
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load RFDETR model.\n\nError: {e}")
            return

        with open(os.path.join(OUT_ROOT, "data.yaml"), "w") as f:
            f.write("names:\n")
            for cid, cname in names.items():
                f.write(f"  {cid}: {cname}\n")

        # --- (B) Boucle de Prédiction RFDETR et Conversion YOLO ---
        for img_path in image_paths:
            fn = os.path.basename(img_path)
            stem, _ = os.path.splitext(fn)
            txt_path = os.path.join(LBL_DIR, f"{stem}.txt")
            
            try:
                # 1. Charger l'image en PIL (requis par RFDETR.predict)
                image_pil = Image.open(img_path).convert("RGB")
                width, height = image_pil.size
                
                # 2. Faire la prédiction (L'API RFDETR/Supervision fait le travail)
                # La sortie est un objet supervision.Detections
                detections = model.predict(image_pil, threshold=conf)
                
                with open(txt_path, "w") as f:
                    # 3. Conversion du format Supervision au format YOLO
                    if detections.xyxy.shape[0] > 0:
                        
                        xyxy = detections.xyxy 
                        class_ids = detections.class_id 
                        
                        # Si les class_ids sont manquants (parfois le cas pour un modèle 1-classe), nous forçons à 0
                        if class_ids is None or len(class_ids) != xyxy.shape[0]:
                             class_ids = [0] * xyxy.shape[0]
                        
                        for bbox, cls_id in zip(xyxy, class_ids):
                            xmin, ymin, xmax, ymax = bbox.tolist()
                            
                            # Conversion en coordonnées normalisées YOLO
                            x_center = ((xmin + xmax) / 2) / width
                            y_center = ((ymin + ymax) / 2) / height
                            box_width = (xmax - xmin) / width
                            box_height = (ymax - ymin) / height
                            
                            # Clamp pour la sécurité
                            x_center = max(0, min(1, x_center))
                            y_center = max(0, min(1, y_center))
                            box_width = max(0, min(1, box_width))
                            box_height = max(0, min(1, box_height))
                            
                            cls = int(cls_id)
                            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
                            
                print(f"OK (RFDETR) {fn} -> {txt_path}")
            
            except Exception as e:
                print(f"❌ Erreur de traitement pour {fn}: {e}")
                continue


    # ==========================================
    # END OF PROCESSING
    # ==========================================
    detected = len([p for p in image_paths if os.path.exists(os.path.join(LBL_DIR, os.path.splitext(os.path.basename(p))[0] + ".txt")) and os.path.getsize(os.path.join(LBL_DIR, os.path.splitext(os.path.basename(p))[0] + ".txt")) > 0])
    
    print("\n✅ Processing complete!")
    messagebox.showinfo("Done", f"Processing complete!\nMode: {model_type}\n{detected} images annotated in {OUT_ROOT}")


# ------------------------
# GUI FUNCTIONS
# ------------------------

def browse_video():
    path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv")])
    if path:
        video_path_var.set(path)

def browse_model():
    title = "Select Model Weights (.pt for YOLO, .pth/.safetensors for Custom/D-FINE)"
    filetypes = [
        ("Model files", "*.pt *.pth *.safetensors"),
        ("All files", "*.*")
    ]
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        
    if path:
        model_path_var.set(path)

def browse_images():
    path = filedialog.askdirectory()
    if path:
        images_dir_var.set(path)

def run_process():
    try:
        model_path = model_path_var.get()
        video_path = video_path_var.get() or None
        images_dir = images_dir_var.get() or "images"
        
        # Validation des paramètres numériques
        try:
            frame_interval = int(frame_interval_var.get())
        except ValueError:
             messagebox.showerror("Error", "Frame interval must be an integer.")
             return
             
        try:
            time_interval_str = time_interval_var.get()
            time_interval = float(time_interval_str) if time_interval_str else None
        except ValueError:
             messagebox.showerror("Error", "Time interval must be a number.")
             return
        
        try:
            conf = float(conf_var.get())
            iou = float(iou_var.get())
        except ValueError:
             messagebox.showerror("Error", "Confidence and IoU must be numbers.")
             return
        
        if not (0 <= conf <= 1 and 0 <= iou <= 1):
             messagebox.showerror("Error", "Confidence and IoU must be between 0 and 1.")
             return

        # Le modèle est déterminé à l'intérieur de process_images
        process_images(model_path, video_path, images_dir, frame_interval, time_interval, conf, iou)

    except Exception as e:
        messagebox.showerror("Error", str(e))


# ------------------------
# TKINTER UI SETUP
# ------------------------

root = tk.Tk()
root.title("Auto Annotate (YOLO, D-FINE & Custom)")
root.geometry("600x500")
root.resizable(False, False)

# Variables
model_path_var = tk.StringVar()
video_path_var = tk.StringVar()
images_dir_var = tk.StringVar()
frame_interval_var = tk.StringVar(value="60")
time_interval_var = tk.StringVar(value="")
conf_var = tk.StringVar(value="0.55")
iou_var = tk.StringVar(value="0.45")

# 1. Sélectionner le Modèle
tk.Label(root, text="1. Model Path (YOLO .pt, Custom .pth, ou vide pour D-FINE en ligne):").pack(anchor="w", padx=20, pady=(10, 0))
entry_frame = tk.Frame(root)
entry_frame.pack(fill="x", padx=20)
tk.Entry(entry_frame, textvariable=model_path_var).pack(side="left", fill="x", expand=True)
tk.Button(entry_frame, text="Browse...", command=browse_model).pack(side="left", padx=5)

# 2. Video ou Images
tk.Label(root, text="2. Video (optional):").pack(anchor="w", padx=20, pady=(10, 0))
entry_frame_vid = tk.Frame(root)
entry_frame_vid.pack(fill="x", padx=20)
tk.Entry(entry_frame_vid, textvariable=video_path_var).pack(side="left", fill="x", expand=True)
tk.Button(entry_frame_vid, text="Browse...", command=browse_video).pack(side="left", padx=5)

tk.Label(root, text="3. Image folder (if no video):").pack(anchor="w", padx=20, pady=(10, 0))
entry_frame_img = tk.Frame(root)
entry_frame_img.pack(fill="x", padx=20)
tk.Entry(entry_frame_img, textvariable=images_dir_var).pack(side="left", fill="x", expand=True)
tk.Button(entry_frame_img, text="Choose folder...", command=browse_images).pack(side="left", padx=5)

# 4. Parameters
frame_params = tk.Frame(root)
frame_params.pack(pady=10)
tk.Label(frame_params, text="Frame interval:").grid(row=0, column=0, padx=5)
tk.Entry(frame_params, textvariable=frame_interval_var, width=6).grid(row=0, column=1, padx=5)
tk.Label(frame_params, text="Time interval (s):").grid(row=0, column=2, padx=5)
tk.Entry(frame_params, textvariable=time_interval_var, width=6).grid(row=0, column=3, padx=5)

frame_conf = tk.Frame(root)
frame_conf.pack(pady=5)
tk.Label(frame_conf, text="Confidence (CONF):").grid(row=0, column=0, padx=5)
tk.Entry(frame_conf, textvariable=conf_var, width=6).grid(row=0, column=1, padx=5)
tk.Label(frame_conf, text="IoU Threshold:").grid(row=0, column=2, padx=5)
tk.Entry(frame_conf, textvariable=iou_var, width=6).grid(row=0, column=3, padx=5)

# 5. Start Button
tk.Button(root, text="START AUTO ANNOTATION", command=run_process, bg="#2ecc71", fg="white", font=("Arial", 12, "bold")).pack(pady=20, ipadx=10, ipady=5)
tk.Label(root, text="Auto Annotate YOLO/D-FINE/Custom", fg="gray").pack(side="bottom", pady=5)

root.mainloop()