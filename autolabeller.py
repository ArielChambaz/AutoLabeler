# auto_annotate_yolo.py
import os, glob, shutil, cv2
import tkinter as tk
from tkinter import filedialog, messagebox

# --- IMPORTS ---
from ultralytics import YOLO
import torch
# CORRECTION MAJEURE ICI : Utilisation de DFineForObjectDetection
from transformers import DFineForObjectDetection, AutoImageProcessor
from PIL import Image

# Modèle par défaut (si laissé vide)
DEFAULT_HF_MODEL = "ustc-community/dfine-xlarge-obj365"

# ------------------------
# FONCTIONS PRINCIPALES
# ------------------------

def extract_frames_from_video(video_path, output_dir, frame_interval=30, time_interval=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {os.path.basename(video_path)} | FPS: {fps} | Total frames: {total_frames}")

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
    if video_path:
        name = os.path.splitext(os.path.basename(video_path))[0]
        out_root = name
    else:
        out_root = "images_dataset"
    os.makedirs(out_root, exist_ok=True)
    return out_root


def process_images(model_path, model_type, video_path, images_dir, frame_interval, time_interval, conf, iou):
    # 1. Préparation des images
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
    # BRANCHE 1 : YOLO (Ultralytics)
    # ==========================================
    if model_type == "YOLO":
        print(f"🚀 Loading YOLO model from file: {model_path}")
        try:
            model = YOLO(model_path)
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
    # BRANCHE 2 : D-FINE (Hugging Face / Official Docs)
    # ==========================================
    elif model_type == "DFINE":
        
        # --- LOGIQUE DE CHARGEMENT ---
        if not model_path or model_path.strip() == "":
            print(f"🌐 No local file selected. Downloading/Loading from Hugging Face: {DEFAULT_HF_MODEL}")
            target_model = DEFAULT_HF_MODEL
        else:
            print(f"📂 Local file selected. Loading configuration from folder parent...")
            # Si c'est un fichier (.pth, .safetensors), on prend le dossier parent
            if os.path.isfile(model_path):
                target_model = os.path.dirname(model_path)
            else:
                target_model = model_path
            
            # Vérification de sécurité
            if not os.path.exists(os.path.join(target_model, "config.json")):
                 messagebox.showerror("Config Error", 
                    f"Could not find 'config.json' in:\n{target_model}\n\n"
                    "Ensure your .pth/.safetensors file is in the same folder as config.json")
                 return
        
        try:
            # --- UTILISATION DE LA CLASSE OFFICIELLE ---
            print(f"Loading DFineForObjectDetection from {target_model}")
            image_processor = AutoImageProcessor.from_pretrained(target_model)
            model = DFineForObjectDetection.from_pretrained(target_model)
            names = model.config.id2label
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load D-FINE model.\nSource: {target_model}\n\nError: {e}")
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

            # --- POST-PROCESSING OFFICIEL ---
            # D-FINE attend 'target_sizes' pour redimensionner les boites
            target_sizes = torch.tensor([(height, width)]) # Format (H, W)
            
            results = image_processor.post_process_object_detection(
                outputs, 
                threshold=conf, 
                target_sizes=target_sizes
            )[0]

            txt_path = os.path.join(LBL_DIR, f"{stem}.txt")
            with open(txt_path, "w") as f:
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    # Pas besoin de filtrer par score ici si threshold=conf est déjà passé à post_process
                    # Mais on garde la sécurité
                    if score >= conf:
                        xmin, ymin, xmax, ymax = box.tolist()
                        
                        # Conversion vers format YOLO normalisé (x_center, y_center, w, h)
                        x_center = ((xmin + xmax) / 2) / width
                        y_center = ((ymin + ymax) / 2) / height
                        box_width = (xmax - xmin) / width
                        box_height = (ymax - ymin) / height
                        
                        # Clamp pour rester entre 0 et 1 (sécurité)
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        box_width = max(0, min(1, box_width))
                        box_height = max(0, min(1, box_height))

                        cls = label.item()
                        f.write(f"{cls} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
            print(f"OK (D-FINE) {fn} -> {txt_path}")

    # ==========================================
    # FIN
    # ==========================================
    detected = len([p for p in image_paths if os.path.exists(os.path.join(LBL_DIR, os.path.splitext(os.path.basename(p))[0] + ".txt")) and os.path.getsize(os.path.join(LBL_DIR, os.path.splitext(os.path.basename(p))[0] + ".txt")) > 0])
    
    print("\n✅ Processing complete!")
    messagebox.showinfo("Done", f"Processing complete!\nMode: {model_type}\n{detected} images annotated in {OUT_ROOT}")


# ------------------------
# GUI FUNCTIONS
# ------------------------

def update_browse_behavior():
    """Efface le chemin si on change de mode."""
    model_path_var.set("")

def browse_video():
    path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv")])
    if path:
        video_path_var.set(path)

def browse_model():
    mode = model_type_var.get()
    
    if mode == "YOLO":
        title = "Select YOLO Weights (.pt)"
        filetypes = [("YOLO weights", "*.pt *.pth")]
    else:
        title = "Select D-FINE Weights (Optional - Cancel to use Online)"
        filetypes = [("D-FINE weights", "*.pth *.pt *.safetensors")]
        
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
        model_type = model_type_var.get()
        video_path = video_path_var.get() or None
        images_dir = images_dir_var.get() or "images"
        frame_interval = int(frame_interval_var.get())
        time_interval = float(time_interval_var.get()) if time_interval_var.get() else None
        conf = float(conf_var.get())
        iou = float(iou_var.get())

        if model_type == "YOLO" and not os.path.exists(model_path):
            messagebox.showerror("Error", "YOLO requires a local .pt file path.")
            return
        elif model_type == "DFINE" and model_path and not os.path.exists(model_path):
            messagebox.showerror("Error", "The provided D-FINE path does not exist.\nClear the field to download from Hugging Face automatically.")
            return

        process_images(model_path, model_type, video_path, images_dir, frame_interval, time_interval, conf, iou)

    except Exception as e:
        messagebox.showerror("Error", str(e))


# ------------------------
# TKINTER UI SETUP
# ------------------------

root = tk.Tk()
root.title("Auto Annotate (YOLO & D-FINE)")
root.geometry("600x580")
root.resizable(False, False)

# Variables
model_type_var = tk.StringVar(value="YOLO")
model_path_var = tk.StringVar()
video_path_var = tk.StringVar()
images_dir_var = tk.StringVar()
frame_interval_var = tk.StringVar(value="60")
time_interval_var = tk.StringVar(value="")
conf_var = tk.StringVar(value="0.55")
iou_var = tk.StringVar(value="0.45")

# 1. Sélection du Mode
type_frame = tk.LabelFrame(root, text="1. Select Model Type", padx=10, pady=5)
type_frame.pack(fill="x", padx=20, pady=10)

tk.Radiobutton(type_frame, text="Ultralytics YOLO (.pt)", variable=model_type_var, value="YOLO", command=update_browse_behavior).pack(side="left", padx=20)
tk.Radiobutton(type_frame, text="Hugging Face D-FINE (Online/Local)", variable=model_type_var, value="DFINE", command=update_browse_behavior).pack(side="left", padx=20)

# 2. Sélection du Modèle
tk.Label(root, text="Model Path (Leave empty for D-FINE online download):").pack(anchor="w", padx=20, pady=(5, 0))
entry_frame = tk.Frame(root)
entry_frame.pack(fill="x", padx=20)
tk.Entry(entry_frame, textvariable=model_path_var).pack(side="left", fill="x", expand=True)
tk.Button(entry_frame, text="Browse...", command=browse_model).pack(side="left", padx=5)

# 3. Vidéo ou Images
tk.Label(root, text="Video (optional):").pack(anchor="w", padx=20, pady=(10, 0))
entry_frame_vid = tk.Frame(root)
entry_frame_vid.pack(fill="x", padx=20)
tk.Entry(entry_frame_vid, textvariable=video_path_var).pack(side="left", fill="x", expand=True)
tk.Button(entry_frame_vid, text="Browse...", command=browse_video).pack(side="left", padx=5)

tk.Label(root, text="Image folder (if no video):").pack(anchor="w", padx=20, pady=(10, 0))
entry_frame_img = tk.Frame(root)
entry_frame_img.pack(fill="x", padx=20)
tk.Entry(entry_frame_img, textvariable=images_dir_var).pack(side="left", fill="x", expand=True)
tk.Button(entry_frame_img, text="Choose folder...", command=browse_images).pack(side="left", padx=5)

# 4. Paramètres
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

# 5. Bouton Start
tk.Button(root, text="START AUTO ANNOTATION", command=run_process, bg="#2ecc71", fg="white", font=("Arial", 12, "bold")).pack(pady=20, ipadx=10, ipady=5)
tk.Label(root, text="Auto Annotate YOLO/D-FINE", fg="gray").pack(side="bottom", pady=5)

root.mainloop()