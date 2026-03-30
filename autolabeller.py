# auto_annotate_yolo.py
# pip install ultralytics opencv-python
import os, glob, shutil, cv2
import tkinter as tk
from tkinter import filedialog, messagebox

from ultralytics import YOLO


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
    print(f"Extraction complete: {extracted_count} frames saved")
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
    """Full pipeline: video frame extraction → YOLO prediction → Label generation."""

    # Validate model path
    if not model_path or model_path.strip() == "":
        messagebox.showerror("Error", "Please select a YOLO model (.pt file).")
        return

    ext = os.path.splitext(model_path)[-1].lower()
    if ext != ".pt":
        messagebox.showerror("Error", f"Unsupported model format: '{ext}'\nOnly YOLO .pt files are supported.")
        return

    print(f"Model: {model_path}")

    # 1. Image preparation
    if video_path and os.path.exists(video_path):
        print("\nVideo mode enabled")
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

    # 2. Load YOLO model
    print(f"Loading YOLO model: {model_path}")
    try:
        model = YOLO(model_path)
        names = model.names
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load YOLO model:\n{e}")
        return

    # 3. Write data.yaml
    with open(os.path.join(OUT_ROOT, "data.yaml"), "w") as f:
        f.write("names:\n")
        for cid, cname in names.items():
            f.write(f"  {cid}: {cname}\n")

    # 4. Run inference and write labels
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
        print(f"OK {fn} -> {txt_path}")

    detected = len([
        p for p in image_paths
        if os.path.exists(os.path.join(LBL_DIR, os.path.splitext(os.path.basename(p))[0] + ".txt"))
        and os.path.getsize(os.path.join(LBL_DIR, os.path.splitext(os.path.basename(p))[0] + ".txt")) > 0
    ])

    print("\nProcessing complete!")
    messagebox.showinfo("Done", f"Processing complete!\n{detected} images annotated in '{OUT_ROOT}'")


# ------------------------
# GUI FUNCTIONS
# ------------------------

def browse_video():
    path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv")])
    if path:
        video_path_var.set(path)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def browse_model():
    path = filedialog.askopenfilename(
        title="Select YOLO model weights (.pt)",
        initialdir=SCRIPT_DIR,
        filetypes=[("YOLO model", "*.pt"), ("All files", "*.*")]
    )
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

        process_images(model_path, video_path, images_dir, frame_interval, time_interval, conf, iou)

    except Exception as e:
        messagebox.showerror("Error", str(e))


# ------------------------
# TKINTER UI SETUP
# ------------------------

BG      = "#f0f0f0"
FG      = "#1a1a1a"
ENTRY_BG = "#ffffff"
BTN_BG  = "#e0e0e0"

root = tk.Tk()
root.title("Auto Annotate (YOLO .pt)")
root.geometry("600x460")
root.resizable(False, False)
root.configure(bg=BG)

# Force light palette on all widgets
root.tk_setPalette(background=BG, foreground=FG,
                   activeBackground=BTN_BG, activeForeground=FG,
                   highlightBackground=BG)

def lbl(parent, text, **kw):
    return tk.Label(parent, text=text, bg=BG, fg=FG, **kw)

def entry(parent, textvariable, **kw):
    return tk.Entry(parent, textvariable=textvariable, bg=ENTRY_BG, fg=FG,
                    insertbackground=FG, relief="solid", bd=1, **kw)

def btn(parent, text, command, **kw):
    return tk.Button(parent, text=text, command=command,
                     bg=BTN_BG, fg=FG, activebackground="#cccccc",
                     relief="raised", bd=1, **kw)

def frame(parent, **kw):
    return tk.Frame(parent, bg=BG, **kw)

# Variables
model_path_var = tk.StringVar()
video_path_var = tk.StringVar()
images_dir_var = tk.StringVar()
frame_interval_var = tk.StringVar(value="60")
time_interval_var = tk.StringVar(value="")
conf_var = tk.StringVar(value="0.55")
iou_var = tk.StringVar(value="0.45")

# 1. Model
lbl(root, "1. Model Path (YOLO .pt):").pack(anchor="w", padx=20, pady=(10, 0))
entry_frame = frame(root)
entry_frame.pack(fill="x", padx=20)
entry(entry_frame, model_path_var).pack(side="left", fill="x", expand=True)
btn(entry_frame, "Browse...", browse_model).pack(side="left", padx=5)

# 2. Video
lbl(root, "2. Video (optional):").pack(anchor="w", padx=20, pady=(10, 0))
entry_frame_vid = frame(root)
entry_frame_vid.pack(fill="x", padx=20)
entry(entry_frame_vid, video_path_var).pack(side="left", fill="x", expand=True)
btn(entry_frame_vid, "Browse...", browse_video).pack(side="left", padx=5)

# 3. Images folder
lbl(root, "3. Image folder (if no video):").pack(anchor="w", padx=20, pady=(10, 0))
entry_frame_img = frame(root)
entry_frame_img.pack(fill="x", padx=20)
entry(entry_frame_img, images_dir_var).pack(side="left", fill="x", expand=True)
btn(entry_frame_img, "Choose folder...", browse_images).pack(side="left", padx=5)

# 4. Parameters
frame_params = frame(root)
frame_params.pack(pady=10)
lbl(frame_params, "Frame interval:").grid(row=0, column=0, padx=5)
entry(frame_params, frame_interval_var, width=6).grid(row=0, column=1, padx=5)
lbl(frame_params, "Time interval (s):").grid(row=0, column=2, padx=5)
entry(frame_params, time_interval_var, width=6).grid(row=0, column=3, padx=5)

frame_conf = frame(root)
frame_conf.pack(pady=5)
lbl(frame_conf, "Confidence (CONF):").grid(row=0, column=0, padx=5)
entry(frame_conf, conf_var, width=6).grid(row=0, column=1, padx=5)
lbl(frame_conf, "IoU Threshold:").grid(row=0, column=3, padx=5)
entry(frame_conf, iou_var, width=6).grid(row=0, column=4, padx=5)

# 5. Start
tk.Button(root, text="START AUTO ANNOTATION", command=run_process,
          bg="#2ecc71", fg="white", activebackground="#27ae60",
          font=("Arial", 12, "bold"), relief="raised", bd=1).pack(pady=20, ipadx=10, ipady=5)
tk.Label(root, text="Auto Annotate YOLO (.pt only)", bg=BG, fg="#888888").pack(side="bottom", pady=5)

root.mainloop()
