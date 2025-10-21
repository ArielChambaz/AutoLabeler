# auto_annotate_yolo.py
# pip install ultralytics opencv-python tkinter
from ultralytics import YOLO
import os, glob, shutil, cv2
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

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
    """Full pipeline: video frame extraction → YOLO prediction → YOLO label generation."""
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

    print("Loading YOLO model...")
    model = YOLO(model_path)
    names = model.model.names

    with open(os.path.join(OUT_ROOT, "data.yaml"), "w") as f:
        f.write("names:\n")
        for cid, cname in names.items():
            f.write(f"  {cid}: {cname}\n")

    for img_path in image_paths:
        fn = os.path.basename(img_path)
        stem, _ = os.path.splitext(fn)
        im = cv2.imread(img_path)
        if im is None:
            print(f"❌ Skipped invalid image: {img_path}")
            continue

        results = model.predict(img_path, conf=conf, iou=iou, verbose=False)[0]
        txt_path = os.path.join(LBL_DIR, f"{stem}.txt")
        with open(txt_path, "w") as f:
            if results.boxes is not None and len(results.boxes) > 0:
                for b in results.boxes:
                    cls = int(b.cls.item())
                    xcn, ycn, wn, hn = b.xywhn[0].tolist()
                    f.write(f"{cls} {xcn:.6f} {ycn:.6f} {wn:.6f} {hn:.6f}\n")
        print(f"OK {fn} -> {txt_path}")

    detected = len([p for p in image_paths if os.path.getsize(os.path.join(LBL_DIR, os.path.splitext(os.path.basename(p))[0] + ".txt")) > 0])
    print("\n✅ Processing complete!")
    print(f"Output folder: {OUT_ROOT}")
    print(f"images/: {len(image_paths)} files | labels/: {detected} annotated")
    messagebox.showinfo("Done", f"Processing complete!\n\n{detected} images annotated in {OUT_ROOT}")


# ------------------------
# GUI
# ------------------------

def browse_video():
    path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv")])
    if path:
        video_path_var.set(path)

def browse_model():
    path = filedialog.askopenfilename(filetypes=[("YOLO weights", "*.pt")])
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
        frame_interval = int(frame_interval_var.get())
        time_interval = float(time_interval_var.get()) if time_interval_var.get() else None
        conf = float(conf_var.get())
        iou = float(iou_var.get())

        if not os.path.exists(model_path):
            messagebox.showerror("Error", "Invalid model path.")
            return

        process_images(model_path, video_path, images_dir, frame_interval, time_interval, conf, iou)

    except Exception as e:
        messagebox.showerror("Error", str(e))


# ------------------------
# TKINTER UI
# ------------------------

root = tk.Tk()
root.title("Auto Annotate YOLO")
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

# UI Layout
tk.Label(root, text="YOLO Model (.pt):").pack(anchor="w", padx=20, pady=(10, 0))
tk.Entry(root, textvariable=model_path_var, width=70).pack(padx=20)
tk.Button(root, text="Browse...", command=browse_model).pack(padx=20, pady=5)

tk.Label(root, text="Video (optional):").pack(anchor="w", padx=20, pady=(10, 0))
tk.Entry(root, textvariable=video_path_var, width=70).pack(padx=20)
tk.Button(root, text="Browse...", command=browse_video).pack(padx=20, pady=5)

tk.Label(root, text="Image folder (if no video):").pack(anchor="w", padx=20, pady=(10, 0))
tk.Entry(root, textvariable=images_dir_var, width=70).pack(padx=20)
tk.Button(root, text="Choose folder...", command=browse_images).pack(padx=20, pady=5)

frame = tk.Frame(root)
frame.pack(pady=10)
tk.Label(frame, text="Frame interval:").grid(row=0, column=0, padx=5)
tk.Entry(frame, textvariable=frame_interval_var, width=6).grid(row=0, column=1, padx=5)
tk.Label(frame, text="Time interval (sec, optional):").grid(row=0, column=2, padx=5)
tk.Entry(frame, textvariable=time_interval_var, width=6).grid(row=0, column=3, padx=5)

frame2 = tk.Frame(root)
frame2.pack(pady=10)
tk.Label(frame2, text="Confidence (CONF):").grid(row=0, column=0, padx=5)
tk.Entry(frame2, textvariable=conf_var, width=6).grid(row=0, column=1, padx=5)
tk.Label(frame2, text="IoU:").grid(row=0, column=2, padx=5)
tk.Entry(frame2, textvariable=iou_var, width=6).grid(row=0, column=3, padx=5)

tk.Button(root, text="Start Annotation", command=run_process, bg="#2ecc71", fg="white", font=("Arial", 12, "bold")).pack(pady=20, ipadx=10, ipady=5)

tk.Label(root, text="Auto Annotate YOLO by Ariel Chambaz", fg="gray").pack(side="bottom", pady=10)

root.mainloop()
