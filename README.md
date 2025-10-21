
# 🧠 AutoLabeler

**AutoLabeler** is a Python tool that automatically extracts images from a video or folder and annotates them using a YOLO model to generate a ready-to-use dataset in YOLO format.

---


## ⭐ Features

AutoLabeler lets you save time and effort when managing datasets by using your already trained YOLO model to automatically generate annotations for new images or video frames.

🎞 Extract images from a video (by frame or time interval)
Turn any video into a set of images automatically. You can choose how often to extract frames (every n frames or every n seconds) to build a rich dataset from your footage.

🖼 Copy images from an existing folder
If your images are already stored somewhere, AutoLabeler can directly use them — no need for a video source.

🤖 Automatically annotate using your trained YOLO model
Instead of redrawing boxes for every image, AutoLabeler uses your pretrained YOLO model to detect objects and automatically create annotation files.
This is especially useful when your model already performs well — it allows you to skip manual reannotation for objects it recognizes reliably, drastically reducing dataset labeling time.

📂 Generate YOLO .txt label files and a data.yaml for Roboflow
AutoLabeler outputs a clean YOLO-format dataset (images/, labels/, and data.yaml) ready to upload directly to Roboflow.
You can then visually check and fine-tune the automatically generated annotations if needed.

💡 In short: AutoLabeler uses your YOLO model as a “smart annotator.”
You just provide videos or images, and it automatically produces labeled data — perfect for dataset expansion, model fine-tuning, or quick reannotation without wasting time on what your model already masters.

---

## ⚙️ Requirements

- Python 3.8+  
- [Ultralytics YOLO](https://docs.ultralytics.com/)
  ```bash
  pip install ultralytics
  ```
- OpenCV
  ```bash
  pip install opencv-python
  ```

---

🛠️ **Installation**

```bash
pip install ultralytics opencv-python
```

---

🚀 **Usage**

1. Place your video or images in the project folder.
2. Run the script:
    ```bash
    python autolabeller.py
    ```
3. Configure the parameters (model, video source, image folder, extraction frequency, confidence thresholds, etc.) directly in the user interface (UI) that appears.

---

🧩 **Example Configuration**

Example UI

<img width="597" height="502" alt="Screenshot from 2025-10-21 10-09-03" src="https://github.com/user-attachments/assets/b2ce0aa2-8c30-4eda-898c-b57fe46fd448" />

---

📁 **Output**

The script generates an output folder with the following structure:

```
<video_or_images_dataset_name>/
  images/   # extracted or copied images
  labels/   # YOLO annotations
  data.yaml # detected classes
```

Once the YOLO-format folders with pre-annotations are generated, simply upload them to Roboflow to finalize your dataset.

---

👤 **Author**

Ariel Chambaz

---

📜 **License**

MIT License
