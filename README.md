
# 🧠 AutoLabeler

**AutoLabeler** is a Python tool that automatically extracts images from a video or folder and annotates them using a YOLO model to generate a ready-to-use dataset in YOLO format.

---

## ⭐ Features

- Extract images from a video (by frame or time interval)  
- Copy images from an existing folder  
- Automatically annotate images using a YOLO (Ultralytics) model  
- Generate YOLO `.txt` label files and a `data.yaml` configuration file compatible with Roboflow  

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
