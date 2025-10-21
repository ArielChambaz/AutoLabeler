# AutoLabeler

AutoLabeler is a Python tool for automatic image extraction from a video or folder, and automatic annotation of these images using a YOLO model to generate a ready-to-use dataset (YOLO format).

## Fonctionnalités
- Features
- Extract images from a video (by frame or time interval)
- Copy images from an existing folder
- Automatic annotation with a YOLO model (Ultralytics)
- Generation of YOLO `.txt` files and a `data.yaml` file compatible with Roboflow

## Prérequis
- Requirements
- Python 3.8+
- [Ultralytics YOLO](https://docs.ultralytics.com/) (`pip install ultralytics`)
- OpenCV (`pip install opencv-python`)

## Installation
## Installation
```bash
pip install ultralytics opencv-python
```

## Utilisation
## Usage
1. Place your video or images in the project folder.
2. Run the script:
  ```bash
  python autolabeller.py
  ```
3. Configure the parameters (model, video, images, extraction frequency, thresholds, etc.) directly in the user interface (UI) that appears.

## Exemple de configuration
<img width="597" height="497" alt="image" src="https://github.com/user-attachments/assets/8e87922f-29d3-4a15-9c76-bdcc8582da58" />
## Example configuration
<img width="597" height="497" alt="image" src="https://github.com/user-attachments/assets/8e87922f-29d3-4a15-9c76-bdcc8582da58" />

## Sortie
## Output
The script creates an output folder with the following structure:
```
<video_or_images_dataset_name>/
  images/   # extracted or copied images
  labels/   # YOLO annotations
  data.yaml # detected classes
```
**Once the folders are generated in YOLO format with pre-annotations, all you have to do is upload them to Roboflow to finalize your dataset.**

## Auteurs
## Authors
- Ariel Chambaz

## License
MIT
