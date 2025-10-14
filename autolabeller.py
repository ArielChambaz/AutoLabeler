# auto_annotate_yolo.py
# pip install ultralytics opencv-python
from ultralytics import YOLO
import os, glob, shutil
import cv2
from datetime import datetime

# ---------- Config ----------
MODEL_PATH = "/home/achambaz/sananga/stockfit/Stockfit/models/q-inventory.v50i.yolov11.pt"     # ton modèle YOLO
IMAGES_DIR = "images"              # dossier d'images en entrée (utilisé si pas de vidéo)

# Configuration vidéo - Modifie ces valeurs selon tes besoins
VIDEO_PATH = "/home/achambaz/sananga/stockfit/AutoLabeler/IMG_0322.mp4"           # chemin vers la vidéo à traiter (None pour traiter les images du dossier)
FRAME_INTERVAL = 60                # extraire une image tous les X frames
TIME_INTERVAL = None               # ou extraire une image toutes les X secondes (priorité sur FRAME_INTERVAL si défini)
                                   # Exemple: TIME_INTERVAL = 2.0 pour une image toutes les 2 secondes

# Paramètres de détection
CONF = 0.55                        # seuil de confiance
IOU = 0.45                         # NMS IoU

# Exemples de configuration :
# 1. Vidéo avec extraction toutes les 2 secondes :
#    VIDEO_PATH = "ma_video.mp4"
#    TIME_INTERVAL = 2.0
#    FRAME_INTERVAL = None  # ignoré
#
# 2. Vidéo avec extraction tous les 60 frames :
#    VIDEO_PATH = "ma_video.mp4"
#    TIME_INTERVAL = None
#    FRAME_INTERVAL = 60
#
# 3. Images existantes dans un dossier :
#    VIDEO_PATH = None
#    IMAGES_DIR = "mon_dossier_images"
# ----------------------------

def extract_frames_from_video(video_path, output_dir, frame_interval=30, time_interval=None):
    """
    Extrait des frames d'une vidéo selon un intervalle donné
    
    Args:
        video_path: chemin vers la vidéo
        output_dir: dossier de sortie pour les images
        frame_interval: extraire une image tous les X frames
        time_interval: extraire une image toutes les X secondes (priorité sur frame_interval)
    
    Returns:
        Liste des chemins des images extraites
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la vidéo: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Vidéo: {os.path.basename(video_path)}")
    print(f"FPS: {fps}, Total frames: {total_frames}")
    
    # Calculer l'intervalle en frames
    if time_interval is not None:
        interval_frames = int(fps * time_interval)
        print(f"Extraction: 1 image toutes les {time_interval} secondes ({interval_frames} frames)")
    else:
        interval_frames = frame_interval
        print(f"Extraction: 1 image tous les {frame_interval} frames")
    
    extracted_paths = []
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % interval_frames == 0:
            # Nom du fichier avec numéro de frame
            timestamp = frame_count / fps
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            frame_filename = f"frame_{frame_count:06d}_{minutes:02d}m{seconds:02d}s.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            
            cv2.imwrite(frame_path, frame)
            extracted_paths.append(frame_path)
            extracted_count += 1
            
            if extracted_count % 10 == 0:
                print(f"Extrait {extracted_count} frames...")
        
        frame_count += 1
    
    cap.release()
    print(f"Extraction terminée: {extracted_count} images extraites")
    return extracted_paths

def create_output_directory(video_path=None):
    """
    Crée un dossier de sortie avec le nom de la vidéo (sans timestamp)
    
    Returns:
        Chemin vers le dossier de sortie créé
    """
    if video_path:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_root = video_name  # Directement le nom de la vidéo
    else:
        output_root = "images_dataset"  # Pour le mode images
    
    os.makedirs(output_root, exist_ok=True)
    
    return output_root

# Détermine le mode de fonctionnement
if VIDEO_PATH and os.path.exists(VIDEO_PATH):
    print("Mode vidéo activé")
    # Crée le dossier de sortie avec nom de vidéo + timestamp
    OUT_ROOT = create_output_directory(VIDEO_PATH)
    IMG_DIR = os.path.join(OUT_ROOT, "images")
    os.makedirs(IMG_DIR, exist_ok=True)
    
    # Extrait les frames de la vidéo
    print("Extraction des frames de la vidéo...")
    image_paths = extract_frames_from_video(VIDEO_PATH, IMG_DIR, FRAME_INTERVAL, TIME_INTERVAL)
else:
    print("Mode images activé")
    # Mode original : traitement d'images existantes
    if VIDEO_PATH:
        print(f"Attention: Vidéo spécifiée ({VIDEO_PATH}) mais fichier non trouvé")
    
    OUT_ROOT = create_output_directory()
    IMG_DIR = os.path.join(OUT_ROOT, "images")
    os.makedirs(IMG_DIR, exist_ok=True)
    
    # récupère images existantes
    ex = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff")
    image_paths = []
    for e in ex:
        image_paths += glob.glob(os.path.join(IMAGES_DIR, e))
    
    # copie les images dans le dossier de sortie
    copied_paths = []
    for img_path in image_paths:
        fn = os.path.basename(img_path)
        dst_img = os.path.join(IMG_DIR, fn)
        shutil.copy2(img_path, dst_img)
        copied_paths.append(dst_img)
    image_paths = copied_paths

print(f"{len(image_paths)} images à traiter")

LBL_DIR = os.path.join(OUT_ROOT, "labels")
os.makedirs(LBL_DIR, exist_ok=True)

# charge modèle
model = YOLO(MODEL_PATH)
names = model.model.names  # dict {id: "classname"}

# crée data.yaml pour Roboflow
with open(os.path.join(OUT_ROOT, "data.yaml"), "w") as f:
    f.write("names:\n")
    for cid, cname in names.items():
        f.write(f"  {cid}: {cname}\n")

# traitement des images
for img_path in image_paths:
    fn = os.path.basename(img_path)
    stem, _ = os.path.splitext(fn)

    # lit pour récupérer W,H
    im = cv2.imread(img_path)
    if im is None:
        print(f"Skip invalide: {img_path}")
        continue
    h, w = im.shape[:2]

    # prédiction
    results = model.predict(
        img_path,
        conf=CONF,
        iou=IOU,
        verbose=False
    )[0]

    # écrit le .txt YOLO
    txt_path = os.path.join(LBL_DIR, f"{stem}.txt")
    with open(txt_path, "w") as f:
        if results.boxes is not None and len(results.boxes) > 0:
            # xywhn est déjà normalisé [0..1]
            for b in results.boxes:
                cls = int(b.cls.item())
                xcn, ycn, wn, hn = b.xywhn[0].tolist()
                f.write(f"{cls} {xcn:.6f} {ycn:.6f} {wn:.6f} {hn:.6f}\n")

    print(f"OK {fn} -> {txt_path}")

print("\nTraitement terminé !")
print("Dossier de sortie:", OUT_ROOT)
print("Structure créée:")
print(f"  {OUT_ROOT}/")
print(f"    images/ ({len(image_paths)} images)")
print(f"    labels/ ({len(image_paths)} fichiers .txt)")
print(f"    data.yaml")
print(f"\nDataset prêt ! Le modèle a trouvé des objets dans {len([p for p in image_paths if os.path.getsize(os.path.join(LBL_DIR, os.path.splitext(os.path.basename(p))[0] + '.txt')) > 0])} images")
print(f"Pour Roboflow, zippe le dossier:")
print(f"zip -r {OUT_ROOT}.zip {OUT_ROOT}")
