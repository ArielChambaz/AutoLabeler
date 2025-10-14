# AutoLabeler

AutoLabeler est un outil Python pour l'extraction automatique d'images depuis une vidéo ou un dossier, et l'annotation automatique de ces images avec un modèle YOLO pour générer un dataset prêt à l'emploi (format YOLO).

## Fonctionnalités
- Extraction d'images depuis une vidéo (par intervalle de frames ou de temps)
- Copie d'images depuis un dossier existant
- Annotation automatique avec un modèle YOLO (Ultralytics)
- Génération des fichiers `.txt` YOLO et d'un fichier `data.yaml` compatible Roboflow

## Prérequis
- Python 3.8+
- [Ultralytics YOLO](https://docs.ultralytics.com/) (`pip install ultralytics`)
- OpenCV (`pip install opencv-python`)

## Installation
```bash
pip install ultralytics opencv-python
```

## Utilisation
1. Placez votre vidéo ou vos images dans le dossier du projet.
2. Lancez le script :
    ```bash
    python autolabeller.py
    ```
3. Configurez les paramètres (modèle, vidéo, images, fréquence d'extraction, seuils, etc.) directement dans l'interface utilisateur (UI) qui s'affiche.

## Exemple de configuration
- **Vidéo, extraction toutes les 2 secondes** :
  ```python
  VIDEO_PATH = "ma_video.mp4"
  TIME_INTERVAL = 2.0
  FRAME_INTERVAL = None
  ```
- **Vidéo, extraction tous les 60 frames** :
  ```python
  VIDEO_PATH = "ma_video.mp4"
  TIME_INTERVAL = None
  FRAME_INTERVAL = 60
  ```
- **Images dans un dossier** :
  ```python
  VIDEO_PATH = None
  IMAGES_DIR = "mon_dossier_images"
  ```

## Sortie
Le script crée un dossier de sortie avec la structure suivante :
```
<nom_video_ou_images_dataset>/
  images/   # images extraites ou copiées
  labels/   # annotations YOLO
  data.yaml # classes détectées
```

## Auteurs
- Ariel Chambaz

## Licence
MIT
