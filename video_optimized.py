# video_super_fast.py
from ultralytics import YOLO
import cv2
import time
from pathlib import Path

print("="*60)
print("⚡ DÉTECTION VIDÉO - SUPER RAPIDE")
print("="*60)

# Charger le modèle
model = YOLO('models/best.pt')

# Ouvrir la vidéo
video_path = input("📹 Chemin de la vidéo (ou 'webcam'): ").strip()

if video_path.lower() == 'webcam':
    cap = cv2.VideoCapture(0)
    video_name = "webcam"
else:
    cap = cv2.VideoCapture(video_path)
    video_name = Path(video_path).stem

if not cap.isOpened():
    print("❌ Impossible d'ouvrir la vidéo")
    exit()

# Paramètres EXTREME pour vitesse maximale
TARGET_WIDTH = 160      # Très petite résolution
TARGET_HEIGHT = 120
PROCESS_EVERY_N_FRAMES = 10  # Traiter 1 frame sur 10
CONFIDENCE = 0.3        # Seuil plus bas

print(f"\n⚡ OPTIMISATIONS EXTREMES:")
print(f"   Résolution: {TARGET_WIDTH}x{TARGET_HEIGHT}")
print(f"   Traitement: 1 frame sur {PROCESS_EVERY_N_FRAMES}")
print(f"   Seuil: {CONFIDENCE}")

# Sortie
out = cv2.VideoWriter(
    f'outputs/video_super_fast_{video_name}.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    15,  # FPS fixe réduit
    (TARGET_WIDTH, TARGET_HEIGHT)
)

print("\n🔴 Traitement...")
frame_count = 0
processed = 0
start = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Traiter seulement certaines frames
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        # Réduire taille
        small = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        
        # Détection
        results = model(small, conf=CONFIDENCE, verbose=False)
        annotated = results[0].plot()
        
        n = len(results[0].boxes) if results[0].boxes else 0
        cv2.putText(annotated, f'V: {n}', (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        out.write(annotated)
        processed += 1
        
        if processed % 50 == 0:
            print(f"   Frames: {processed}")
    else:
        # Répéter la dernière frame
        if 'last_annotated' in locals():
            out.write(last_annotated)
    
    last_annotated = annotated if 'annotated' in locals() else None

cap.release()
out.release()

elapsed = time.time() - start
print(f"\n✅ Terminé! {processed} frames en {elapsed:.1f}s")
print(f"📁 output: outputs/video_super_fast_{video_name}.mp4")