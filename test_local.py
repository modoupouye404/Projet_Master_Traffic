# test_local.py - Version corrigée
from ultralytics import YOLO
import cv2
import os
from pathlib import Path

print("="*60)
print("🔍 RECHERCHE DU MODÈLE")
print("="*60)

# Vérifier si le dossier models existe
if not os.path.exists('models'):
    os.makedirs('models')
    print("✅ Dossier 'models' créé")

# Chercher le modèle best.pt
possible_paths = [
    'models/best.pt',
    'best.pt',
    'models/yolov11_trained/weights/best.pt',
    'yolo_trained/traffic_model/weights/best.pt',
    'C:/Users/modou/Downloads/best.pt'
]

model_path = None
for path in possible_paths:
    if os.path.exists(path):
        model_path = path
        print(f"✅ Modèle trouvé: {path}")
        break

if model_path is None:
    print("❌ Modèle non trouvé!")
    print("\nVeuillez placer le fichier best.pt dans:")
    print("   C:\\Users\\modou\\Downloads\\Projet_Master_Traffic\\models\\best.pt")
    exit()

# Charger le modèle
print(f"\n📥 Chargement du modèle: {model_path}")
model = YOLO(model_path)

# Tester sur une image
test_image_path = 'data/test/images/'
print(f"\n🔍 Recherche d'images de test dans: {test_image_path}")

# Chercher une image de test
import glob
test_images = glob.glob(f"{test_image_path}/*.*")

if test_images:
    image_path = test_images[0]
    print(f"📷 Image trouvée: {image_path}")
    
    # Prédiction
    results = model(image_path)
    
    # Afficher le résultat
    print(f"\n✅ Détections: {len(results[0].boxes) if results[0].boxes else 0} véhicules")
    
    # Sauvegarder
    results[0].save('result.jpg')
    print("📁 Résultat sauvegardé: result.jpg")
    
    # Afficher avec OpenCV
    img = cv2.imread(image_path)
    cv2.imshow('Image originale', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("❌ Aucune image trouvée dans data/test/images/")
    print("   Veuillez vérifier le dossier")

print("\n✅ Test terminé!")