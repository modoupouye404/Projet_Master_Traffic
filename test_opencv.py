# test_opencv.py
from ultralytics import YOLO
import cv2
from pathlib import Path

print("="*60)
print("🚦 TEST DE DÉTECTION AVEC OPENCV")
print("="*60)

# 1. Charger le modèle
model_path = Path('models/best.pt')
if not model_path.exists():
    print(f"❌ Modèle non trouvé: {model_path}")
    print("Veuillez placer best.pt dans le dossier models/")
    exit()

print(f"✅ Modèle trouvé: {model_path}")

try:
    model = YOLO(str(model_path))
    print("✅ Modèle chargé avec succès")
except Exception as e:
    print(f"❌ Erreur chargement: {e}")
    exit()

# 2. Chercher les images de test
test_images = list(Path('data/test/images').glob('*.*'))

if not test_images:
    # Chercher dans d'autres dossiers
    test_images = list(Path('data/valid/images').glob('*.*'))
    if not test_images:
        print("❌ Aucune image trouvée!")
        print("   Vérifiez: data/test/images/ ou data/valid/images/")
        exit()

print(f"\n📷 {len(test_images)} images trouvées")

# 3. Créer le dossier outputs s'il n'existe pas
Path('outputs').mkdir(exist_ok=True)

# 4. Traiter les images
n_images = min(5, len(test_images))
print(f"\n🔍 Traitement de {n_images} images...")
print("-"*60)

for i in range(n_images):
    img_path = test_images[i]
    print(f"\n📸 Image {i+1}: {img_path.name}")
    
    # Détection
    results = model(str(img_path))
    
    # Récupérer l'image annotée
    annotated_img = results[0].plot()
    
    # Compter les véhicules
    n_vehicles = len(results[0].boxes) if results[0].boxes is not None else 0
    print(f"   🚗 Véhicules détectés: {n_vehicles}")
    
    # Afficher les confiances
    if results[0].boxes is not None:
        for j, box in enumerate(results[0].boxes):
            conf = box.conf[0].item()
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            print(f"      Voiture {j+1}: {conf:.1%} confiance")
            print(f"      Position: [{int(x1)},{int(y1)},{int(x2)},{int(y2)}]")
    
    # Sauvegarder l'image annotée
    output_path = f'outputs/result_{i+1}.jpg'
    cv2.imwrite(output_path, annotated_img)
    print(f"   💾 Sauvegardé: {output_path}")

# 5. Résumé
print("\n" + "="*60)
print("📊 RÉSUMÉ")
print("="*60)
print(f"✅ {n_images} images traitées avec succès!")
print(f"📁 Résultats dans: outputs/")
print("\nPour visualiser les résultats:")
print("   - Ouvrez le dossier 'outputs'")
print("   - Double-cliquez sur les fichiers result_*.jpg")

print("\n" + "="*60)