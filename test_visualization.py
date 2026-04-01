# test_visualization.py
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

print("="*60)
print("🚦 TEST AVEC VISUALISATION")
print("="*60)

# Charger le modèle
model = YOLO('models/best.pt')

# Tester sur plusieurs images
test_images = list(Path('data/test/images').glob('*.*'))[:6]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, img_path in enumerate(test_images):
    # Prédiction
    results = model(str(img_path))
    
    # Récupérer l'image annotée
    annotated_img = results[0].plot()
    
    # Afficher
    axes[i].imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    axes[i].axis('off')
    
    n_vehicles = len(results[0].boxes) if results[0].boxes else 0
    axes[i].set_title(f'{img_path.name}\nVéhicules: {n_vehicles}', fontsize=10)

plt.tight_layout()
plt.savefig('test_results.png', dpi=150)
plt.show()
print("✅ Visualisation sauvegardée: test_results.png")