# final_check.py
from pathlib import Path
import yaml

print("="*70)
print("✅ VÉRIFICATION FINALE AVANT ENTRAÎNEMENT")
print("="*70)

# 1. Vérifier que data.yaml est au bon endroit
data_yaml = Path('data/data.yaml')
if not data_yaml.exists():
    print("❌ data.yaml non trouvé dans data/")
    print("   Vérifiez que votre dataset est dans le dossier 'data/'")
    exit()

print("✅ 1. data.yaml trouvé")

# 2. Lire et vérifier la configuration
with open(data_yaml, 'r') as f:
    config = yaml.safe_load(f)

print(f"\n📋 Configuration:")
print(f"   Train: {config['train']}")
print(f"   Val: {config['val']}")
print(f"   Test: {config['test']}")
print(f"   Classes: {config['names']}")

# 3. Vérifier les dossiers d'images
print("\n📷 Vérification des images:")

total_images = 0
for split in ['train', 'val', 'test']:
    images_path = Path(f'data/{config[split]}')
    if images_path.exists():
        n_images = len(list(images_path.glob('*.*')))
        total_images += n_images
        print(f"   ✅ {split.upper()}: {n_images} images")
        
        # Afficher un exemple
        if n_images > 0:
            example = list(images_path.glob('*.*'))[0]
            print(f"      Exemple: {example.name}")
    else:
        print(f"   ❌ {split.upper()}: Dossier non trouvé: {images_path}")

# 4. Vérifier les labels
print("\n🏷️ Vérification des annotations:")
for split in ['train', 'val', 'test']:
    labels_path = Path(f'data/{split}/labels')
    if labels_path.exists():
        n_labels = len(list(labels_path.glob('*.txt')))
        print(f"   ✅ {split.upper()}: {n_labels} annotations")
        
        # Lire un exemple d'annotation
        if n_labels > 0:
            example = list(labels_path.glob('*.txt'))[0]
            with open(example, 'r') as f:
                content = f.readline().strip()
                print(f"      Exemple: {content}")
    else:
        print(f"   ❌ {split.upper()}: Dossier labels non trouvé")

# 5. Vérifier les modèles disponibles
print("\n🤖 Vérification YOLOv11:")
try:
    from ultralytics import YOLO
    print("   ✅ Ultralytics installé")
    
    # Vérifier les modèles disponibles
    models = ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt']
    for model_name in models:
        if Path(model_name).exists():
            print(f"   ✅ {model_name} téléchargé")
        else:
            print(f"   ⚠️ {model_name} sera téléchargé à l'entraînement")
            
except ImportError:
    print("   ❌ Ultralytics non installé!")
    print("   Exécutez: pip install ultralytics")

# 6. Vérifier PyTorch et GPU
print("\n💻 Vérification matérielle:")
try:
    import torch
    print(f"   ✅ PyTorch: {torch.__version__}")
    print(f"   ✅ CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✅ GPU: {torch.cuda.get_device_name()}")
    else:
        print("   ⚠️ GPU non disponible, entraînement sur CPU (plus lent)")
except ImportError:
    print("   ❌ PyTorch non installé!")

# 7. Résumé final
print("\n" + "="*70)
print("📊 RÉSUMÉ")
print("="*70)

if total_images > 0:
    print(f"✅ Total images: {total_images}")
    print("✅ Dataset prêt pour l'entraînement!")
    
    # Recommandations
    print("\n🚀 Recommandations d'entraînement:")
    print(f"   Modèle recommandé: yolo11m.pt")
    print(f"   Epochs: 100")
    print(f"   Batch size: 16 (ajustez selon votre GPU)")
    print(f"   Image size: 640")
    
    print("\n📝 Commande pour lancer l'entraînement:")
    print("   python train_yolov11.py")
    
else:
    print("❌ Aucune image trouvée! Vérifiez la structure de vos dossiers.")

print("\n" + "="*70)