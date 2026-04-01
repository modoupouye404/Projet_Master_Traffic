# app.py - Version sans OpenCV pour l'affichage
import streamlit as st
import numpy as np
from ultralytics import YOLO
import tempfile
import time
import os
from PIL import Image
import io

st.set_page_config(
    page_title="Traffic Detection System",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Détection de Véhicules - IA Prédictive")
st.markdown("---")

# Chargement du modèle
@st.cache_resource
def load_model():
    if os.path.exists('models/best.pt'):
        return YOLO('models/best.pt')
    try:
        return YOLO('yolo11n.pt')
    except:
        return None

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    confidence = st.slider("Seuil de confiance", 0.3, 0.9, 0.5, 0.05)
    skip_frames = st.slider("Accélération", 1, 5, 2)
    st.markdown("---")
    st.info("💡 Modèle YOLOv11")

# Charger le modèle
with st.spinner("🔄 Chargement du modèle..."):
    model = load_model()
    if model:
        st.success("✅ Modèle chargé avec succès!")
    else:
        st.error("❌ Erreur de chargement du modèle")
        st.stop()

# Tracker simple
class SimpleTracker:
    def __init__(self):
        self.next_id = 1
    
    def update(self, detections):
        tracks = []
        for det in detections:
            tracks.append({
                'track_id': self.next_id,
                'bbox': det['bbox'],
                'confidence': det['confidence']
            })
            self.next_id += 1
        return tracks

# Fonction pour dessiner sur l'image avec PIL
def draw_boxes_pil(image_array, tracks):
    """Dessine les bounding boxes avec PIL"""
    # Convertir numpy array en PIL Image
    if image_array.dtype != np.uint8:
        image_array = image_array.astype(np.uint8)
    
    # Si l'image est en BGR, convertir en RGB
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # Vérifier si c'est BGR (OpenCV) ou RGB
        # On suppose que c'est RGB pour l'instant
        pil_img = Image.fromarray(image_array)
    else:
        pil_img = Image.fromarray(image_array)
    
    from PIL import ImageDraw
    draw = ImageDraw.Draw(pil_img)
    
    for track in tracks:
        x1, y1, x2, y2 = track['bbox']
        color = (track['track_id'] * 73 % 255, 
                 track['track_id'] * 137 % 255, 
                 track['track_id'] * 211 % 255)
        
        # Dessiner le rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Dessiner le texte
        draw.text((x1, y1-15), f"ID:{track['track_id']}", fill=color)
    
    return np.array(pil_img)

# Mode d'entrée
mode = st.radio("Source", ["🎥 Vidéo", "📹 Webcam"], horizontal=True)

if mode == "🎥 Vidéo":
    st.subheader("📹 Détection sur Vidéo")
    
    uploaded = st.file_uploader("Choisissez une vidéo", type=['mp4', 'avi', 'mov'])
    
    if uploaded:
        # Sauvegarder
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded.read())
        video_path = tfile.name
        tfile.close()
        
        st.video(video_path)
        
        if st.button("🚀 Démarrer la détection", type="primary"):
            # Importer cv2 UNIQUEMENT ici pour la lecture vidéo
            import cv2
            
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            tracker = SimpleTracker()
            
            # Interface
            col_vid, col_stats = st.columns([2, 1])
            
            with col_vid:
                video_placeholder = st.empty()
                progress = st.progress(0)
            
            with col_stats:
                st.markdown("### 📊 Statistiques")
                vehicle_placeholder = st.empty()
                fps_placeholder = st.empty()
                congestion_placeholder = st.empty()
            
            frame_count = 0
            processed = 0
            start_time = time.time()
            vehicle_history = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames
                if frame_count % skip_frames != 0:
                    continue
                
                # Redimensionner pour performance
                h, w = frame.shape[:2]
                if w > 640:
                    scale = 640 / w
                    new_w = 640
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h))
                
                # Convertir BGR en RGB pour YOLO
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Détection
                results = model(frame_rgb, conf=confidence, verbose=False)
                
                # Extraire détections
                detections = []
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(box.conf[0].cpu().numpy())
                        })
                
                # Tracking
                tracks = tracker.update(detections)
                
                # Visualisation avec PIL
                annotated = draw_boxes_pil(frame_rgb, tracks)
                
                n_vehicles = len(tracks)
                vehicle_history.append(n_vehicles)
                
                # FPS
                elapsed = time.time() - start_time
                fps_val = processed / elapsed if elapsed > 0 else 0
                
                # Congestion
                congestion = min(100, int(n_vehicles * 8))
                if congestion < 30:
                    congestion_status = "Fluide"
                    congestion_color = "green"
                elif congestion < 60:
                    congestion_status = "Modéré"
                    congestion_color = "orange"
                else:
                    congestion_status = "Dense"
                    congestion_color = "red"
                
                # Ajouter du texte avec PIL
                from PIL import ImageDraw, ImageFont
                pil_img = Image.fromarray(annotated)
                draw = ImageDraw.Draw(pil_img)
                
                try:
                    font = ImageFont.load_default()
                except:
                    font = ImageFont.load_default()
                
                draw.text((10, 30), f"FPS: {fps_val:.1f}", fill=(0, 255, 0), font=font)
                draw.text((10, 60), f"Vehicules: {n_vehicles}", fill=(0, 255, 0), font=font)
                draw.text((10, 90), f"Congestion: {congestion}%", fill=(0, 255, 0), font=font)
                
                annotated = np.array(pil_img)
                
                # Affichage
                video_placeholder.image(annotated, use_container_width=True)
                progress.progress(frame_count / total_frames)
                
                # Mise à jour stats
                vehicle_placeholder.metric("🚗 Véhicules", n_vehicles)
                fps_placeholder.metric("⚡ FPS", f"{fps_val:.1f}")
                congestion_placeholder.markdown(f"""
                <div style='background-color:{congestion_color}20; padding:10px; border-radius:10px; text-align:center'>
                    <span>🚦 CONGESTION</span><br>
                    <span style='font-size:2rem; color:{congestion_color}'>{congestion}%</span><br>
                    <span>{congestion_status}</span>
                </div>
                """, unsafe_allow_html=True)
                
                processed += 1
                time.sleep(0.005)
            
            cap.release()
            
            st.success(f"✅ Analyse terminée! {processed} frames traitées")
            
            # Graphique
            if vehicle_history:
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=vehicle_history, mode='lines', 
                                        name='Véhicules', line=dict(color='#2ecc71', width=2)))
                fig.update_layout(
                    title="Évolution du trafic",
                    xaxis_title="Frame",
                    yaxis_title="Nombre de véhicules",
                    height=300,
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            os.unlink(video_path)

else:  # Webcam
    st.subheader("📹 Détection en Temps Réel")
    st.warning("⚠️ La webcam n'est pas supportée sur Streamlit Cloud. Utilisez le mode Vidéo.")

st.markdown("---")
st.caption("🚦 Système de détection de véhicules - YOLOv11")
