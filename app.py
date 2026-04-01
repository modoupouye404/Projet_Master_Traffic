# app.py - Version avec affichage optimisé
import streamlit as st
import numpy as np
import tempfile
import time
import os

st.set_page_config(
    page_title="Traffic Detection",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Détection de Véhicules")
st.markdown("---")

# Importation
try:
    from ultralytics import YOLO
    import cv2
    YOLO_AVAILABLE = True
except Exception as e:
    st.error(f"Erreur d'import: {e}")
    st.stop()

# Chargement du modèle
@st.cache_resource
def load_model():
    try:
        model = YOLO('yolo11n.pt')
        return model
    except Exception as e:
        st.error(f"Erreur chargement modèle: {e}")
        return None

# Sidebar
with st.sidebar:
    st.header("⚙️ Paramètres")
    confidence = st.slider("Seuil de confiance", 0.3, 0.9, 0.5, 0.05)
    display_freq = st.slider("Fréquence d'affichage", 1, 10, 3, 
                              help="Plus la valeur est élevée, plus l'affichage est fluide")
    st.info("💡 Modèle YOLOv11 (yolo11n.pt)")

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

# Mode Vidéo
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
        
        frame_count = 0
        processed = 0
        display_counter = 0
        start_time = time.time()
        vehicle_history = []
        
        # Stocker la dernière image affichée
        last_displayed_frame = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Traiter 1 frame sur 2 pour accélérer
            if frame_count % 2 != 0:
                continue
            
            # Redimensionner
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640 / w
                new_w = 640
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h))
            
            # Détection
            results = model(frame, conf=confidence, verbose=False)
            
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
            
            # Visualisation
            annotated = frame.copy()
            for track in tracks:
                x1, y1, x2, y2 = track['bbox']
                color = (track['track_id'] * 73 % 255, 
                         track['track_id'] * 137 % 255, 
                         track['track_id'] * 211 % 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{track['track_id']}", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            n_vehicles = len(tracks)
            vehicle_history.append(n_vehicles)
            
            # FPS
            elapsed = time.time() - start_time
            fps_val = processed / elapsed if elapsed > 0 else 0
            
            # Texte
            cv2.putText(annotated, f"FPS: {fps_val:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Vehicules: {n_vehicles}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Mise à jour des stats (toujours mise à jour)
            vehicle_placeholder.metric("🚗 Véhicules", n_vehicles)
            fps_placeholder.metric("⚡ FPS", f"{fps_val:.1f}")
            
            # Affichage uniquement toutes les X frames pour éviter l'erreur DOM
            display_counter += 1
            if display_counter >= display_freq:
                try:
                    # Convertir BGR en RGB pour Streamlit
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(annotated_rgb, use_container_width=True)
                    last_displayed_frame = annotated_rgb
                    display_counter = 0
                except Exception as e:
                    # En cas d'erreur, réutiliser la dernière frame
                    if last_displayed_frame is not None:
                        video_placeholder.image(last_displayed_frame, use_container_width=True)
            
            progress.progress(frame_count / total_frames)
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

st.markdown("---")
st.caption("🚦 Système de détection de véhicules - YOLOv11")
