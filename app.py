# app.py - Version compatible Python 3.11
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

# Imports avec gestion d'erreur
try:
    import cv2
    CV2_OK = True
except ImportError as e:
    st.error(f"❌ OpenCV non disponible: {e}")
    CV2_OK = False
    st.stop()

try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError as e:
    st.error(f"❌ Ultralytics non disponible: {e}")
    YOLO_OK = False
    st.stop()

# Chargement du modèle
@st.cache_resource
def load_model():
    try:
        model = YOLO('yolo11n.pt')
        return model
    except Exception as e:
        st.error(f"Erreur: {e}")
        return None

# Sidebar
with st.sidebar:
    st.header("⚙️ Paramètres")
    confidence = st.slider("Seuil de confiance", 0.3, 0.9, 0.5, 0.05)
    st.info(f"🐍 Python 3.11 | Modèle YOLOv11")

# Charger le modèle
with st.spinner("🔄 Chargement..."):
    model = load_model()
    if model:
        st.success("✅ Modèle chargé!")
    else:
        st.stop()

# Upload vidéo
st.subheader("📹 Upload vidéo")
uploaded = st.file_uploader("Choisissez une vidéo", type=['mp4', 'avi', 'mov'])

if uploaded:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded.read())
    video_path = tfile.name
    tfile.close()
    
    st.video(video_path)
    
    if st.button("🚀 Analyser", type="primary"):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            video_placeholder = st.empty()
            progress = st.progress(0)
        
        with col2:
            vehicle_placeholder = st.empty()
            fps_placeholder = st.empty()
        
        frame_count = 0
        processed = 0
        start_time = time.time()
        vehicle_history = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 1 frame sur 2
            if frame_count % 2 != 0:
                continue
            
            # Redimensionner
            h, w = frame.shape[:2]
            if w > 640:
                frame = cv2.resize(frame, (640, int(h * 640 / w)))
            
            # Détection
            results = model(frame, conf=confidence, verbose=False)
            
            # Visualisation
            annotated = frame.copy()
            n_vehicles = 0
            
            if results[0].boxes is not None:
                n_vehicles = len(results[0].boxes)
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            vehicle_history.append(n_vehicles)
            
            # FPS
            elapsed = time.time() - start_time
            fps_val = processed / elapsed if elapsed > 0 else 0
            
            # Texte
            cv2.putText(annotated, f"FPS: {fps_val:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Vehicules: {n_vehicles}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Affichage
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_placeholder.image(annotated_rgb, use_container_width=True)
            
            # Stats
            vehicle_placeholder.metric("🚗 Véhicules", n_vehicles)
            fps_placeholder.metric("⚡ FPS", f"{fps_val:.1f}")
            
            progress.progress(frame_count / total_frames)
            processed += 1
            time.sleep(0.005)
        
        cap.release()
        st.success(f"✅ Terminé! {processed} frames")
        
        # Graphique
        if vehicle_history:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=vehicle_history, mode='lines', name='Véhicules'))
            fig.update_layout(title="Évolution du trafic", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        os.unlink(video_path)

st.markdown("---")
st.caption("🚦 Détection de véhicules - YOLOv11")
