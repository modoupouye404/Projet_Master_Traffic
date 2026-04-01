# app.py - Version ultra simplifiée pour Streamlit Cloud
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import time
import os

st.set_page_config(page_title="Traffic Detection", layout="wide")

st.title("🚦 Détection de Véhicules - YOLOv11")
st.markdown("---")

# Chargement du modèle
@st.cache_resource
def load_model():
    try:
        # Essayer de charger le modèle personnalisé
        if os.path.exists('models/best.pt'):
            model = YOLO('models/best.pt')
        else:
            # Sinon utiliser le modèle par défaut
            model = YOLO('yolo11n.pt')
        return model
    except Exception as e:
        st.error(f"Erreur de chargement du modèle: {e}")
        return None

model = load_model()

if model is None:
    st.warning("⚠️ Modèle non disponible. Utilisation du mode démo.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Paramètres")
    confidence = st.slider("Seuil de confiance", 0.3, 0.9, 0.5, 0.05)
    
    st.markdown("---")
    st.info("💡 **Modèle:** YOLOv11")
    st.caption("Détection de véhicules en temps réel")

# Upload vidéo
st.subheader("📹 Uploader une vidéo")
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
        
        # Interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            video_placeholder = st.empty()
            progress_bar = st.progress(0)
        
        with col2:
            st.markdown("### 📊 Statistiques")
            vehicle_placeholder = st.empty()
            fps_placeholder = st.empty()
            congestion_placeholder = st.empty()
        
        frame_count = 0
        processed = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Traiter 1 frame sur 2 pour accélérer
            if frame_count % 2 != 0:
                continue
            
            # Redimensionner pour performance
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640 / w
                new_w = 640
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h))
            
            # Détection
            results = model(frame, conf=confidence, verbose=False)
            annotated = results[0].plot()
            
            n_vehicles = len(results[0].boxes) if results[0].boxes else 0
            
            # FPS
            elapsed = time.time() - start_time
            fps_val = processed / elapsed if elapsed > 0 else 0
            
            # Congestion
            congestion = min(100, int(n_vehicles * 8))
            if congestion < 30:
                congestion_text = "🟢 Fluide"
            elif congestion < 60:
                congestion_text = "🟡 Modéré"
            else:
                congestion_text = "🔴 Dense"
            
            # Texte sur l'image
            cv2.putText(annotated, f"FPS: {fps_val:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(annotated, f"Vehicules: {n_vehicles}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(annotated, f"Congestion: {congestion}%", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            # Affichage
            video_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                   use_container_width=True)
            progress_bar.progress(frame_count / total_frames)
            
            # Statistiques
            with vehicle_placeholder:
                st.metric("🚗 Véhicules", n_vehicles)
            with fps_placeholder:
                st.metric("⚡ FPS", f"{fps_val:.1f}")
            with congestion_placeholder:
                st.metric("📊 Congestion", congestion_text)
            
            processed += 1
        
        cap.release()
        st.success(f"✅ Analyse terminée! {processed} frames traitées")
        os.unlink(video_path)

st.markdown("---")
st.caption("🚦 Système de détection de véhicules avec YOLOv11")
