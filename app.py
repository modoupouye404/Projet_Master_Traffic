# app.py - Version corrigée pour Streamlit Cloud
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
        # Essayer le modèle personnalisé
        if os.path.exists('models/best.pt'):
            model = YOLO('models/best.pt')
            st.sidebar.success("✅ Modèle personnalisé chargé")
        else:
            # Modèle par défaut
            model = YOLO('yolo11n.pt')
            st.sidebar.info("📌 Modèle par défaut (yolo11n)")
        return model
    except Exception as e:
        st.sidebar.error(f"Erreur: {e}")
        return None

model = load_model()

if model is None:
    st.error("❌ Impossible de charger le modèle")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Paramètres")
    confidence = st.slider("Seuil de confiance", 0.3, 0.9, 0.5, 0.05)
    st.markdown("---")
    st.caption("Détection de véhicules en temps réel")

# Upload vidéo
st.subheader("📹 Uploader une vidéo")
uploaded = st.file_uploader("Choisissez une vidéo", type=['mp4', 'avi', 'mov'])

if uploaded:
    # Sauvegarder temporairement
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded.read())
    video_path = tfile.name
    tfile.close()
    
    # Afficher la vidéo
    st.video(video_path)
    
    if st.button("🚀 Démarrer la détection", type="primary"):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Placeholders
        video_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        # Colonnes pour les stats
        col1, col2, col3 = st.columns(3)
        with col1:
            vehicle_metric = st.empty()
        with col2:
            fps_metric = st.empty()
        with col3:
            congestion_metric = st.empty()
        
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
            
            # Récupérer l'image annotée
            if hasattr(results[0], 'plot'):
                annotated = results[0].plot()
            else:
                annotated = frame.copy()
            
            n_vehicles = 0
            if results[0].boxes is not None:
                n_vehicles = len(results[0].boxes)
            
            # FPS
            elapsed = time.time() - start_time
            fps_val = processed / elapsed if elapsed > 0 else 0
            
            # Congestion
            congestion = min(100, int(n_vehicles * 8))
            if congestion < 30:
                congestion_text = "Fluide"
            elif congestion < 60:
                congestion_text = "Modéré"
            else:
                congestion_text = "Dense"
            
            # Ajouter du texte
            cv2.putText(annotated, f"FPS: {fps_val:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(annotated, f"Vehicules: {n_vehicles}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(annotated, f"Congestion: {congestion}%", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            # Affichage
            video_placeholder.image(annotated, channels="BGR", use_container_width=True)
            progress_bar.progress(frame_count / total_frames)
            
            # Mise à jour des métriques
            vehicle_metric.metric("🚗 Véhicules", n_vehicles)
            fps_metric.metric("⚡ FPS", f"{fps_val:.1f}")
            congestion_metric.metric("📊 Congestion", congestion_text)
            
            processed += 1
            
            # Éviter la surcharge
            time.sleep(0.005)
        
        cap.release()
        
        st.success(f"✅ Analyse terminée! {processed} frames traitées")
        os.unlink(video_path)

st.markdown("---")
st.caption("🚦 Système de détection de véhicules avec YOLOv11")
