# app.py - Version optimisée pour Streamlit Cloud
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import time
import os

st.set_page_config(page_title="Traffic Detection", layout="wide")

st.title("🚦 Détection de Véhicules - IA Prédictive")
st.markdown("---")

# Chargement du modèle avec fallback
@st.cache_resource
def load_model():
    try:
        # Essayer de charger le modèle personnalisé
        model = YOLO('models/best.pt')
        return model
    except:
        try:
            # Si pas trouvé, utiliser le modèle par défaut
            model = YOLO('yolo11n.pt')
            st.info("📌 Modèle par défaut chargé")
            return model
        except:
            st.error("❌ Aucun modèle disponible")
            return None

model = load_model()

if model is None:
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Paramètres")
    confidence = st.slider("Seuil de confiance", 0.3, 0.9, 0.5, 0.05)
    skip_frames = st.slider("Accélération", 1, 5, 2)
    
    st.markdown("---")
    st.info("💡 **Info:** Modèle YOLOv11")

# Upload vidéo
uploaded = st.file_uploader("Choisissez une vidéo", type=['mp4', 'avi', 'mov'])

if uploaded:
    # Sauvegarder temporairement
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded.read())
    video_path = tfile.name
    tfile.close()
    
    st.video(video_path)
    
    if st.button("🚀 Démarrer la détection", type="primary"):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Placeholders
        video_placeholder = st.empty()
        progress_bar = st.progress(0)
        stats_placeholder = st.empty()
        
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
            
            # Détection
            try:
                results = model(frame, conf=confidence, verbose=False)
                annotated = results[0].plot()
                
                n_vehicles = len(results[0].boxes) if results[0].boxes else 0
                vehicle_history.append(n_vehicles)
                
                # FPS
                elapsed = time.time() - start_time
                fps_val = processed / elapsed if elapsed > 0 else 0
                
                # Congestion
                congestion = min(100, int(n_vehicles * 6))
                if congestion < 30:
                    congestion_status = "Fluide 🟢"
                elif congestion < 60:
                    congestion_status = "Modéré 🟡"
                else:
                    congestion_status = "Dense 🔴"
                
                # Texte
                cv2.putText(annotated, f"FPS: {fps_val:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(annotated, f"Vehicules: {n_vehicles}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(annotated, f"Congestion: {congestion}%", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                
                # Affichage
                video_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                       use_container_width=True)
                
                # Stats
                with stats_placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    col1.metric("🚗 Véhicules", n_vehicles)
                    col2.metric("⚡ FPS", f"{fps_val:.1f}")
                    col3.metric("📊 Congestion", congestion_status)
                
                progress_bar.progress(frame_count / total_frames)
                processed += 1
                
            except Exception as e:
                continue
        
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
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        os.unlink(video_path)

st.markdown("---")
st.caption("🚦 Système de détection de véhicules - YOLOv11")
