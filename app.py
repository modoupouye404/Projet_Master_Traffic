# app.py - Version optimisée pour Streamlit Cloud
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import time
import os
import requests

st.set_page_config(
    page_title="Traffic Detection System",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Détection de Véhicules - IA Prédictive")
st.markdown("---")

# Chargement du modèle depuis Google Drive ou URL
@st.cache_resource
def load_model():
    # Option 1: Modèle local (si uploadé)
    if os.path.exists('models/best.pt'):
        return YOLO('models/best.pt')
    
    # Option 2: Télécharger depuis Google Drive (liens publics)
    # Créez un lien public: https://drive.google.com/uc?id=VOTRE_ID
    drive_url = "https://drive.google.com/uc?id=VOTRE_ID_FICHIER"
    
    try:
        response = requests.get(drive_url, stream=True)
        with open('best.pt', 'wb') as f:
            f.write(response.content)
        return YOLO('best.pt')
    except:
        # Option 3: Modèle par défaut
        st.warning("⚠️ Modèle personnalisé non trouvé, utilisation du modèle par défaut")
        return YOLO('yolo11n.pt')

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    confidence = st.slider("Seuil de confiance", 0.3, 0.9, 0.5, 0.05)
    skip_frames = st.slider("Accélération (1=normal, 3=rapide)", 1, 5, 2)
    
    st.markdown("---")
    st.subheader("🎨 Affichage")
    show_speed = st.checkbox("Afficher les vitesses", True)
    show_labels = st.checkbox("Afficher les labels", True)
    
    st.markdown("---")
    st.info("💡 **Performance:**")
    st.write("- FPS optimisé pour serveur Cloud")
    st.write("- Traitement par lots disponible")

# Charger le modèle
with st.spinner("🔄 Chargement du modèle..."):
    model = load_model()
    st.success("✅ Modèle chargé avec succès!")

# Tracker optimisé
class OptimizedTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
    
    def update(self, detections):
        tracks = []
        for det in detections:
            self.tracks[self.next_id] = {'bbox': det['bbox']}
            tracks.append({
                'track_id': self.next_id,
                'bbox': det['bbox'],
                'confidence': det['confidence']
            })
            self.next_id += 1
        return tracks
    
    def visualize(self, frame, tracks):
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            color = (track['track_id'] * 73 % 255, 
                     track['track_id'] * 137 % 255, 
                     track['track_id'] * 211 % 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if show_labels:
                cv2.putText(frame, f"ID:{track['track_id']}", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

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
        
        if st.button("🚀 Démarrer la détection", type="primary", use_container_width=True):
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            tracker = OptimizedTracker()
            
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
                risk_placeholder = st.empty()
            
            frame_count = 0
            processed = 0
            start_time = time.time()
            vehicle_history = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames pour performance
                if frame_count % skip_frames != 0:
                    continue
                
                # Redimensionner pour accélérer (optionnel)
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
                annotated = tracker.visualize(frame.copy(), tracks)
                
                n_vehicles = len(tracks)
                vehicle_history.append(n_vehicles)
                
                # Calcul FPS
                elapsed = time.time() - start_time
                fps_val = processed / elapsed if elapsed > 0 else 0
                
                # Congestion
                congestion = min(100, int(n_vehicles * 8))
                if congestion < 30:
                    congestion_color = "green"
                    congestion_status = "Fluide"
                elif congestion < 60:
                    congestion_color = "orange"
                    congestion_status = "Modéré"
                else:
                    congestion_color = "red"
                    congestion_status = "Dense"
                
                # Risque
                risk = min(100, int(n_vehicles * 3))
                if risk < 20:
                    risk_color = "green"
                    risk_level = "Faible"
                elif risk < 50:
                    risk_color = "orange"
                    risk_level = "Modéré"
                else:
                    risk_color = "red"
                    risk_level = "Élevé"
                
                # Ajouter texte
                cv2.putText(annotated, f"FPS: {fps_val:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated, f"Vehicules: {n_vehicles}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Mise à jour interface
                with vehicle_placeholder.container():
                    st.metric("🚗 Véhicules", n_vehicles, 
                             delta=f"{congestion_status}")
                
                with fps_placeholder.container():
                    st.metric("⚡ FPS", f"{fps_val:.1f}")
                
                with congestion_placeholder.container():
                    st.markdown(f"""
                    <div style='background-color:{congestion_color}20; padding:10px; border-radius:10px; text-align:center'>
                        <span>🚦 CONGESTION</span><br>
                        <span style='font-size:2rem; color:{congestion_color}'>{congestion}%</span><br>
                        <span>{congestion_status}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with risk_placeholder.container():
                    st.markdown(f"""
                    <div style='background-color:{risk_color}20; padding:10px; border-radius:10px; text-align:center'>
                        <span>⚠️ RISQUE</span><br>
                        <span style='font-size:2rem; color:{risk_color}'>{risk}%</span><br>
                        <span>{risk_level}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Affichage vidéo
                video_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                       use_container_width=True)
                progress.progress(frame_count / total_frames)
                processed += 1
                
                # Petit délai pour éviter surcharge
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
                
                # Statistiques
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Max", max(vehicle_history))
                with col2:
                    st.metric("📊 Moyenne", f"{np.mean(vehicle_history):.1f}")
                with col3:
                    st.metric("📈 FPS moyen", f"{processed/elapsed:.1f}")
                with col4:
                    st.metric("⏱️ Durée", f"{frame_count/fps:.1f}s")
            
            os.unlink(video_path)

else:  # Webcam
    st.subheader("📹 Détection en Temps Réel")
    
    run = st.button("▶️ Démarrer", type="primary")
    stop = st.button("⏹️ Arrêter")
    
    if run:
        cap = cv2.VideoCapture(0)
        tracker = OptimizedTracker()
        
        video_placeholder = st.empty()
        stats_placeholder = st.empty()
        
        frame_count = 0
        start_time = time.time()
        
        while not stop:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Redimensionner pour performance
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640 / w
                frame = cv2.resize(frame, (640, int(h * scale)))
            
            results = model(frame, conf=confidence, verbose=False)
            
            detections = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(box.conf[0].cpu().numpy())
                    })
            
            tracks = tracker.update(detections)
            annotated = tracker.visualize(frame.copy(), tracks)
            
            n_vehicles = len(tracks)
            
            elapsed = time.time() - start_time
            fps_val = frame_count / elapsed if elapsed > 0 else 0
            
            cv2.putText(annotated, f"FPS: {fps_val:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Vehicules: {n_vehicles}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            video_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                   use_container_width=True)
            
            with stats_placeholder.container():
                col1, col2 = st.columns(2)
                col1.metric("🚗 Véhicules", n_vehicles)
                col2.metric("⚡ FPS", f"{fps_val:.1f}")
            
            time.sleep(0.03)
        
        cap.release()

st.markdown("---")
st.caption("🚦 Système de détection de véhicules - YOLOv11 | IA Prédictive")