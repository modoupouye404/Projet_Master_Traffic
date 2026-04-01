# app.py - Version avec affichage manuel par étapes
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
    
    # Utiliser session_state pour stocker l'état
    if 'frame_index' not in st.session_state:
        st.session_state.frame_index = 0
    if 'frames' not in st.session_state:
        st.session_state.frames = []
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    
    # Bouton pour charger la vidéo
    if st.button("📥 Charger la vidéo en mémoire", type="primary"):
        with st.spinner("Chargement de la vidéo..."):
            cap = cv2.VideoCapture(video_path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Redimensionner
                h, w = frame.shape[:2]
                if w > 640:
                    scale = 640 / w
                    new_w = 640
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h))
                frames.append(frame)
            cap.release()
            st.session_state.frames = frames
            st.session_state.processed = False
            st.session_state.frame_index = 0
            st.success(f"✅ {len(frames)} frames chargées")
    
    # Afficher les contrôles si des frames sont chargées
    if st.session_state.frames:
        total_frames = len(st.session_state.frames)
        
        # Slider pour naviguer
        frame_num = st.slider("Frame", 0, total_frames - 1, st.session_state.frame_index)
        
        if frame_num != st.session_state.frame_index:
            st.session_state.frame_index = frame_num
            st.session_state.processed = False
        
        # Bouton pour traiter la frame courante
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Traiter cette frame"):
                st.session_state.processed = False
        
        with col2:
            if st.button("📊 Traiter toutes les frames"):
                st.session_state.processed = True
        
        # Traitement
        if not st.session_state.processed:
            # Traiter une seule frame
            frame = st.session_state.frames[st.session_state.frame_index]
            
            with st.spinner("Détection en cours..."):
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
                tracker = SimpleTracker()
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
                
                # Texte
                cv2.putText(annotated, f"Vehicules: {n_vehicles}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Affichage
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_container_width=True)
                
                # Métriques
                col1, col2, col3 = st.columns(3)
                col1.metric("🚗 Véhicules", n_vehicles)
                col2.metric("📊 Frame", f"{st.session_state.frame_index}/{total_frames}")
                col3.metric("📈 Progression", f"{st.session_state.frame_index/total_frames*100:.0f}%")
        
        else:
            # Traiter toutes les frames
            progress_bar = st.progress(0)
            vehicle_counts = []
            
            for i, frame in enumerate(st.session_state.frames):
                # Détection
                results = model(frame, conf=confidence, verbose=False)
                
                n_vehicles = 0
                if results[0].boxes is not None:
                    n_vehicles = len(results[0].boxes)
                vehicle_counts.append(n_vehicles)
                
                progress_bar.progress((i + 1) / total_frames)
            
            # Graphique
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=vehicle_counts, mode='lines', 
                                    name='Véhicules', line=dict(color='#2ecc71', width=2)))
            fig.update_layout(
                title="Évolution du trafic",
                xaxis_title="Frame",
                yaxis_title="Nombre de véhicules",
                height=400,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📊 Max", max(vehicle_counts))
            col2.metric("📊 Moyenne", f"{np.mean(vehicle_counts):.1f}")
            col3.metric("📊 Min", min(vehicle_counts))
            col4.metric("📊 Total frames", total_frames)
            
            st.session_state.processed = False
        
        os.unlink(video_path)

st.markdown("---")
st.caption("🚦 Système de détection de véhicules - YOLOv11")
