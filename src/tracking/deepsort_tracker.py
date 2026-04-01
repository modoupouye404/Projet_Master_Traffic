# src/tracking/deepsort_tracker.py
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

class VehicleTracker:
    """
    Tracker multi-objets utilisant DeepSORT pour le suivi de véhicules
    """
    
    def __init__(self, max_age=30, n_init=3, nn_budget=100, use_cuda=True):
        """
        Initialise le tracker DeepSORT
        
        Args:
            max_age: Nombre de frames maximum sans mise à jour avant suppression
            n_init: Nombre de frames initiales pour confirmer un track
            nn_budget: Budget pour la ré-identification
            use_cuda: Utiliser GPU si disponible
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nms_max_overlap=0.5,
            max_cosine_distance=0.3,
            nn_budget=nn_budget,
            use_cuda=use_cuda,
            embedder='mobilenet'  # Utiliser MobileNet pour l'embedding
        )
        
        # Historique des trajectoires
        self.trajectories = {}  # track_id -> liste des positions
        self.vehicle_stats = {}  # track_id -> statistiques
        self.frame_count = 0
        
    def update(self, frame, detections):
        """
        Met à jour les tracks avec les nouvelles détections
        
        Args:
            frame: Image numpy (BGR)
            detections: Liste de détections [{'bbox': [x1,y1,x2,y2], 'confidence': float, 'class_id': int}]
            
        Returns:
            tracks_info: Liste des tracks avec leurs informations
        """
        self.frame_count += 1
        
        # Convertir les détections au format DeepSORT
        deepsort_dets = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            w = x2 - x1
            h = y2 - y1
            confidence = det['confidence']
            class_id = det['class_id']
            
            deepsort_dets.append(
                (([x1, y1, w, h]), confidence, class_id)
            )
        
        # Mettre à jour le tracker
        tracked_objects = self.tracker.update_tracks(deepsort_dets, frame=frame)
        
        tracks_info = []
        
        for track in tracked_objects:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()  # Left, Top, Right, Bottom
            x1, y1, x2, y2 = ltrb
            class_id = track.get_det_class()
            
            # Calculer le centre
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Mettre à jour l'historique des trajectoires
            if track_id not in self.trajectories:
                self.trajectories[track_id] = []
                self.vehicle_stats[track_id] = {
                    'class_id': class_id,
                    'first_seen': self.frame_count,
                    'last_seen': self.frame_count,
                    'track_length': 0,
                    'positions': [],
                    'speeds': []
                }
            
            self.trajectories[track_id].append(center)
            self.vehicle_stats[track_id]['positions'].append(center)
            self.vehicle_stats[track_id]['track_length'] += 1
            self.vehicle_stats[track_id]['last_seen'] = self.frame_count
            
            tracks_info.append({
                'track_id': track_id,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'class_id': class_id,
                'center': center,
                'confidence': track.get_det_conf()
            })
        
        # Nettoyer les tracks trop anciens
        self._cleanup_old_tracks()
        
        return tracks_info
    
    def get_vehicle_speed(self, track_id, fps, pixel_to_meter_ratio=0.05):
        """
        Calcule la vitesse du véhicule en km/h
        
        Args:
            track_id: ID du track
            fps: Frames par seconde
            pixel_to_meter_ratio: Conversion pixels -> mètres
            
        Returns:
            speed_kmh: Vitesse en km/h
        """
        if track_id not in self.trajectories:
            return 0
        
        positions = self.trajectories[track_id]
        if len(positions) < 5:  # Besoin d'au moins 5 positions
            return 0
        
        # Utiliser les 5 dernières positions pour plus de précision
        recent_positions = positions[-5:]
        
        # Calculer la distance totale parcourue
        total_distance = 0
        for i in range(1, len(recent_positions)):
            p1 = recent_positions[i-1]
            p2 = recent_positions[i]
            distance_pixels = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            total_distance += distance_pixels
        
        # Temps écoulé (frames * temps par frame)
        time_seconds = (len(recent_positions) - 1) / fps
        
        if time_seconds > 0:
            distance_meters = total_distance * pixel_to_meter_ratio
            speed_mps = distance_meters / time_seconds
            speed_kmh = speed_mps * 3.6
            return speed_kmh
        
        return 0
    
    def get_traffic_density(self, frame_shape, tracks_info, zones=None):
        """
        Calcule la densité de trafic par zone
        
        Args:
            frame_shape: Dimensions de l'image (height, width)
            tracks_info: Liste des tracks
            zones: Dictionnaire des zones (optionnel)
            
        Returns:
            density: Densité par zone
            zone_counts: Nombre de véhicules par zone
        """
        height, width = frame_shape[:2]
        
        # Définir les zones par défaut
        if zones is None:
            zones = {
                'north': (0, 0, width, height//3),
                'center': (0, height//3, width, height//3),
                'south': (0, 2*height//3, width, height//3)
            }
        
        zone_counts = {zone: 0 for zone in zones}
        
        for track in tracks_info:
            x1, y1, x2, y2 = track['bbox']
            center_y = (y1 + y2) // 2
            
            if center_y < height//3:
                zone_counts['north'] += 1
            elif center_y < 2*height//3:
                zone_counts['center'] += 1
            else:
                zone_counts['south'] += 1
        
        # Calculer la densité (véhicules par 100m²)
        density = {}
        for zone, count in zone_counts.items():
            zone_area = zones[zone][2] * zones[zone][3]
            density[zone] = count / (zone_area / 10000)  # par 100m²
        
        return density, zone_counts
    
    def detect_traffic_anomaly(self, tracks_info):
        """
        Détecte les anomalies de circulation (vitesse excessive, arrêt brusque)
        
        Returns:
            anomalies: Liste des anomalies détectées
        """
        anomalies = []
        
        for track in tracks_info:
            track_id = track['track_id']
            
            # Vérifier la vitesse
            if track_id in self.vehicle_stats:
                speed = self.vehicle_stats[track_id].get('current_speed', 0)
                
                if speed > 80:  # > 80 km/h
                    anomalies.append({
                        'type': 'speeding',
                        'track_id': track_id,
                        'speed': speed,
                        'severity': 'high'
                    })
                elif speed < 5 and self.vehicle_stats[track_id]['track_length'] > 30:
                    anomalies.append({
                        'type': 'stopped',
                        'track_id': track_id,
                        'duration': self.vehicle_stats[track_id]['track_length'],
                        'severity': 'medium'
                    })
        
        return anomalies
    
    def _cleanup_old_tracks(self, max_age_frames=60):
        """
        Supprime les tracks trop anciens
        """
        to_remove = []
        for track_id, stats in self.vehicle_stats.items():
            if self.frame_count - stats['last_seen'] > max_age_frames:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.trajectories[track_id]
            del self.vehicle_stats[track_id]
    
    def visualize_tracks(self, frame, tracks_info, show_speed=True, show_trajectory=True):
        """
        Visualise les tracks sur la frame
        
        Args:
            frame: Image numpy
            tracks_info: Liste des tracks
            show_speed: Afficher la vitesse
            show_trajectory: Afficher la trajectoire
            
        Returns:
            frame: Image annotée
        """
        # Générer des couleurs aléatoires mais stables par track_id
        np.random.seed(42)
        colors = {}
        
        for track in tracks_info:
            track_id = track['track_id']
            
            # Générer une couleur pour ce track
            if track_id not in colors:
                colors[track_id] = tuple(np.random.randint(0, 255, 3).tolist())
            
            color = colors[track_id]
            x1, y1, x2, y2 = track['bbox']
            
            # Dessiner la bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Dessiner l'ID du track
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Afficher la vitesse
            if show_speed:
                speed = track.get('speed', 0)
                cv2.putText(frame, f"{speed:.1f} km/h", (x1, y2 + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Dessiner la trajectoire
            if show_trajectory and track_id in self.trajectories:
                points = self.trajectories[track_id]
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], color, 2)
        
        return frame
    
    def get_statistics(self):
        """
        Retourne les statistiques du tracking
        
        Returns:
            stats: Dictionnaire des statistiques
        """
        return {
            'total_vehicles_tracked': len(self.trajectories),
            'active_tracks': len([s for s in self.vehicle_stats.values() 
                                 if self.frame_count - s['last_seen'] < 30]),
            'average_track_length': np.mean([s['track_length'] 
                                           for s in self.vehicle_stats.values()]) if self.vehicle_stats else 0
        }