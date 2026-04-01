# src/prediction/accident_predictor.py
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from collections import deque
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class AccidentPredictor:
    """
    Prédicteur de risques d'accidents
    Détecte les comportements anormaux et les situations dangereuses
    """
    
    def __init__(self, history_length=50):
        """
        Initialise le prédicteur d'accidents
        
        Args:
            history_length: Longueur de l'historique des comportements
        """
        self.history_length = history_length
        self.trajectory_history = deque(maxlen=history_length)
        self.speed_history = {}  # track_id -> liste des vitesses
        self.acceleration_history = {}  # track_id -> liste des accélérations
        self.headway_history = deque(maxlen=history_length)  # Distances inter-véhicules
        
        self.anomaly_model = None
        self.risk_model = None
        self.scaler = StandardScaler()
        self.model_trained = False
        
        # Seuils de risque
        self.RISK_THRESHOLDS = {
            'low': 0.3,      # Risque faible
            'moderate': 0.5,  # Risque modéré
            'high': 0.7,     # Risque élevé
            'critical': 0.9  # Risque critique
        }
        
        # Facteurs de risque
        self.risk_factors = {
            'sudden_braking': 0.35,
            'swerving': 0.25,
            'tailgating': 0.20,
            'speed_violation': 0.20
        }
        
    def extract_behavior_features(self, tracks_info, fps):
        """
        Extrait les caractéristiques comportementales des véhicules
        
        Args:
            tracks_info: Liste des tracks actuels
            fps: Frames par seconde
        
        Returns:
            features: Vecteur de caractéristiques de comportement
        """
        if len(tracks_info) < 2:
            return None
        
        # Caractéristiques inter-véhicules
        distances = []
        relative_speeds = []
        headways = []
        
        for i in range(len(tracks_info)):
            for j in range(i+1, len(tracks_info)):
                track1 = tracks_info[i]
                track2 = tracks_info[j]
                
                # Distance entre véhicules
                c1 = track1.get('center', (0, 0))
                c2 = track2.get('center', (0, 0))
                distance = np.sqrt((c2[0] - c1[0])**2 + (c2[1] - c1[1])**2)
                distances.append(distance)
                
                # Vitesse relative
                speed1 = track1.get('speed', 0)
                speed2 = track2.get('speed', 0)
                relative_speed = abs(speed1 - speed2)
                relative_speeds.append(relative_speed)
                
                # Headway (temps inter-véhiculaire)
                if relative_speed > 0:
                    headway = distance / (relative_speed * 0.2778)  # Conversion km/h -> m/s
                    headways.append(headway)
        
        if not distances:
            return None
        
        # Statistiques agrégées
        min_distance = min(distances)
        avg_distance = np.mean(distances)
        std_distance = np.std(distances)
        min_headway = min(headways) if headways else float('inf')
        avg_headway = np.mean(headways) if headways else 0
        
        # Vitesses moyennes
        speeds = [t.get('speed', 0) for t in tracks_info]
        avg_speed = np.mean(speeds) if speeds else 0
        std_speed = np.std(speeds) if speeds else 0
        max_speed = max(speeds) if speeds else 0
        
        # Taux de changement
        speed_variance = std_speed / max(avg_speed, 1)
        
        # Détection des anomalies comportementales
        sudden_brake_count = 0
        swerving_count = 0
        
        for track in tracks_info:
            track_id = track.get('track_id')
            if track_id in self.acceleration_history:
                accels = self.acceleration_history[track_id]
                if len(accels) > 1:
                    # Détection de freinage brusque
                    if accels[-1] < -5:  # Décélération > 5 m/s²
                        sudden_brake_count += 1
                    
                    # Détection de changement de direction
                    if len(self.trajectory_history) > 1:
                        pass
        
        # Features finales
        features = np.array([
            min_distance,           # Distance minimale
            avg_distance,           # Distance moyenne
            std_distance,           # Écart-type des distances
            min_headway,            # Headway minimum
            avg_headway,            # Headway moyen
            avg_speed,              # Vitesse moyenne
            std_speed,              # Écart-type des vitesses
            max_speed,              # Vitesse maximale
            speed_variance,         # Variance de vitesse
            sudden_brake_count,     # Nombre de freinages brusques
            swerving_count,         # Nombre de changements de direction
            len(tracks_info)        # Nombre de véhicules
        ])
        
        return features.reshape(1, -1)
    
    def train_anomaly_detector(self, normal_data=None):
        """
        Entraîne le détecteur d'anomalies
        
        Args:
            normal_data: Données de comportement normal
        """
        if normal_data is None:
            # Créer des données synthétiques de comportement normal
            np.random.seed(42)
            n_samples = 500
            
            normal_data = []
            for _ in range(n_samples):
                min_dist = np.random.uniform(50, 150)  # Distance normale: 50-150px
                avg_dist = np.random.uniform(80, 200)
                headway = np.random.uniform(1.5, 3.5)  # Headway normal: 1.5-3.5s
                speed = np.random.uniform(30, 80)  # Vitesse normale: 30-80 km/h
                speed_var = np.random.uniform(0, 0.3)
                
                normal_data.append([min_dist, avg_dist, headway, speed, speed_var, 0, 0, 0])
            
            normal_data = np.array(normal_data)
        
        # Normaliser
        self.scaler.fit(normal_data)
        normal_scaled = self.scaler.transform(normal_data)
        
        # Entraîner Isolation Forest
        self.anomaly_model = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.anomaly_model.fit(normal_scaled)
        
        # Entraîner le classifieur de risque
        self.risk_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Simuler des données d'entraînement pour le risque
        X_risk = normal_data
        y_risk = np.random.uniform(0, 0.3, len(normal_data))  # Risque bas pour données normales
        
        self.risk_model.fit(X_risk, y_risk)
        
        self.model_trained = True
        
        # Sauvegarder
        os.makedirs('models/prediction_models', exist_ok=True)
        joblib.dump(self.anomaly_model, 'models/prediction_models/anomaly_detector.pkl')
        joblib.dump(self.risk_model, 'models/prediction_models/risk_model.pkl')
        joblib.dump(self.scaler, 'models/prediction_models/risk_scaler.pkl')
        
        print("✅ Modèles de prédiction d'accidents entraînés")
    
    def load_models(self):
        """Charge les modèles pré-entraînés"""
        model_paths = [
            'models/prediction_models/anomaly_detector.pkl',
            'models/prediction_models/risk_model.pkl',
            'models/prediction_models/risk_scaler.pkl'
        ]
        
        if all(os.path.exists(p) for p in model_paths):
            self.anomaly_model = joblib.load(model_paths[0])
            self.risk_model = joblib.load(model_paths[1])
            self.scaler = joblib.load(model_paths[2])
            self.model_trained = True
            return True
        return False
    
    def update_vehicle_behavior(self, track_id, speed, acceleration, position):
        """
        Met à jour l'historique des comportements des véhicules
        """
        if track_id not in self.speed_history:
            self.speed_history[track_id] = deque(maxlen=20)
            self.acceleration_history[track_id] = deque(maxlen=20)
        
        self.speed_history[track_id].append(speed)
        self.acceleration_history[track_id].append(acceleration)
        
        # Mettre à jour la trajectoire
        self.trajectory_history.append((track_id, position))
    
    def predict_accident_risk(self, tracks_info, fps, weather='normal', road_condition='dry'):
        """
        Prédit le risque d'accident
        
        Returns:
            dict: Niveau de risque et facteurs contributifs
        """
        if not self.model_trained:
            if not self.load_models():
                self.train_anomaly_detector()
        
        # Extraire les features comportementales
        behavior = self.extract_behavior_features(tracks_info, fps)
        
        if behavior is None:
            return self._default_risk_response()
        
        # Calculer le risque basé sur différents facteurs
        risk_factors = []
        risk_score = 0
        
        # Facteur 1: Proximité excessive (tailgating)
        min_distance = behavior[0][0]
        if min_distance < 30:
            risk_score += self.risk_factors['tailgating']
            risk_factors.append({
                'type': 'tailgating',
                'description': 'Distance de sécurité insuffisante',
                'severity': 'high',
                'value': f'{min_distance:.0f}px'
            })
        elif min_distance < 50:
            risk_score += self.risk_factors['tailgating'] * 0.6
            risk_factors.append({
                'type': 'tailgating',
                'description': 'Distance de sécurité réduite',
                'severity': 'moderate',
                'value': f'{min_distance:.0f}px'
            })
        
        # Facteur 2: Différence de vitesse excessive
        avg_speed = behavior[0][5]
        std_speed = behavior[0][6]
        speed_variance = behavior[0][8]
        
        if speed_variance > 0.5:
            risk_score += self.risk_factors['speed_violation']
            risk_factors.append({
                'type': 'speed_variance',
                'description': 'Grande variation de vitesse entre véhicules',
                'severity': 'high',
                'value': f'{speed_variance:.2f}'
            })
        
        # Facteur 3: Vitesse excessive
        max_speed = behavior[0][7]
        if max_speed > 90:
            risk_score += self.risk_factors['speed_violation']
            risk_factors.append({
                'type': 'speeding',
                'description': 'Vitesse excessive',
                'severity': 'high',
                'value': f'{max_speed:.0f} km/h'
            })
        elif max_speed > 70:
            risk_score += self.risk_factors['speed_violation'] * 0.5
            risk_factors.append({
                'type': 'speeding',
                'description': 'Vitesse élevée',
                'severity': 'moderate',
                'value': f'{max_speed:.0f} km/h'
            })
        
        # Facteur 4: Détection d'anomalie comportementale
        try:
            behavior_scaled = self.scaler.transform(behavior)
            anomaly_score = self.anomaly_model.decision_function(behavior_scaled)[0]
            is_anomaly = self.anomaly_model.predict(behavior_scaled)[0] == -1
            
            if is_anomaly:
                risk_score += self.risk_factors['sudden_braking'] * 1.5
                risk_factors.append({
                    'type': 'anomaly',
                    'description': 'Comportement anormal détecté',
                    'severity': 'critical',
                    'value': f'Score: {anomaly_score:.2f}'
                })
        except:
            pass
        
        # Facteur 5: Conditions météorologiques
        weather_factors = {
            'rain': 0.2,
            'snow': 0.3,
            'fog': 0.25,
            'storm': 0.35
        }
        risk_score += weather_factors.get(weather, 0)
        
        # Facteur 6: État de la route
        road_factors = {
            'wet': 0.15,
            'icy': 0.3,
            'snow': 0.25,
            'damaged': 0.2
        }
        risk_score += road_factors.get(road_condition, 0)
        
        # Normaliser le risque (0-1)
        risk_level = min(risk_score, 1.0)
        
        # Déterminer le niveau d'alerte
        if risk_level >= self.RISK_THRESHOLDS['critical']:
            alert_level = "CRITIQUE"
            color = "darkred"
            icon = "🔴⚠️"
            recommendation = "RISQUE ÉLEVÉ - Ralentir immédiatement"
        elif risk_level >= self.RISK_THRESHOLDS['high']:
            alert_level = "ÉLEVÉ"
            color = "red"
            icon = "🔴"
            recommendation = "Soyez extrêmement vigilant"
        elif risk_level >= self.RISK_THRESHOLDS['moderate']:
            alert_level = "MODÉRÉ"
            color = "orange"
            icon = "🟡"
            recommendation = "Surveillance accrue recommandée"
        else:
            alert_level = "FAIBLE"
            color = "green"
            icon = "🟢"
            recommendation = "Circulation normale"
        
        return {
            'risk_level': risk_level,
            'risk_percentage': int(risk_level * 100),
            'alert_level': alert_level,
            'color': color,
            'icon': icon,
            'recommendation': recommendation,
            'risk_factors': risk_factors,
            'vehicle_count': len(tracks_info),
            'avg_speed': behavior[0][5] if behavior is not None else 0,
            'min_distance': behavior[0][0] if behavior is not None else 0
        }
    
    def _default_risk_response(self):
        """Retourne une réponse par défaut"""
        return {
            'risk_level': 0.1,
            'risk_percentage': 10,
            'alert_level': 'FAIBLE',
            'color': 'green',
            'icon': '🟢',
            'recommendation': 'Pas assez de données pour évaluer',
            'risk_factors': [],
            'vehicle_count': 0,
            'avg_speed': 0,
            'min_distance': 0
        }
    
    def detect_dangerous_vehicles(self, tracks_info):
        """
        Identifie les véhicules présentant un comportement dangereux
        
        Returns:
            list: Véhicules dangereux avec leurs comportements
        """
        dangerous = []
        
        for track in tracks_info:
            track_id = track.get('track_id')
            speed = track.get('speed', 0)
            
            risk = 0
            reasons = []
            
            # Vérifier la vitesse
            if speed > 90:
                risk += 0.5
                reasons.append('vitesse excessive')
            elif speed > 70:
                risk += 0.3
                reasons.append('vitesse élevée')
            
            # Vérifier les freinages brusques
            if track_id in self.acceleration_history:
                accels = list(self.acceleration_history[track_id])
                if len(accels) > 2 and min(accels) < -5:
                    risk += 0.4
                    reasons.append('freinage brusque')
            
            if risk > 0.3:
                dangerous.append({
                    'track_id': track_id,
                    'risk': risk,
                    'reasons': reasons,
                    'speed': speed
                })
        
        return sorted(dangerous, key=lambda x: x['risk'], reverse=True)