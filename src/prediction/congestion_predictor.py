# src/prediction/congestion_predictor.py
import numpy as np
import pandas as pd
from collections import deque
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CongestionPredictor:
    """
    Prédicteur de congestion du trafic
    Utilise les données en temps réel pour prédire les niveaux de congestion
    """
    
    def __init__(self, time_window=30):
        """
        Initialise le prédicteur de congestion
        
        Args:
            time_window: Fenêtre temporelle en frames (30 frames ≈ 1 seconde à 30fps)
        """
        self.time_window = time_window
        self.vehicle_counts = deque(maxlen=time_window)
        self.avg_speeds = deque(maxlen=time_window)
        self.density_history = deque(maxlen=time_window)
        self.occupancy_history = deque(maxlen=time_window)
        
        self.model = None
        self.scaler = StandardScaler()
        self.model_trained = False
        
        # Seuils de congestion
        self.CONGESTION_THRESHOLDS = {
            'free': 0.3,      # < 30% - Trafic fluide
            'moderate': 0.6,   # 30-60% - Trafic modéré
            'heavy': 0.8,      # 60-80% - Trafic dense
            'severe': 1.0      # > 80% - Trafic saturé
        }
        
        # Facteurs de pondération
        self.weights = {
            'vehicle_count': 0.35,
            'avg_speed': 0.35,
            'density': 0.20,
            'occupancy': 0.10
        }
        
    def extract_features(self, vehicle_count, avg_speed, density, occupancy, time_of_day):
        """
        Extrait les features pour la prédiction
        
        Returns:
            features: Vecteur de caractéristiques
        """
        # Statistiques de la fenêtre temporelle
        if len(self.vehicle_counts) > 0:
            count_mean = np.mean(self.vehicle_counts)
            count_std = np.std(self.vehicle_counts)
            count_trend = (self.vehicle_counts[-1] - self.vehicle_counts[0]) / max(len(self.vehicle_counts), 1)
            count_acceleration = count_trend - (self.vehicle_counts[-1] - self.vehicle_counts[-2]) if len(self.vehicle_counts) > 1 else 0
        else:
            count_mean = vehicle_count
            count_std = 0
            count_trend = 0
            count_acceleration = 0
        
        # Statistiques de vitesse
        if len(self.avg_speeds) > 0:
            speed_mean = np.mean(self.avg_speeds)
            speed_std = np.std(self.avg_speeds)
            speed_trend = (self.avg_speeds[-1] - self.avg_speeds[0]) / max(len(self.avg_speeds), 1)
        else:
            speed_mean = avg_speed
            speed_std = 0
            speed_trend = 0
        
        # Taux de changement
        change_rate = (vehicle_count - count_mean) / max(count_mean, 1)
        
        # Heure normalisée (0-24)
        hour_normalized = time_of_day / 24.0
        
        # Features
        features = np.array([
            vehicle_count,           # Nombre actuel de véhicules
            avg_speed,               # Vitesse moyenne
            density,                 # Densité de trafic
            occupancy,               # Taux d'occupation
            time_of_day,             # Heure de la journée
            count_mean,              # Moyenne historique
            count_std,               # Écart-type
            count_trend,             # Tendance
            count_acceleration,      # Accélération du trafic
            speed_mean,              # Vitesse moyenne historique
            speed_std,               # Écart-type de vitesse
            speed_trend,             # Tendance de vitesse
            change_rate,             # Taux de changement
            hour_normalized          # Heure normalisée
        ])
        
        return features.reshape(1, -1)
    
    def train_model(self, historical_data=None):
        """
        Entraîne le modèle de prédiction de congestion
        
        Args:
            historical_data: DataFrame avec colonnes ['vehicle_count', 'avg_speed', 
                            'density', 'occupancy', 'time_of_day', 'congestion_level']
        """
        if historical_data is None:
            # Créer des données synthétiques pour l'exemple
            np.random.seed(42)
            n_samples = 1000
            
            # Simuler différents scénarios de trafic
            data = []
            for _ in range(n_samples):
                hour = np.random.uniform(0, 24)
                # Heures de pointe: 8-9h et 17-19h
                peak_factor = 1 + 2 * np.exp(-((hour - 8.5)**2) / 4) + 1.5 * np.exp(-((hour - 18)**2) / 4)
                
                vehicle_count = int(np.random.poisson(10 * peak_factor))
                avg_speed = max(10, 60 - 40 * (vehicle_count / 30))
                density = min(1, vehicle_count / 40)
                occupancy = min(1, density * 1.2)
                
                # Niveau de congestion réel
                congestion_level = min(1, (vehicle_count / 30) * (1 - avg_speed/60))
                
                data.append([vehicle_count, avg_speed, density, occupancy, hour, congestion_level])
            
            historical_data = pd.DataFrame(data, columns=['vehicle_count', 'avg_speed', 
                                                          'density', 'occupancy', 
                                                          'time_of_day', 'congestion_level'])
        
        # Préparer les features
        X = historical_data[['vehicle_count', 'avg_speed', 'density', 'occupancy', 'time_of_day']].values
        y = historical_data['congestion_level'].values
        
        # Normalisation
        X_scaled = self.scaler.fit_transform(X)
        
        # Entraîner Random Forest
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)
        
        self.model_trained = True
        
        # Sauvegarder le modèle
        os.makedirs('models/prediction_models', exist_ok=True)
        joblib.dump(self.model, 'models/prediction_models/congestion_model.pkl')
        joblib.dump(self.scaler, 'models/prediction_models/congestion_scaler.pkl')
        
        # Score du modèle
        score = self.model.score(X_scaled, y)
        print(f"✅ Modèle de congestion entraîné (R² = {score:.3f})")
        
        return score
    
    def load_model(self):
        """Charge le modèle pré-entraîné"""
        model_path = 'models/prediction_models/congestion_model.pkl'
        scaler_path = 'models/prediction_models/congestion_scaler.pkl'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.model_trained = True
            return True
        return False
    
    def predict_congestion(self, vehicle_count, avg_speed, density, occupancy, time_of_day):
        """
        Prédit le niveau de congestion
        
        Returns:
            dict: Niveau de congestion et métriques associées
        """
        if not self.model_trained:
            if not self.load_model():
                self.train_model()
        
        # Extraire les features
        features = self.extract_features(vehicle_count, avg_speed, density, occupancy, time_of_day)
        
        # Normaliser
        current_features = np.array([[vehicle_count, avg_speed, density, occupancy, time_of_day]])
        features_scaled = self.scaler.transform(current_features)
        
        # Prédiction
        congestion_level = self.model.predict(features_scaled)[0]
        
        # Calculer le niveau de confiance
        confidence = 1 - abs(congestion_level - self._get_historical_mean())
        confidence = max(0.5, min(0.95, confidence))
        
        # Mettre à jour l'historique
        self.vehicle_counts.append(vehicle_count)
        self.avg_speeds.append(avg_speed)
        self.density_history.append(density)
        self.occupancy_history.append(occupancy)
        
        # Déterminer le statut
        if congestion_level < self.CONGESTION_THRESHOLDS['free']:
            status = "Fluide"
            color = "green"
            icon = "🟢"
            recommendation = "Circulation normale"
        elif congestion_level < self.CONGESTION_THRESHOLDS['moderate']:
            status = "Modéré"
            color = "orange"
            icon = "🟡"
            recommendation = "Ralentissements possibles"
        elif congestion_level < self.CONGESTION_THRESHOLDS['heavy']:
            status = "Dense"
            color = "red"
            icon = "🔴"
            recommendation = "Prévoir des retards"
        else:
            status = "Saturé"
            color = "darkred"
            icon = "⛔"
            recommendation = "Éviter si possible"
        
        return {
            'level': congestion_level,
            'percentage': int(congestion_level * 100),
            'status': status,
            'color': color,
            'icon': icon,
            'recommendation': recommendation,
            'confidence': confidence,
            'vehicle_count': vehicle_count,
            'avg_speed': avg_speed,
            'density': density
        }
    
    def _get_historical_mean(self):
        """Retourne la moyenne historique de congestion"""
        if len(self.density_history) > 0:
            return np.mean(self.density_history)
        return 0.5
    
    def detect_congestion_trend(self):
        """
        Détecte la tendance de congestion
        
        Returns:
            dict: Tendance et prévisions
        """
        if len(self.density_history) < 10:
            return {
                'trend': 'Insufficient data',
                'direction': 'stable',
                'prediction': 'N/A'
            }
        
        # Calculer la tendance sur les 10 dernières mesures
        recent_densities = list(self.density_history)[-10:]
        x = np.arange(len(recent_densities))
        slope = np.polyfit(x, recent_densities, 1)[0]
        
        # Déterminer la direction
        if slope > 0.03:
            direction = "en augmentation"
            trend_icon = "📈"
            prediction = "La congestion va s'aggraver dans les prochaines minutes"
        elif slope < -0.03:
            direction = "en diminution"
            trend_icon = "📉"
            prediction = "La circulation devrait s'améliorer prochainement"
        else:
            direction = "stable"
            trend_icon = "➡️"
            prediction = "La situation devrait rester stable"
        
        # Calculer la prédiction à court terme
        next_values = []
        for i in range(1, 6):  # Prédire les 5 prochaines frames
            pred = np.polyval(np.polyfit(x, recent_densities, 2), len(recent_densities) + i)
            next_values.append(max(0, min(1, pred)))
        
        return {
            'trend': direction,
            'slope': slope,
            'icon': trend_icon,
            'prediction': prediction,
            'next_values': next_values,
            'intensity': abs(slope) * 100
        }
    
    def get_congestion_forecast(self, minutes_ahead=5):
        """
        Génère une prévision de congestion
        
        Args:
            minutes_ahead: Minutes à prévoir
        """
        if len(self.density_history) < 20:
            return None
        
        # Analyse des motifs horaires
        current_hour = datetime.now().hour
        is_peak_hour = (7 <= current_hour <= 9) or (17 <= current_hour <= 19)
        
        # Prédiction simple basée sur la tendance et l'heure
        trend = self.detect_congestion_trend()
        
        forecast = []
        for i in range(minutes_ahead):
            factor = 1 + (trend['slope'] * i / 10)
            if is_peak_hour:
                factor *= 1.2
            predicted = min(1, max(0, trend['slope'] * i + np.mean(self.density_history)))
            forecast.append(predicted)
        
        return {
            'minutes': list(range(1, minutes_ahead + 1)),
            'values': forecast,
            'peak_hour': is_peak_hour,
            'trend': trend['trend']
        }