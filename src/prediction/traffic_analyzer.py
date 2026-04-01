# src/prediction/traffic_analyzer.py
import numpy as np
from collections import deque
from datetime import datetime
import pandas as pd

class TrafficAnalyzer:
    """
    Analyseur de trafic en temps réel
    Combine les données de congestion et d'accidents
    """
    
    def __init__(self):
        self.congestion_predictor = None
        self.accident_predictor = None
        self.vehicle_history = deque(maxlen=300)  # Derniers 10 secondes à 30fps
        self.speed_history = deque(maxlen=300)
        self.risk_history = deque(maxlen=300)
        
    def initialize(self, congestion_predictor, accident_predictor):
        """Initialise avec les prédicteurs"""
        self.congestion_predictor = congestion_predictor
        self.accident_predictor = accident_predictor
    
    def analyze_frame(self, tracks_info, fps, vehicle_count, avg_speed, density, occupancy):
        """
        Analyse complète d'une frame
        
        Returns:
            dict: Résultats complets de l'analyse
        """
        # Heure de la journée
        current_hour = datetime.now().hour + datetime.now().minute / 60
        
        # Prédiction de congestion
        congestion = self.congestion_predictor.predict_congestion(
            vehicle_count, avg_speed, density, occupancy, current_hour
        )
        
        # Tendance de congestion
        congestion_trend = self.congestion_predictor.detect_congestion_trend()
        
        # Prédiction de risque d'accident
        accident_risk = self.accident_predictor.predict_accident_risk(tracks_info, fps)
        
        # Véhicules dangereux
        dangerous_vehicles = self.accident_predictor.detect_dangerous_vehicles(tracks_info)
        
        # Mettre à jour l'historique
        self.vehicle_history.append(vehicle_count)
        self.speed_history.append(avg_speed)
        self.risk_history.append(accident_risk['risk_level'])
        
        # Calculer les tendances
        vehicle_trend = self._calculate_trend(self.vehicle_history)
        speed_trend = self._calculate_trend(self.speed_history)
        
        # Score de santé du trafic (0-100)
        health_score = self._calculate_health_score(congestion, accident_risk)
        
        return {
            'timestamp': datetime.now(),
            'congestion': congestion,
            'congestion_trend': congestion_trend,
            'accident_risk': accident_risk,
            'dangerous_vehicles': dangerous_vehicles,
            'statistics': {
                'vehicle_count': vehicle_count,
                'avg_speed': avg_speed,
                'density': density,
                'occupancy': occupancy
            },
            'trends': {
                'vehicle_trend': vehicle_trend,
                'speed_trend': speed_trend,
                'risk_trend': self._calculate_trend(self.risk_history)
            },
            'health_score': health_score,
            'recommendations': self._generate_recommendations(congestion, accident_risk, dangerous_vehicles)
        }
    
    def _calculate_trend(self, history):
        """Calcule la tendance d'une série temporelle"""
        if len(history) < 10:
            return {'direction': 'stable', 'change': 0}
        
        recent = list(history)[-10:]
        slope = np.polyfit(range(len(recent)), recent, 1)[0]
        
        if slope > 0.5:
            direction = 'up'
            description = 'en augmentation'
        elif slope < -0.5:
            direction = 'down'
            description = 'en diminution'
        else:
            direction = 'stable'
            description = 'stable'
        
        return {
            'direction': direction,
            'description': description,
            'slope': slope,
            'change_percent': (recent[-1] - recent[0]) / max(recent[0], 1) * 100
        }
    
    def _calculate_health_score(self, congestion, accident_risk):
        """
        Calcule un score de santé du trafic (0-100)
        Plus le score est élevé, meilleure est la circulation
        """
        # Pondération
        congestion_weight = 0.6
        risk_weight = 0.4
        
        # Convertir en score (inverse)
        congestion_score = (1 - congestion['level']) * 100
        risk_score = (1 - accident_risk['risk_level']) * 100
        
        health_score = congestion_score * congestion_weight + risk_score * risk_weight
        
        return {
            'score': int(health_score),
            'level': self._get_health_level(health_score),
            'congestion_contribution': int(congestion_score * congestion_weight),
            'risk_contribution': int(risk_score * risk_weight)
        }
    
    def _get_health_level(self, score):
        """Détermine le niveau de santé du trafic"""
        if score >= 80:
            return 'excellent'
        elif score >= 60:
            return 'good'
        elif score >= 40:
            return 'fair'
        elif score >= 20:
            return 'poor'
        else:
            return 'critical'
    
    def _generate_recommendations(self, congestion, accident_risk, dangerous_vehicles):
        """
        Génère des recommandations basées sur l'analyse
        """
        recommendations = []
        
        # Recommandations de congestion
        if congestion['level'] > 0.7:
            recommendations.append({
                'type': 'congestion',
                'priority': 'high',
                'message': 'Trafic saturé - Itinéraires alternatifs recommandés',
                'action': 'alert'
            })
        elif congestion['level'] > 0.5:
            recommendations.append({
                'type': 'congestion',
                'priority': 'medium',
                'message': 'Ralentissements détectés - Anticiper les retards',
                'action': 'caution'
            })
        
        # Recommandations de risque
        if accident_risk['risk_level'] > 0.7:
            recommendations.append({
                'type': 'accident_risk',
                'priority': 'critical',
                'message': 'RISQUE ÉLEVÉ D\'ACCIDENT - Ralentir immédiatement',
                'action': 'emergency'
            })
        elif accident_risk['risk_level'] > 0.5:
            recommendations.append({
                'type': 'accident_risk',
                'priority': 'high',
                'message': 'Situation dangereuse - Vigilance accrue requise',
                'action': 'warning'
            })
        
        # Recommandations sur les véhicules dangereux
        if dangerous_vehicles:
            recommendations.append({
                'type': 'dangerous_vehicle',
                'priority': 'high',
                'message': f'{len(dangerous_vehicles)} véhicule(s) à comportement dangereux détecté(s)',
                'action': 'monitor'
            })
        
        return recommendations