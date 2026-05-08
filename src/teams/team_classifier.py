import cv2
import numpy as np
from sklearn.cluster import KMeans

class TeamClassifier:
    def __init__(self):
        self.team_colors = {}
        self.kmeans = None

    def get_player_color(self, frame, bbox):
        # 1. Extraer el ROI (Región de Interés)
        # Solo queremos la parte superior del torso (aprox. 20% al 40% de la altura de la caja)
        x1, y1, x2, y2 = map(int, bbox)
        width = x2 - x1
        height = y2 - y1
        
        # Recortamos un cuadrado en el pecho
        roi_y1 = y1 + int(height * 0.2)
        roi_y2 = y1 + int(height * 0.5)
        roi_x1 = x1 + int(width * 0.2)
        roi_x2 = x1 + int(width * 0.8)
        
        player_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        if player_roi.size == 0:
            return None

        # 2. Preprocesar para K-Means
        # Convertimos de BGR a RGB y aplanamos la imagen
        roi_rgb = cv2.cvtColor(player_roi, cv2.COLOR_BGR2RGB)
        pixels = roi_rgb.reshape(-1, 3)

        # Usamos KMeans para encontrar el color dominante en ese parche
        # n_clusters=1 porque queremos el color principal de ESTE jugador
        kmeans = KMeans(n_clusters=1, n_init=1, max_iter=10).fit(pixels)
        return kmeans.cluster_centers_[0]

    def fit_teams(self, player_colors):
        # Agrupamos todos los colores detectados en 2 grandes grupos (los 2 equipos)
        self.kmeans = KMeans(n_clusters=2, n_init=10)
        self.kmeans.fit(player_colors)
        
        # Guardamos los colores promedio de cada equipo
        self.team_colors[0] = self.kmeans.cluster_centers_[0]
        self.team_colors[1] = self.kmeans.cluster_centers_[1]
        print(f"Colores de equipos detectados: \nE1: {self.team_colors[0]}\nE2: {self.team_colors[1]}")

    def predict_team(self, color):
        if self.kmeans is None:
            return "unknown"
        # Predice si el color pertenece al cluster 0 o 1
        prediction = self.kmeans.predict(color.reshape(1, -1))[0]
        return "home" if prediction == 0 else "away"