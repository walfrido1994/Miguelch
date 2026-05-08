from ultralytics import YOLO

class PlayerDetector:
    def __init__(self, model_path='models/yolov8x.pt'):
        # Cargamos el modelo en la GPU
        self.model = YOLO(model_path)
    
    def detect(self, frame):
        # Ejecutamos la detección
        # classes=[0, 32] -> 0 es persona, 32 es sports ball (COCO dataset)
        results = self.model(frame, conf=0.4, classes=[0, 32], verbose=False)[0]
        return results