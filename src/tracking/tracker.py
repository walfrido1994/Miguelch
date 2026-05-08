import supervision as sv

class PlayerTracker:
    def __init__(self):
        # Inicializamos ByteTrack de la librería supervision
        self.tracker = sv.ByteTrack()

    def update(self, detections):
        # detections debe ser un objeto de supervisión
        # Esto asigna los tracker_id a cada caja
        return self.tracker.update_with_detections(detections)