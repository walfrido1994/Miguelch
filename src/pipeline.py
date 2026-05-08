import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from datetime import datetime
from src.detection.detector import PlayerDetector
from src.tracking.tracker import PlayerTracker
from src.teams.team_classifier import TeamClassifier
from src.ocr.jersey_ocr import JerseyOCR

def process_video(input_path, output_path):
    # --- MÉTRICAS ---
    start_time = datetime.now()
    print(f"\n🚀 Inicio del proceso: {start_time.strftime('%H:%M:%S')}")

    # --- 1. INICIALIZAR HERRAMIENTAS ---
    detector = PlayerDetector()
    tracker = PlayerTracker()
    team_classifier = TeamClassifier()
    ocr_model = JerseyOCR() # Solo se carga una vez
    
    # Memoria para dorsales y colores
    dorsal_memory = {} 
    collected_colors = []
    
    ellipse_annotator = sv.EllipseAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_padding=4, text_scale=0.5)

    video_info = sv.VideoInfo.from_video_path(input_path)
    total_frames = video_info.total_frames
    generator = sv.get_video_frames_generator(input_path)
    
    # --- 2. PROCESAMIENTO FRAME A FRAME ---
    with sv.VideoSink(output_path, video_info) as sink:
        for frame_idx, frame in enumerate(tqdm(generator, total=total_frames, desc="Analizando video")):
            
            results = detector.detect(frame)
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update(detections)
            
            if not detections or detections.tracker_id is None:
                sink.write_frame(frame)
                continue

            new_class_ids = []
            labels = []

            for i, bbox in enumerate(detections.xyxy):
                tracker_id = detections.tracker_id[i]
                
                # --- LÓGICA DE JUGADORES ---
                if detections.class_id[i] == 0:
                    # A. Clasificación de equipo
                    player_color = team_classifier.get_player_color(frame, bbox)
                    team_idx = 0 
                    team_label = "..."

                    if frame_idx < 100:
                        if player_color is not None: collected_colors.append(player_color)
                    elif frame_idx == 100:
                        if len(collected_colors) > 0: team_classifier.fit_teams(collected_colors)
                    else:
                        if player_color is not None:
                            team_label = team_classifier.predict_team(player_color)
                            team_idx = 0 if team_label == "home" else 1

                    # B. OCR de Dorsales (cada 30 frames para no frenar)
                    current_dorsal = ""
                    
                    # Si ya hemos visto este ID con el mismo número 5 veces, lo damos por bueno
                    if tracker_id in dorsal_memory and dorsal_memory[tracker_id]['count'] >= 5:
                        current_dorsal = f" #{dorsal_memory[tracker_id]['numero']}"
                    
                    # Intentamos leer cada 30 frames
                    elif frame_idx % 30 == 0:
                        read_num = ocr_model.read_dorsal(frame, bbox)
                        if read_num:
                            if tracker_id not in dorsal_memory:
                                dorsal_memory[tracker_id] = {'numero': read_num, 'count': 1}
                            elif dorsal_memory[tracker_id]['numero'] == read_num:
                                dorsal_memory[tracker_id]['count'] += 1
                            else:
                                # Si hay conflicto, reseteamos contador para confirmar de nuevo
                                dorsal_memory[tracker_id] = {'numero': read_num, 'count': 1}

                    new_class_ids.append(team_idx)
                    labels.append(f"ID:{tracker_id}{current_dorsal} ({team_label})")
                
                else: # Balón
                    new_class_ids.append(2)
                    labels.append("BALL")

            # Aplicar cambios a las detecciones
            detections.class_id = np.array(new_class_ids)

            # C. Dibujar
            annotated_frame = ellipse_annotator.annotate(scene=frame.copy(), detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            sink.write_frame(annotated_frame)

    # --- 3. MÉTRICAS FINALES ---
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n✅ Proceso completado | Tiempo: {str(duration).split('.')[0]} | FPS: {total_frames / duration.total_seconds():.2f}")

if __name__ == "__main__":
    process_video("input_videos/test.mp4", "output/test_annotated.mp4")