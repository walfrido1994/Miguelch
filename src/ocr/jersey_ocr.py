import os
import logging

# --- HACK PARA WINDOWS Y PADDLE v3 ---
# Desactivamos MKLDNN y el nuevo ejecutor PIR que están dando el NotImplementedError
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_allocator_strategy'] = 'naive_best_fit'
os.environ['PADDLE_ONEDNN_VERBOSE'] = '0'

from paddleocr import PaddleOCR
import cv2
import numpy as np

class JerseyOCR:
    def __init__(self):
        logging.getLogger("ppocr").setLevel(logging.ERROR)
        self.ocr = None

    def read_dorsal(self, frame, bbox):
        if self.ocr is None:
            # Inicialización ultra-limpia
            # Al no pasar argumentos, evitamos el ValueError
            self.ocr = PaddleOCR(lang='en')
            
        x1, y1, x2, y2 = map(int, bbox)
        width, height = x2 - x1, y2 - y1

        # Recorte del torso
        roi_y1, roi_y2 = y1 + int(height * 0.1), y1 + int(height * 0.5)
        roi_x1, roi_x2 = x1 + int(width * 0.1), x1 + int(width * 0.9)
        
        crop = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if crop.size == 0: return None

        # Ejecutamos el OCR
        try:
            result = self.ocr.ocr(crop)
            if result and result[0]:
                for line in result[0]:
                    text = line[1][0]
                    conf = line[1][1]
                    digit_text = ''.join(filter(str.isdigit, text))
                    if digit_text and conf > 0.5:
                        return digit_text
        except Exception:
            # Si falla el OCR por cualquier motivo técnico, 
            # devolvemos None para que el pipeline no se detenga
            return None
        return None