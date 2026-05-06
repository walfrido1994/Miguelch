# ⚽ TacticalCore Video AI

Sistema avanzado de análisis táctico de fútbol basado en Computer Vision. El proyecto utiliza modelos de Deep Learning para automatizar el seguimiento de jugadores, reconocimiento de dorsales y mapeo de posiciones reales sobre el terreno de juego.

## 🚀 Estado del Proyecto: Fase 1 (Setup del Entorno)
- [x] Configuración de entorno virtual y dependencias.
- [x] Verificación de aceleración por hardware (GPU).
- [x] Estructura de carpetas del proyecto definida.
- [ ] Fase 2: Detección + tracking básico (Próximamente).

---

## 🛠️ Requisitos Previos

### Hardware
* **GPU:** NVIDIA RTX 2070 / 3090 (o superior con soporte CUDA).
* **Drivers:** Controladores de NVIDIA actualizados a la última versión.
* **CUDA:** [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) instalado.

### Software
* **Python:** v3.11.9
* **Git:** Para gestión de versiones.

---

## 🔧 Configuración del Entorno en Windows

El proyecto se encuentra centralizado en la unidad `/tacticalcore-video-ai`.

### 1. Inicialización del entorno virtual
Desde la terminal en la carpeta raíz, ejecutar:

```powershell
py -m venv venv
.\venv\Scripts\activate
```

Nota: Si recibes un error de permisos al ejecutar el script de activación, abre una terminal como Administrador y ejecuta:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

2. Instalación de dependencias
El archivo requirements.txt incluye todas las librerías necesarias (Ultralytics, OpenCV, PaddleOCR, etc.). Para instalarlas:

```powershell
pip install -r requirements.txt
```

🖥️ Verificación de Hardware (GPU)
Para asegurar que los modelos de IA utilizarán la potencia de la tarjeta gráfica y no el procesador (CPU), se utiliza el script check_gpu.py:

```powershell
import torch

def check():
    print("--- Verificación de GPU ---")
    available = torch.cuda.is_available()
    print(f"¿CUDA disponible?: {available}")
    
    if available:
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM total: {vram:.2f} GB")
    else:
        print("¡OJO! PyTorch no detecta la GPU. Revisa los drivers.")

if __name__ == "__main__":
    check()
```
Ejecución:
```powershell
python check_gpu.py
```
Resultado esperado:
```plaintext
--- Verificación de GPU ---
¿CUDA disponible?: True
GPU detectada: NVIDIA GeForce RTX 2070
VRAM total: 8.00 GB
```
⚠️ Solución de Problemas (Troubleshooting)
Error en la detección de GPU / PyTorch
Si al ejecutar el script de verificación CUDA disponible devuelve False, es probable que exista un conflicto con la versión de PyTorch predeterminada.

Solución aplicada:

1. Desinstalar versiones previas de torch:
```powershell
pip uninstall torch torchvision torchaudio -y
```
2. Reinstalar la versión compatible con CUDA 12.1:
```powershell
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```
