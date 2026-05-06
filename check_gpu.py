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