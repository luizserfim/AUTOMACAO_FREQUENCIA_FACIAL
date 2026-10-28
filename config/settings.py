
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Modelo de detecção facial
MODELO_YUNET = (
    BASE_DIR
    / "modelos"
    / "face_detection_yunet_2023mar.onnx"
)

# Configurações da webcam
CAMERA_ID = 0

LARGURA_CAMERA = 640
ALTURA_CAMERA = 480

# Configurações do YuNet
LIMIAR_DETECCAO = 0.8
LIMIAR_NMS = 0.3
TOP_K = 5000

# Intervalo entre quadros (milissegundos)
INTERVALO_VIDEO = 30
