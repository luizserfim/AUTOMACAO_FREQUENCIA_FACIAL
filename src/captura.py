
import cv2

from config.settings import (
    CAMERA_ID,
    LARGURA_CAMERA,
    ALTURA_CAMERA,
)


class CapturaVideo:

    def __init__(self):
        self.camera = None

    def iniciar(self):
        if self.camera is not None:
            self.encerrar()

        self.camera = cv2.VideoCapture(
            CAMERA_ID,
            cv2.CAP_DSHOW
        )

        if not self.camera.isOpened():
            self.encerrar()
            raise RuntimeError(
                "Não foi possível acessar a webcam."
            )

        self.camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            LARGURA_CAMERA
        )

        self.camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            ALTURA_CAMERA
        )

    def capturar(self):
        if self.camera is None:
            return None

        sucesso, frame = self.camera.read()

        if not sucesso:
            return None

        return frame

    def encerrar(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None
