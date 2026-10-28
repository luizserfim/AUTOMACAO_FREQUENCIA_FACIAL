
import cv2

from config.settings import (
    MODELO_YUNET,
    LIMIAR_DETECCAO,
    LIMIAR_NMS,
    TOP_K,
)


class DetectorFacial:

    def __init__(self):

        if not MODELO_YUNET.is_file():
            raise FileNotFoundError(
                f"Modelo YuNet não encontrado: {MODELO_YUNET}"
            )

        self.detector = cv2.FaceDetectorYN.create(
            str(MODELO_YUNET),
            "",
            (320, 320),
            LIMIAR_DETECCAO,
            LIMIAR_NMS,
            TOP_K
        )

    def detectar(self, frame):

        altura, largura = frame.shape[:2]

        self.detector.setInputSize(
            (largura, altura)
        )

        _, rostos = self.detector.detect(frame)

        return rostos

    def desenhar_rostos(self, frame, rostos):

        if rostos is None:
            return frame

        for rosto in rostos:

            # Coordenadas do rosto detectado
            x, y, largura, altura = rosto[:4].astype(int)

            # Confiança convertida em porcentagem
            confianca = float(rosto[-1]) * 100

            # Desenha o retângulo ao redor do rosto
            cv2.rectangle(
                frame,
                (x, y),
                (x + largura, y + altura),
                (0, 255, 0),
                2
            )

            # Texto com a porcentagem
            texto = f"Rosto: {confianca:.1f}%"

            fonte = cv2.FONT_HERSHEY_SIMPLEX
            escala = 0.6
            espessura = 2

            # Calcula o tamanho do texto
            (largura_texto, altura_texto), _ = (
                cv2.getTextSize(
                    texto,
                    fonte,
                    escala,
                    espessura
                )
            )

            # Posiciona o texto dentro da imagem
            altura_frame, largura_frame = frame.shape[:2]

            texto_x = max(
                0,
                min(x, largura_frame - largura_texto - 10)
            )

            texto_y = (
                y - 10
                if y >= altura_texto + 15
                else y + altura_texto + 15
            )

            texto_y = max(
                altura_texto + 5,
                min(texto_y, altura_frame - 5)
            )

            # Fundo verde para destacar a porcentagem
            cv2.rectangle(
                frame,
                (texto_x, texto_y - altura_texto - 5),
                (texto_x + largura_texto + 10, texto_y + 5),
                (0, 150, 0),
                -1
            )

            # Escreve a porcentagem em branco
            cv2.putText(
                frame,
                texto,
                (texto_x + 5, texto_y),
                fonte,
                escala,
                (255, 255, 255),
                espessura
            )

        return frame
