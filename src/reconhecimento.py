
import cv2

from pathlib import Path

from config.settings import (
    MODELO_SFACE,
    LIMIAR_SIMILARIDADE,
)


class ReconhecimentoFacial:

    def __init__(self):

        # Obtém o caminho configurado para o modelo
        self.caminho_modelo = Path(MODELO_SFACE)

        # Verifica se o arquivo existe
        if not self.caminho_modelo.is_file():
            raise FileNotFoundError(
                "Modelo SFace não encontrado!\n\n"
                f"Caminho esperado:\n{self.caminho_modelo}\n\n"
                "Verifique se o arquivo foi baixado "
                "e colocado na pasta 'modelos'."
            )

        # Verifica se o arquivo está vazio
        if self.caminho_modelo.stat().st_size == 0:
            raise ValueError(
                "O arquivo do modelo SFace está vazio."
            )

        # Inicializa o modelo de reconhecimento facial
        try:
            self.reconhecedor = cv2.FaceRecognizerSF.create(
                str(self.caminho_modelo),
                ""
            )

        except cv2.error as erro:
            raise RuntimeError(
                "Não foi possível carregar o modelo SFace. "
                "O arquivo pode estar corrompido ou "
                "ser incompatível com o OpenCV."
            ) from erro

    def extrair_caracteristicas(self, imagem, rosto):

        if imagem is None or rosto is None:
            raise ValueError(
                "Imagem ou dados faciais inválidos."
            )

        # Alinha o rosto usando os pontos detectados pelo YuNet
        rosto_alinhado = self.reconhecedor.alignCrop(
            imagem,
            rosto
        )

        # Extrai as características faciais
        caracteristicas = self.reconhecedor.feature(
            rosto_alinhado
        )

        return caracteristicas

    def comparar(self, caracteristicas_1, caracteristicas_2):

        if caracteristicas_1 is None or caracteristicas_2 is None:
            raise ValueError(
                "Características faciais inválidas."
            )

        # Calcula a similaridade de cosseno
        similaridade = self.reconhecedor.match(
            caracteristicas_1,
            caracteristicas_2,
            cv2.FaceRecognizerSF_FR_COSINE
        )

        similaridade = float(similaridade)

        # Verifica se a pontuação atingiu o limiar
        correspondencia = (
            similaridade >= LIMIAR_SIMILARIDADE
        )

        return similaridade, correspondencia
