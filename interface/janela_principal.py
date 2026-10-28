
import tkinter as tk
from tkinter import messagebox

import cv2
from PIL import Image, ImageTk

from src.captura import CapturaVideo
from src.detector_facial import DetectorFacial
from config.settings import INTERVALO_VIDEO


class JanelaPrincipal:

    def __init__(self, janela):

        self.janela = janela
        self.janela.title("Frequência Facial")
        self.janela.geometry("760x650")

        self.captura = CapturaVideo()
        self.detector = None

        self.camera_ativa = False
        self.deteccao_ativa = False
        self.job_video = None

        self.criar_componentes()

        self.janela.protocol(
            "WM_DELETE_WINDOW",
            self.fechar
        )

    def criar_componentes(self):

        titulo = tk.Label(
            self.janela,
            text="Sistema de Frequência Facial",
            font=("Arial", 18, "bold")
        )
        titulo.pack(pady=15)

        self.video_label = tk.Label(
            self.janela,
            text="Webcam desligada",
            bg="#202020",
            fg="white",
            width=80,
            height=25
        )
        self.video_label.pack(pady=10)

        painel = tk.Frame(self.janela)
        painel.pack(pady=10)

        self.botao_camera = tk.Button(
            painel,
            text="Iniciar câmera",
            command=self.alternar_camera,
            width=18
        )
        self.botao_camera.grid(
            row=0, column=0, padx=5
        )

        self.botao_deteccao = tk.Button(
            painel,
            text="Ativar detecção",
            command=self.alternar_deteccao,
            width=18,
            state=tk.DISABLED
        )
        self.botao_deteccao.grid(
            row=0, column=1, padx=5
        )

        self.status = tk.Label(
            self.janela,
            text="Sistema pronto.",
            font=("Arial", 11)
        )
        self.status.pack(pady=10)

    def alternar_camera(self):

        if self.camera_ativa:
            self.parar_camera()
        else:
            self.iniciar_camera()

    def iniciar_camera(self):

        try:
            self.captura.iniciar()

        except Exception as erro:
            messagebox.showerror(
                "Erro na câmera",
                str(erro)
            )
            return

        self.camera_ativa = True

        self.botao_camera.config(
            text="Parar câmera"
        )

        self.botao_deteccao.config(
            state=tk.NORMAL
        )

        self.atualizar_video()

    def alternar_deteccao(self):

        if self.deteccao_ativa:
            self.deteccao_ativa = False
            self.botao_deteccao.config(
                text="Ativar detecção"
            )
            return

        try:
            if self.detector is None:
                self.detector = DetectorFacial()

        except Exception as erro:
            messagebox.showerror(
                "Erro no YuNet",
                str(erro)
            )
            return

        self.deteccao_ativa = True

        self.botao_deteccao.config(
            text="Desativar detecção"
        )

    def atualizar_video(self):

        if not self.camera_ativa:
            return

        frame = self.captura.capturar()

        if frame is None:
            self.parar_camera()
            messagebox.showerror(
                "Erro",
                "Falha na leitura da webcam."
            )
            return

        quantidade = 0

        if self.deteccao_ativa:

            try:
                rostos = self.detector.detectar(frame)

            except Exception as erro:
                self.parar_camera()
                messagebox.showerror(
                    "Erro na detecção",
                    str(erro)
                )
                return

            if rostos is not None:
                quantidade = len(rostos)

                frame = self.detector.desenhar_rostos(
                    frame, rostos
                )

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        imagem = Image.fromarray(frame_rgb)

        imagem.thumbnail((640, 480))

        foto = ImageTk.PhotoImage(imagem)

        self.video_label.config(
            image=foto,
            text="",
            width=640,
            height=480
        )

        self.video_label.image = foto

        if self.deteccao_ativa:
            self.status.config(
                text=f"Rostos detectados: {quantidade}"
            )
        else:
            self.status.config(
                text="Câmera ativa | Detecção desligada"
            )

        self.job_video = self.janela.after(
            INTERVALO_VIDEO,
            self.atualizar_video
        )

    def parar_camera(self):

        self.camera_ativa = False
        self.deteccao_ativa = False

        if self.job_video is not None:
            self.janela.after_cancel(self.job_video)
            self.job_video = None

        self.captura.encerrar()

        self.video_label.config(
            image="",
            text="Webcam desligada",
            width=80,
            height=25
        )
        self.video_label.image = None

        self.botao_camera.config(
            text="Iniciar câmera"
        )

        self.botao_deteccao.config(
            text="Ativar detecção",
            state=tk.DISABLED
        )

        self.status.config(
            text="Câmera desligada."
        )

    def fechar(self):

        self.parar_camera()
        self.janela.destroy()
