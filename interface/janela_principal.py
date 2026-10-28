
import tkinter as tk
from tkinter import messagebox, filedialog

import cv2
import numpy as np

from PIL import Image, ImageTk

from src.captura import CapturaVideo
from src.detector_facial import DetectorFacial
from src.reconhecimento import ReconhecimentoFacial
from config.settings import INTERVALO_VIDEO


class JanelaPrincipal:

    def __init__(self, janela):

        self.janela = janela
        self.janela.title("Frequência Facial")
        self.janela.geometry("760x650")
        self.janela.resizable(False, False)

        # Módulos
        self.captura = CapturaVideo()
        self.detector = None
        self.reconhecedor = None

        # Controle da webcam
        self.camera_ativa = False
        self.deteccao_ativa = False
        self.job_video = None
        self.ultimo_frame = None

        # Controle da validação humana
        self.janela_validacao = None

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

        # Área da webcam
        self.video_label = tk.Label(
            self.janela,
            text="Webcam desligada",
            bg="#202020",
            fg="white",
            width=80,
            height=25
        )
        self.video_label.pack(pady=10)

        # Painel dos botões
        painel = tk.Frame(self.janela)
        painel.pack(pady=10)

        self.botao_camera = tk.Button(
            painel,
            text="Iniciar câmera",
            command=self.alternar_camera,
            width=20
        )
        self.botao_camera.grid(
            row=0,
            column=0,
            padx=5,
            pady=5
        )

        self.botao_deteccao = tk.Button(
            painel,
            text="Ativar detecção",
            command=self.alternar_deteccao,
            width=20,
            state=tk.DISABLED
        )
        self.botao_deteccao.grid(
            row=0,
            column=1,
            padx=5,
            pady=5
        )

        self.botao_comparar = tk.Button(
            painel,
            text="Comparar referência",
            command=self.comparar_referencia,
            width=25,
            state=tk.DISABLED
        )
        self.botao_comparar.grid(
            row=1,
            column=0,
            columnspan=2,
            pady=10
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

        self.botao_comparar.config(
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

        self.job_video = None

        frame = self.captura.capturar()

        if frame is None:

            self.parar_camera()

            messagebox.showerror(
                "Erro",
                "Falha na leitura da webcam."
            )
            return

        # Guarda o frame sem desenhos
        self.ultimo_frame = frame.copy()

        quantidade = 0

        if self.deteccao_ativa:

            try:
                rostos = self.detector.detectar(frame)

                if rostos is not None:

                    quantidade = len(rostos)

                    frame = self.detector.desenhar_rostos(
                        frame,
                        rostos
                    )

            except Exception as erro:

                self.parar_camera()

                messagebox.showerror(
                    "Erro na detecção",
                    str(erro)
                )
                return

        # Converte para exibição no Tkinter
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

    def carregar_imagem(self, caminho):

        # Leitura compatível com caminhos Unicode
        dados = np.fromfile(
            caminho,
            dtype=np.uint8
        )

        imagem = cv2.imdecode(
            dados,
            cv2.IMREAD_COLOR
        )

        if imagem is None:
            raise ValueError(
                "Não foi possível carregar a imagem."
            )

        return imagem

    def redimensionar_imagem(self, imagem):

        altura, largura = imagem.shape[:2]
        limite = 1280

        maior_dimensao = max(
            altura,
            largura
        )

        if maior_dimensao <= limite:
            return imagem

        escala = limite / maior_dimensao

        return cv2.resize(
            imagem,
            None,
            fx=escala,
            fy=escala,
            interpolation=cv2.INTER_AREA
        )

    def comparar_referencia(self):

        if not self.camera_ativa:

            messagebox.showwarning(
                "Aviso",
                "Inicie a webcam primeiro."
            )
            return

        if self.janela_validacao is not None:
            return

        caminho = filedialog.askopenfilename(
            title="Selecionar imagem de referência",
            filetypes=[
                ("Imagens", "*.jpg *.jpeg *.png")
            ]
        )

        if not caminho:
            return

        try:

            if self.detector is None:
                self.detector = DetectorFacial()

            if self.reconhecedor is None:
                self.reconhecedor = ReconhecimentoFacial()

            # Carrega a foto de referência
            imagem_referencia = self.carregar_imagem(
                caminho
            )

            altura, largura = imagem_referencia.shape[:2]

            print("\n===== DIAGNÓSTICO FACIAL =====")
            print(f"Arquivo: {caminho}")
            print(f"Dimensões: {largura} x {altura}")

            imagem_referencia = (
                self.redimensionar_imagem(
                    imagem_referencia
                )
            )

            # Captura um frame limpo da webcam
            if self.ultimo_frame is None:
                raise RuntimeError(
                    "Nenhum quadro disponível da webcam."
                )

            frame = self.ultimo_frame.copy()

            # Detecta rostos nas duas imagens
            rostos_referencia = self.detector.detectar(
                imagem_referencia
            )

            rostos_camera = self.detector.detectar(
                frame
            )

            quantidade_referencia = (
                0 if rostos_referencia is None
                else len(rostos_referencia)
            )

            quantidade_camera = (
                0 if rostos_camera is None
                else len(rostos_camera)
            )

            print(
                f"Rostos na referência: "
                f"{quantidade_referencia}"
            )

            print(
                f"Rostos na webcam: "
                f"{quantidade_camera}"
            )

            if rostos_referencia is not None:

                for indice, rosto in enumerate(
                    rostos_referencia,
                    start=1
                ):

                    confianca = float(rosto[-1]) * 100

                    print(
                        f"Referência - rosto {indice}: "
                        f"{confianca:.2f}%"
                    )

            if rostos_camera is not None:

                for indice, rosto in enumerate(
                    rostos_camera,
                    start=1
                ):

                    confianca = float(rosto[-1]) * 100

                    print(
                        f"Webcam - rosto {indice}: "
                        f"{confianca:.2f}%"
                    )

            print("================================\n")

            # Valida a quantidade de rostos
            if quantidade_referencia != 1:

                messagebox.showwarning(
                    "Referência inválida",
                    f"Rostos detectados na referência: "
                    f"{quantidade_referencia}\n\n"
                    "É necessário exatamente um rosto."
                )
                return

            if quantidade_camera != 1:

                messagebox.showwarning(
                    "Captura inválida",
                    f"Rostos detectados na webcam: "
                    f"{quantidade_camera}\n\n"
                    "É necessário exatamente um rosto."
                )
                return

            # Extrai características faciais
            caracteristicas_referencia = (
                self.reconhecedor.extrair_caracteristicas(
                    imagem_referencia,
                    rostos_referencia[0]
                )
            )

            caracteristicas_camera = (
                self.reconhecedor.extrair_caracteristicas(
                    frame,
                    rostos_camera[0]
                )
            )

            # Compara os rostos
            similaridade, correspondencia = (
                self.reconhecedor.comparar(
                    caracteristicas_referencia,
                    caracteristicas_camera
                )
            )

            print(
                f"Similaridade SFace: "
                f"{similaridade:.4f}"
            )

            # Solicita validação humana
            self.mostrar_validacao(
                imagem_referencia,
                frame,
                similaridade,
                correspondencia
            )

        except Exception as erro:

            messagebox.showerror(
                "Erro no reconhecimento facial",
                str(erro)
            )

    def preparar_preview(self, imagem):

        # Converte do OpenCV para Pillow
        imagem_rgb = cv2.cvtColor(
            imagem,
            cv2.COLOR_BGR2RGB
        )

        preview = Image.fromarray(imagem_rgb)

        preview.thumbnail((280, 280))

        return ImageTk.PhotoImage(preview)

    def mostrar_validacao(
        self,
        imagem_referencia,
        frame,
        similaridade,
        correspondencia
    ):

        # Janela separada para conferência humana
        validacao = tk.Toplevel(self.janela)

        self.janela_validacao = validacao

        validacao.title("Validação facial")
        validacao.geometry("680x510")
        validacao.resizable(False, False)
        validacao.transient(self.janela)

        self.botao_comparar.config(
            state=tk.DISABLED
        )

        titulo = tk.Label(
            validacao,
            text="Validação de correspondência facial",
            font=("Arial", 16, "bold")
        )
        titulo.pack(pady=15)

        # Apresenta as duas imagens
        painel_imagens = tk.Frame(validacao)
        painel_imagens.pack(pady=10)

        foto_referencia = self.preparar_preview(
            imagem_referencia
        )

        foto_camera = self.preparar_preview(
            frame
        )

        coluna_referencia = tk.Frame(painel_imagens)
        coluna_referencia.grid(
            row=0,
            column=0,
            padx=15
        )

        coluna_camera = tk.Frame(painel_imagens)
        coluna_camera.grid(
            row=0,
            column=1,
            padx=15
        )

        tk.Label(
            coluna_referencia,
            text="Imagem de referência",
            font=("Arial", 11, "bold")
        ).pack(pady=5)

        label_referencia = tk.Label(
            coluna_referencia,
            image=foto_referencia
        )
        label_referencia.pack()

        tk.Label(
            coluna_camera,
            text="Captura da webcam",
            font=("Arial", 11, "bold")
        ).pack(pady=5)

        label_camera = tk.Label(
            coluna_camera,
            image=foto_camera
        )
        label_camera.pack()

        # Mantém as imagens na memória
        validacao.fotos = (
            foto_referencia,
            foto_camera
        )

        tk.Label(
            validacao,
            text=f"Similaridade SFace: {similaridade:.4f}",
            font=("Arial", 13, "bold")
        ).pack(pady=8)

        resultado = (
            "Acima do limiar experimental"
            if correspondencia
            else "Abaixo do limiar experimental"
        )

        tk.Label(
            validacao,
            text=resultado,
            font=("Arial", 11)
        ).pack()

        tk.Label(
            validacao,
            text=(
                "Após verificar as imagens, "
                "você confirma que são da mesma pessoa?"
            ),
            font=("Arial", 10)
        ).pack(pady=15)

        painel_botoes = tk.Frame(validacao)
        painel_botoes.pack()

        def encerrar_validacao():

            if validacao.winfo_exists():
                validacao.grab_release()
                validacao.destroy()

            self.janela_validacao = None

            if self.camera_ativa:
                self.botao_comparar.config(
                    state=tk.NORMAL
                )

        def confirmar():

            encerrar_validacao()

            self.status.config(
                text="Correspondência confirmada manualmente."
            )

            messagebox.showinfo(
                "Validação concluída",
                "Correspondência confirmada pelo operador.\n\n"
                "Nenhuma frequência foi registrada."
            )

        def rejeitar():

            encerrar_validacao()

            self.status.config(
                text="Correspondência rejeitada manualmente."
            )

            messagebox.showinfo(
                "Validação concluída",
                "Correspondência não confirmada.\n\n"
                "Nenhuma frequência foi registrada."
            )

        tk.Button(
            painel_botoes,
            text="Confirmar correspondência",
            command=confirmar,
            bg="#168457",
            fg="white",
            width=24
        ).grid(
            row=0,
            column=0,
            padx=8
        )

        tk.Button(
            painel_botoes,
            text="Rejeitar correspondência",
            command=rejeitar,
            width=24
        ).grid(
            row=0,
            column=1,
            padx=8
        )

        validacao.protocol(
            "WM_DELETE_WINDOW",
            encerrar_validacao
        )

        validacao.grab_set()

    def parar_camera(self):

        self.camera_ativa = False
        self.deteccao_ativa = False
        self.ultimo_frame = None

        if self.job_video is not None:

            self.janela.after_cancel(
                self.job_video
            )

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

        self.botao_comparar.config(
            state=tk.DISABLED
        )

        self.status.config(
            text="Câmera desligada."
        )

    def fechar(self):

        if self.janela_validacao is not None:

            if self.janela_validacao.winfo_exists():
                self.janela_validacao.destroy()

            self.janela_validacao = None

        self.parar_camera()
        self.janela.destroy()
