
import tkinter as tk

from interface.janela_principal import JanelaPrincipal


def main():
    janela = tk.Tk()

    JanelaPrincipal(janela)

    janela.mainloop()


if __name__ == "__main__":
    main()
