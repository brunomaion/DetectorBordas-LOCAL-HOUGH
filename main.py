import cv2
import tkinter as tk
from tkinter import Entry, filedialog
from tkinter import Label
from PIL import Image, ImageTk
import os
import numpy as np


# Função para abrir o seletor de arquivos e carregar A imagem
def selecionar_imagem():
    caminho_imagem = filedialog.askopenfilename(initialdir="imagens", 
                                                title="Selecione uma imagem",
                                                filetypes=(("Todos os arquivos de imagem", "*.bmp;*.jpg;*.jpeg;*.png;*.tiff"),
                                                           ("Arquivos BMP", "*.bmp"),
                                                           ("Arquivos JPG", "*.jpg;*.jpeg"),
                                                           ("Arquivos PNG", "*.png"),
                                                           ("Arquivos TIFF", "*.tiff")))
    if caminho_imagem:
        # Carrega A imagem em cores
        imagem = cv2.imread(caminho_imagem)
        imagem_pil = Image.fromarray(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
        imagem_tk = ImageTk.PhotoImage(imagem_pil)

        label_entrada.config(image=imagem_tk)
        label_entrada.image = imagem_tk
        label_entrada.processed_image = imagem

        label_resultado.config(image=imagem_tk)
        label_resultado.image = imagem_tk
        label_resultado.processed_image = imagem

# Função para converter A imagem para preto e branco
def processar_imagem_pb():
    if hasattr(label_entrada, 'processed_image'):
        imagem = label_entrada.processed_image
        # Converte A imagem para preto e branco
        imagem_pb = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        imagem_pil = Image.fromarray(imagem_pb)
        imagem_tk = ImageTk.PhotoImage(imagem_pil)
        label_resultado.config(image=imagem_tk)
        label_resultado.image = imagem_tk
        label_resultado.processed_image = imagem_pb
        

# Função para converter A imagem preto e branco para negativo
def processar_imagem_negativo():
    if hasattr(label_resultado, 'processed_image'):
        imagem_pb = label_resultado.processed_image
        # Converte A imagem em preto e branco para negativo
        imagem_negativo = cv2.bitwise_not(imagem_pb)
        imagem_pil = Image.fromarray(imagem_negativo)
        imagem_tk = ImageTk.PhotoImage(imagem_pil)
       
        label_resultado.config(image=imagem_tk)
        label_resultado.image = imagem_tk

        # Armazena A imagem negativa para processamento posterior
        label_resultado.processed_image = imagem_negativo


# Função para obter o limiar da imagem
def processar_imagem_limiar():
    if hasattr(label_resultado, 'processed_image'):
        imagem_pb = label_resultado.processed_image

        limiar = escala_limiar.get() 
        _, imagem_limiarizada = cv2.threshold(imagem_pb, limiar, 255, cv2.THRESH_BINARY)

        imagem_pil = Image.fromarray(imagem_limiarizada)
        imagem_tk = ImageTk.PhotoImage(imagem_pil)

        label_resultado.config(image=imagem_tk)
        label_resultado.image = imagem_tk  # Correção aqui

        label_resultado.processed_image = imagem_limiarizada

def aplicar_filtro_passa_baixa():
    if hasattr(label_resultado, 'processed_image'):
        coeficiente = escala_passa_baixa_coef.get()
        imagem_pb = label_resultado.processed_image

        # Aplica o filtro passa baixa (blur) com o coeficiente fornecido
        imagem_suavizada = cv2.blur(imagem_pb, (coeficiente, coeficiente))


        imagem_pil = Image.fromarray(imagem_suavizada)
        imagem_tk = ImageTk.PhotoImage(imagem_pil)

        label_resultado.config(image=imagem_tk)
        label_resultado.image = imagem_tk
        label_resultado.processed_image = imagem_suavizada


def processar_imagem_canny():
    if hasattr(label_resultado, 'processed_image'):
        imagem_pb = label_resultado.processed_image

        # Aplica o algoritmo de detecção de bordas Canny
        imagem_bordas = cv2.Canny(imagem_pb, 100, 200)

        imagem_pil = Image.fromarray(imagem_bordas)
        imagem_tk = ImageTk.PhotoImage(imagem_pil)

        label_resultado.config(image=imagem_tk)
        label_resultado.image = imagem_tk
        label_resultado.processed_image = imagem_bordas


def processar_imagem_sobel():
    if hasattr(label_resultado, 'processed_image'):
        imagem_pb = label_resultado.processed_image
        sobelx = cv2.Sobel(imagem_pb, cv2.CV_64F, 1, 0, ksize=3)  # Gradiente em x
        sobely = cv2.Sobel(imagem_pb, cv2.CV_64F, 0, 1, ksize=3)  # Gradiente em y

        # Calcular A magnitude das bordas
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        sobel_magnitude = np.uint8(np.absolute(sobel_magnitude))

        imagem_pil = Image.fromarray(sobel_magnitude)
        imagem_tk = ImageTk.PhotoImage(imagem_pil)

        label_resultado.config(image=imagem_tk)
        label_resultado.image = imagem_tk
        label_resultado.processed_image = sobel_magnitude

## PROC LOCAL
def processamentoLocal(original_image, Tm, A, Ta):
    sobel_x = cv2.Sobel(original_image, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(original_image, cv2.CV_64F, 0, 1)
    linhas, colunas = original_image.shape

    magnitude, angle = cv2.cartToPolar(sobel_x, sobel_y, angleInDegrees=True)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(magnitude)

    Tm = Tm * maxVal
    imagemg = np.zeros((linhas, colunas))

    for i in range(linhas):
        for j in range(colunas):
            if magnitude[i][j] > Tm:
                angulo = angle[i, j]
                if abs(A - angulo) <= Ta or abs(A - angulo - 180) <= Ta:
                    imagemg[i][j] = 255
                else:
                    imagemg[i][j] = 0
            else:
                imagemg[i][j] = 0
    return imagemg

def correcaoK(original_image, K):
    dimensions = original_image.shape
    linhas, colunas = dimensions
    imagemcorrected = original_image.copy()

    for i in range(linhas):
        j = 0
        while j < colunas:
            if original_image[i][j] == 255:
                count = 0
                k = j + 1
                while k < colunas - 1 and original_image[i][k] == 0:
                    count += 1
                    k += 1
                if k < colunas - 1 and original_image[i][k] == 255 and count <= K:
                    for fill in range(j + 1, k):
                        imagemcorrected[i][fill] = 255
                j = k
            j += 1

    return imagemcorrected

def processar_imagem_local():
    
    Tm = float(entry_Tm.get())
    A = float(entry_A.get())
    Ta = float(entry_Ta.get())
    K = int(entry_K.get())

    img = label_resultado.processed_image
    #print("TEste", img.shape)
    
    imagemh = processamentoLocal(img, Tm, A, Ta)
    
    imagemr = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    imagemv = processamentoLocal(imagemr, Tm, A, Ta)

    imagemv = cv2.rotate(imagemv, cv2.ROTATE_90_COUNTERCLOCKWISE)
    imagemor = cv2.bitwise_or(imagemh, imagemv)
    corrected = correcaoK(imagemor, K)
    imagemRotacionada = cv2.rotate(imagemor, cv2.ROTATE_90_CLOCKWISE)
    corrected2 = correcaoK(imagemRotacionada, K)
    corrected2 = cv2.rotate(corrected2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    resultado = cv2.bitwise_or(corrected, corrected2)

        
    imagem_pil = Image.fromarray(resultado)
    imagem_tk = ImageTk.PhotoImage(imagem_pil)

    label_resultado.config(image=imagem_tk)
    label_resultado.image = imagem_tk
    label_resultado.processed_image = resultado 
    


##JANELAS DAS IMAGENS

# Configuração da interface gráfica usando Tkinter
janela = tk.Tk()
janela.title("Visualizador de Imagens")
janela.geometry("1100x500")

# Botão para selecionar A imagem
botao_selecionar = tk.Button(janela, text="Selecionar Imagem", command=selecionar_imagem)
botao_selecionar.place(x=10, y=10)

# Botão para converter A imagem para preto e branco
botao_pb = tk.Button(janela, text="Preto e Branco", command=processar_imagem_pb)
botao_pb.place(x=10, y=40)

# Botão para converter A imagem para negativo
botao_negativo = tk.Button(janela, text="Negativo", command=processar_imagem_negativo)
botao_negativo.place(x=10, y=70)



# Botão para obter o limiar
botao_limiar = tk.Button(janela, text="Limiar", command=processar_imagem_limiar)
botao_limiar.place(x=10, y=100)
# Etiqueta para A primeira linha do rótulo
label_1 = Label(janela, text="Coeficiente de")
label_1.place(x=10, y=130)
# Etiqueta para A segunda linha do rótulo
label_2 = Label(janela, text="Limiarização")
label_2.place(x=10, y=150)
# Escala para selecionar o coeficiente de limiarização
escala_limiar = tk.Scale(janela, from_=0, to=255, orient=tk.HORIZONTAL)
escala_limiar.set(127)  # Define um valor padrão
escala_limiar.place(x=10, y=170)

botao_filtro_passa_baixa = tk.Button(janela, text="Passa Baixa", command=aplicar_filtro_passa_baixa)
botao_filtro_passa_baixa.place(x=10, y=220)

label_3 = Label(janela, text="Tam. Kernel")
label_3.place(x=10, y=250)
# Escala para selecionar o coeficiente de limiarização
escala_passa_baixa_coef = tk.Scale(janela, from_=3, to=9, orient=tk.HORIZONTAL)
escala_passa_baixa_coef.set(1)  # Define um valor padrão
escala_passa_baixa_coef.place(x=10, y=270)



botao_canny = tk.Button(janela, text="Canny", command=processar_imagem_canny)
botao_canny.place(x=10, y=320)

botao_local = tk.Button(janela, text="Sobel", command=processar_imagem_sobel)
botao_local.place(x=60, y=320)

botao_local = tk.Button(janela, text="Local", command=processar_imagem_local)
botao_local.place(x=10, y=380)

# Labels e campos de entrada para Tm, A, Ta e K
label_Tm = Label(janela, text="Tm:")
label_Tm.place(x=10, y=410)
entry_Tm = Entry(janela)
entry_Tm.place(x=50, y=410)
entry_Tm.insert(0, "0.3") 

label_A = Label(janela, text="A:")
label_A.place(x=10, y=440)
entry_A = Entry(janela)
entry_A.place(x=50, y=440)
entry_A.insert(0, "90") 

label_Ta = Label(janela, text="Ta:")
label_Ta.place(x=10, y=470)
entry_Ta = Entry(janela)
entry_Ta.place(x=50, y=470)
entry_Ta.insert(0, "45") 

label_K = Label(janela, text="K:")
label_K.place(x=10, y=500)
entry_K = Entry(janela)
entry_K.place(x=50, y=500)
entry_K.insert(0, "25") 





botao_local = tk.Button(janela, text="Global", command=processar_imagem_local)
botao_local.place(x=10, y=530)

label_Global = Label(janela, text="Lim:")
label_Global.place(x=10, y=560)
entry_K_Global = Entry(janela)
entry_K_Global.place(x=50, y=560)
entry_K_Global.insert(0, "25") 


# Label para exibir A imagem de entrada e SAIDA
label_entrada = Label(janela)
label_entrada.place(x=200, y=0, width=800, height=800)
label_resultado = Label(janela)
label_resultado.place(x=1000, y=0, width=800, height=800)




# Inicia o loop principal da interface gráfica
janela.mainloop()
