import cv2
import tkinter as tk
from tkinter import Entry, filedialog
from tkinter import Label
from PIL import Image, ImageTk
import os
import numpy as np
import math


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


## PROC GLOBAL


# Etapa 1: Obtenha uma imagem binária com as bordas da imagem
def obter_bordas(img):
    bordas = cv2.Canny(img, 100, 200)
    return bordas

# Etapa 2: Defina como o plano \rho\theta será dividido (estrutura da matriz acumuladora)
def definir_acumuladora(img, passo_angulo=1):
    thetas = np.deg2rad(np.arange(-90.0, 90.0, passo_angulo))
    largura, altura = img.shape
    comprimento_diag = int(round(math.sqrt(largura ** 2 + altura ** 2)))
    rhos = np.linspace(-comprimento_diag, comprimento_diag, comprimento_diag * 2)
    acumuladora = np.zeros((2 * comprimento_diag, len(thetas)), dtype=np.uint8)
    return acumuladora, thetas, rhos

# Etapa 3: Aplique a parametrização aos pontos da imagem das bordas, atualizando a matriz acumuladora
def aplicar_parametrizacao(img_bordas, acumuladora, thetas, rhos):
    largura, altura = img_bordas.shape
    comprimento_diag = len(rhos) // 2
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    y_idxs, x_idxs = np.nonzero(img_bordas)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(len(thetas)):
            rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx])) + comprimento_diag
            acumuladora[rho, t_idx] += 1

    return acumuladora

# Etapa 4: Examine a matriz acumuladora em busca de células com valores elevados
def encontrar_linhas_acumuladora(acumuladora, thetas, rhos, limiar):
    linhas = []
    for r_idx in range(acumuladora.shape[0]):
        for t_idx in range(acumuladora.shape[1]):
            if acumuladora[r_idx, t_idx] > limiar:
                rho = rhos[r_idx]
                theta = thetas[t_idx]
                linhas.append((rho, theta))
    return linhas

# Etapa 5: Examine a relação (principalmente as de continuidade) entre os pixels oriundos das células escolhidas
def desenhar_linhas(img, linhas):
    # Verifica se a imagem está em escala de cinza e converte se necessário
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_linhas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    comprimento_diag = int(round(math.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)))
    for rho, theta in linhas:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + comprimento_diag * (-b))
        y1 = int(y0 + comprimento_diag * (a))
        x2 = int(x0 - comprimento_diag * (-b))
        y2 = int(y0 - comprimento_diag * (a))
        cv2.line(img_linhas, (x1, y1), (x2, y2), (0, 0, 255), 1)
    return img_linhas

def processar_imagem_global():
    img = label_resultado.processed_image
    limiar_valores_elevados = int(entry_K_Global.get())   # Defina o limiar aqui
    # Etapa 1: Obtenha uma imagem binária com as bordas da imagem
    bordas = obter_bordas(img)
    # Etapa 2: Defina como o plano rho e theta será dividido
    acumuladora, thetas, rhos = definir_acumuladora(bordas)
    # Etapa 3: Aplique a parametrização aos pontos da imagem das bordas, atualizando a matriz acumuladora
    acumuladora = aplicar_parametrizacao(bordas, acumuladora, thetas, rhos)
    # Etapa 4: Examine a matriz acumuladora em busca de células com valores elevados
    linhas_detectadas = encontrar_linhas_acumuladora(acumuladora, thetas, rhos, limiar=limiar_valores_elevados)
    # Etapa 5: Examine a relação entre os pixels oriundos das células escolhidas
    img_com_linhas = desenhar_linhas(img, linhas_detectadas)

    imagem_pil = Image.fromarray(img_com_linhas)
    imagem_tk = ImageTk.PhotoImage(imagem_pil)

    label_resultado.config(image=imagem_tk)
    label_resultado.image = imagem_tk
    label_resultado.processed_image = img_com_linhas 



##JANELAS DAS IMAGENS

# Configuração da interface gráfica usando Tkinter
janela = tk.Tk()
janela.title("Visualizador de Imagens")
janela.geometry("1100x900")

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



botao_local = tk.Button(janela, text="Global", command=processar_imagem_global)
botao_local.place(x=10, y=530)

label_Global = Label(janela, text="Lim:")
label_Global.place(x=10, y=560)
entry_K_Global = Entry(janela)
entry_K_Global.place(x=50, y=560)
entry_K_Global.insert(0, "100") 


# Label para exibir A imagem de entrada e SAIDA
label_entrada = Label(janela)
label_entrada.place(x=200, y=0, width=800, height=800)
label_resultado = Label(janela)
label_resultado.place(x=1000, y=0, width=800, height=800)


# Inicia o loop principal da interface gráfica
janela.mainloop()
