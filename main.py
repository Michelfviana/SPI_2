import cv2
import numpy as np

# Função para desenhar os pontos e linhas
def desenhar_pontos(img, pontos, cor=(0, 255, 0), espessura=2):
    for i, ponto in enumerate(pontos):
        cv2.circle(img, (int(ponto[0]), int(ponto[1])), 5, cor, -1)  # Desenha os pontos
        if i > 0:
            cv2.line(img, (int(pontos[i-1][0]), int(pontos[i-1][1])), (int(ponto[0]), int(ponto[1])), cor, espessura)
    # Conecta o último ponto com o primeiro para fechar a figura
    cv2.line(img, (int(pontos[-1][0]), int(pontos[-1][1])), (int(pontos[0][0]), int(pontos[0][1])), cor, espessura)

# Funções de transformação geométrica
def transladar(pontos, tx, ty):
    matriz_translacao = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    pontos_homogeneos = np.hstack([pontos, np.ones((pontos.shape[0], 1))])
    pontos_transladados = matriz_translacao @ pontos_homogeneos.T
    return pontos_transladados[:2].T

def rotacionar(pontos, angulo):
    rad = np.deg2rad(angulo)
    matriz_rotacao = np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])
    pontos_homogeneos = np.hstack([pontos, np.ones((pontos.shape[0], 1))])
    pontos_rotacionados = matriz_rotacao @ pontos_homogeneos.T
    return pontos_rotacionados[:2].T

def escalonar(pontos, sx, sy):
    matriz_escalonamento = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
    pontos_homogeneos = np.hstack([pontos, np.ones((pontos.shape[0], 1))])
    pontos_escalonados = matriz_escalonamento @ pontos_homogeneos.T
    return pontos_escalonados[:2].T

# Função principal
def main():
    largura, altura = 500, 500
    img = np.zeros((altura, largura, 3), dtype=np.uint8)

    while True:
        # Entrada de pontos do usuário
        num_pontos = int(input("Digite o número de pontos: "))
        pontos = []
        for i in range(num_pontos):
            x = float(input(f"Digite a coordenada x do ponto {i+1}: "))
            y = float(input(f"Digite a coordenada y do ponto {i+1}: "))
            pontos.append([x, y])
        pontos = np.array(pontos)

        while True:
            img = np.zeros((altura, largura, 3), dtype=np.uint8)
            desenhar_pontos(img, pontos)
            cv2.imshow("Transformações Geométricas", img)
            cv2.waitKey(1)

            print("Escolha a transformação geométrica:")
            print("1. Translação")
            print("2. Rotação")
            print("3. Escalonamento")
            print("4. Resetar")
            print("5. Sair")
            escolha = int(input("Digite o número da opção desejada: "))

            if escolha == 1:
                tx = float(input("Digite o valor de deslocamento em x: "))
                ty = float(input("Digite o valor de deslocamento em y: "))
                pontos = transladar(pontos, tx, ty)
            elif escolha == 2:
                angulo = float(input("Digite o ângulo de rotação em graus: "))
                pontos = rotacionar(pontos, angulo)
            elif escolha == 3:
                sx = float(input("Digite o fator de escala em x: "))
                sy = float(input("Digite o fator de escala em y: "))
                pontos = escalonar(pontos, sx, sy)
            elif escolha == 4:
                break
            elif escolha == 5:
                cv2.destroyAllWindows()
                return
            else:
                print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()