import cv2
import numpy as np
import math
import os

# --- Configurações do Vídeo ---
largura, altura = 640, 480
fps = 60
duracao_por_trajetoria = 10  # segundos por trajetória
frames_por_traj = fps * duracao_por_trajetoria

# --- Configurações do Objeto (Bola de SSL) ---
raio_bola = 7  # Ajustado ligeiramente para melhor visibilidade sobre as linhas
cor_bola = (0, 165, 255)  # BGR: Laranja Oficial

# Cria a pasta de destino caso não exista
pasta_saida = "Videos_SSL_Projeto"
os.makedirs(pasta_saida, exist_ok=True)

# --- Definição dos 5 Cenários de Simulação ---
nomes_trajetorias = [
    "01_ssl_linear_simples.mp4",      # Movimento linear uniforme cruzando o campo
    "02_ssl_chute_acelerado.mp4",     # Simula a aceleração e curva de um passe/chute
    "03_ssl_mudancas_direcao.mp4",    # Bola rebatendo nas linhas ou tabelando
    "04_ssl_oclusao_robo.mp4",        # Bola passa por baixo de um robô (desaparece)
    "05_ssl_trajetoria_complexa.mp4"  # Movimento caótico de disputa de bola
]

num_trajetorias = len(nomes_trajetorias)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

def criar_campo_ssl(largura, altura):
    """Gera uma imagem estática do campo de SSL com marcações regulamentares."""
    # Fundo Verde Carpete
    campo = np.full((altura, largura, 3), (34, 139, 34), dtype=np.uint8)
    
    cor_linha = (255, 255, 255)
    espessura = 2
    
    # Margens do campo para não colar nas bordas da janela
    m_x, m_y = 40, 30
    xf_min, xf_max = m_x, largura - m_x
    yf_min, yf_max = m_y, altura - m_y
    cx, cy = largura // 2, altura // 2
    
    # 1. Linhas de Contorno Principais
    cv2.rectangle(campo, (xf_min, yf_min), (xf_max, yf_max), cor_linha, espessura)
    
    # 2. Linha Central
    cv2.line(campo, (cx, yf_min), (cx, yf_max), cor_linha, espessura)
    
    # 3. Círculo Central
    raio_central = 60
    cv2.circle(campo, (cx, cy), raio_central, cor_linha, espessura)
    cv2.circle(campo, (cx, cy), 3, cor_linha, -1) # Ponto central de saída
    
    # 4. Áreas de Penalidade (Retangulares conforme padrão SSL)
    largura_area = 60
    altura_area = 140
    # Esquerda
    cv2.rectangle(campo, (xf_min, cy - altura_area//2), (xf_min + largura_area, cy + altura_area//2), cor_linha, espessura)
    # Direita
    cv2.rectangle(campo, (xf_max - largura_area, cy - altura_area//2), (xf_max, cy + altura_area//2), cor_linha, espessura)
    
    # 5. Gols (Bases projetadas para fora do campo)
    largura_gol = 15
    altura_gol = 70
    # Gol Esquerdo
    cv2.rectangle(campo, (xf_min - largura_gol, cy - altura_gol//2), (xf_min, cy + altura_gol//2), cor_linha, espessura)
    # Gol Direito
    cv2.rectangle(campo, (xf_max, cy - altura_gol//2), (xf_max + largura_gol, cy + altura_gol//2), cor_linha, espessura)
    
    return campo

def obter_posicao(indice, frame_atual, total_frames):
    """Retorna a coordenada (x, y) da bola com base na trajetória selecionada."""
    t = frame_atual / total_frames 
    cx, cy = largura / 2, altura / 2
    
    if indice == 0:  # 1. LINEAR SIMPLES
        x = 60 + t * (largura - 120)
        y = 50 + t * (altura - 100)
        
    elif indice == 1:  # 2. CHUTE ACELERADO
        t_mod = t ** 2  # Simula aceleração de um chute de indução
        x = 60 + t_mod * (largura - 120)
        y = cy - 80 * math.sin(t * math.pi)
        
    elif indice == 2:  # 3. MUDANÇAS DE DIREÇÃO (Tabela)
        if t < 0.25:
            x, y = 60 + t * 4 * (largura - 120), 80
        elif t < 0.5:
            x, y = largura - 60, 80 + (t - 0.25) * 4 * (altura - 160)
        elif t < 0.75:
            x, y = largura - 60 - (t - 0.5) * 4 * (largura - 120), altura - 80
        else:
            x, y = 60, altura - 80 - (t - 0.75) * 4 * (altura - 160)
            
    elif indice == 3:  # 4. COM OCLUSÃO (Bola passa por baixo de um robô)
        x = 60 + t * (largura - 120)
        y = cy + 30 * math.sin(t * 2 * math.pi)
        # Oclusão simulada entre 35% e 55% do tempo de trajetória
        if 0.35 < t < 0.55:
            return None, None
            
    elif indice == 4:  # 5. TRAJETÓRIA COMPLEXA
        x = cx + 220 * math.sin(t * 5 * math.pi) * math.cos(t * 2 * math.pi)
        y = cy + 160 * math.cos(t * 4 * math.pi) * math.sin(t * 1.5 * math.pi)
        
    else:
        x, y = cx, cy

    # Garante que a bola não saia dos limites físicos da janela gráfica
    x = max(raio_bola, min(largura - raio_bola, x))
    y = max(raio_bola, min(altura - raio_bola, y))
    return int(x), int(y)

def desenhar_com_sombra(img, x, y, cor, raio, luz_x, luz_y):
    """Desenha a bola com um efeito de sombra projetada realista."""
    if x is None or y is None:
        return
    # Proporção do deslocamento da sombra baseado na posição da iluminação simulada
    deslocamento_sombra_x = (x - luz_x) * 0.05
    deslocamento_sombra_y = (y - luz_y) * 0.05
    sombra_x = int(x + deslocamento_sombra_x)
    sombra_y = int(y + deslocamento_sombra_y)
    
    # Desenha a sombra (preto transparente/suave simulado)
    cv2.circle(img, (sombra_x, sombra_y), int(raio * 1.05), (20, 50, 20), -1)
    # Desenha a bola laranja por cima
    cv2.circle(img, (x, y), raio, cor, -1)


# --- LOOP PRINCIPAL DE GERAÇÃO ---
print("=" * 70)
print(f"Gerando {num_trajetorias} vídeos de simulação no ambiente SSL RoboCup")
print("=" * 70)

# Renderiza o mapa base apenas uma vez para economizar processamento
campo_base = criar_campo_ssl(largura, altura)
luz_x, luz_y = largura // 2, -50  # Posição da luz para projeção da sombra

for traj_idx in range(num_trajetorias):
    nome_arquivo = os.path.join(pasta_saida, nomes_trajetorias[traj_idx])
    out = cv2.VideoWriter(nome_arquivo, fourcc, fps, (largura, altura))
    
    print(f"[{traj_idx + 1}/{num_trajetorias}] Processando: {nomes_trajetorias[traj_idx]}")
    
    for frame_idx in range(frames_por_traj):
        # Cria o frame a partir da cópia limpa do campo SSL
        imagem = campo_base.copy()
        
        # Calcula as coordenadas da bola no frame atual
        x, y = obter_posicao(traj_idx, frame_idx, frames_por_traj)
        
        # Renderiza a bola (se não estiver em estado de oclusão)
        desenhar_com_sombra(imagem, x, y, cor_bola, raio_bola, luz_x, luz_y)
        
        # Grava o frame no arquivo de vídeo
        out.write(imagem)
        
    out.release()
    print("  ✓ Vídeo exportado com sucesso.\n")

print("=" * 70)
print(f"✅ Concluído! Todos os 5 cenários SSL foram gerados na pasta '{pasta_saida}'.")
print("=" * 70)