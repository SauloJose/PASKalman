import cv2
import numpy as np
import math
import os

# --- Configurações do Vídeo ---
largura, altura = 640, 480
fps = 60
duracao_por_trajetoria = 10  # segundos por trajetória
frames_por_traj = fps * duracao_por_trajetoria

# --- Configurações do Objeto ---
raio_objeto = 5
cor_alvo = (0, 165, 255)  # BGR: Laranja (Alvo principal)

# Cria a pasta de destino caso não exista
pasta_saida = "Vídeos para Projeto"
os.makedirs(pasta_saida, exist_ok=True)

# --- Nomes de Saída dos Arquivos ---
nomes_trajetorias = [
    # === PARA ANÁLISE (5) ===
    "01_linear_simples.mp4",
    "02_acelerado.mp4",
    "03_mudancas_direcao.mp4",
    "04_oclusao.mp4",
    "05_complexo.mp4",
    # === PARA TESTE (2) ===
    "06_circular_suave.mp4",
    "07_aleatorio_caotico.mp4",
    # === DESAFIOS (2) ===
    "08_multiplos_objetos_sombras.mp4",
    "09_formas_tamanhos_sombra_parcial.mp4" # NOVO DESAFIO
]

num_trajetorias = len(nomes_trajetorias)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

def obter_posicao(indice, frame_atual, total_frames):
    """Retorna a coordenada (x, y) do objeto alvo com base na trajetória."""
    t = frame_atual / total_frames 
    cx, cy = largura / 2, altura / 2
    
    if indice == 0:  # 1. LINEAR SIMPLES
        x = 50 + t * (largura - 100)
        y = 50 + t * (altura - 100)
        
    elif indice == 1:  # 2. ACELERADO
        t_mod = t ** 2
        x = 50 + t_mod * (largura - 100)
        y = cy + 100 * math.sin(t * math.pi)
        
    elif indice == 2:  # 3. MUDANÇAS DE DIREÇÃO
        if t < 0.25:
            x, y = 50 + t * 4 * (largura - 100), 100
        elif t < 0.5:
            x, y = largura - 50, 100 + (t - 0.25) * 4 * (altura - 200)
        elif t < 0.75:
            x, y = largura - 50 - (t - 0.5) * 4 * (largura - 100), altura - 100
        else:
            x, y = 50, altura - 100 - (t - 0.75) * 4 * (altura - 200)
            
    elif indice == 3:  # 4. COM OCLUSÃO
        x, y = 50 + t * (largura - 100), cy
        if 0.30 < t < 0.50:
            return None, None
            
    elif indice == 4:  # 5. COMPLEXO
        if t < 0.5:
            x, y = 50 + t * 2 * (largura - 100), 50 + t * 2 * 150
        else:
            t_seg = (t - 0.5) * 2
            x = largura - 50 - t_seg * (largura - 100)
            y = 200 + t_seg ** 2 * (altura - 200) + 50 * math.sin(t_seg * 6 * math.pi)
            
    elif indice == 5:  # 6. CIRCULAR SUAVE
        r = 150
        x, y = cx + r * math.cos(t * 2 * math.pi), cy + r * 0.7 * math.sin(t * 2 * math.pi)
        
    elif indice == 6:  # 7. ALEATÓRIO/CAÓTICO
        x = cx + 200 * math.sin(t * 13.7 * math.pi) * math.cos(t * 5.3 * math.pi)
        y = cy + 150 * math.cos(t * 11.1 * math.pi) * math.sin(t * 7.9 * math.pi)
        
    elif indice == 7:  # 8. MÚLTIPLOS OBJETOS
        x = cx + 250 * math.sin(t * 4 * math.pi)
        y = cy + 150 * math.sin(t * 2 * math.pi)
        
    elif indice == 8:  # 9. FORMAS E SOMBRA PARCIAL (Novo)
        # Movimento amplo cruzando a tela horizontalmente várias vezes (entra e sai da sombra)
        x = cx + 250 * math.cos(t * 3 * math.pi)
        y = cy + 180 * math.sin(t * 5 * math.pi)
        
    else:
        x, y = cx, cy

    if x is None or y is None:
        return None, None
    
    x = max(raio_objeto, min(largura - raio_objeto, x))
    y = max(raio_objeto, min(altura - raio_objeto, y))
    return int(x), int(y)

def desenhar_com_sombra(img, x, y, cor, raio, luz_x, luz_y):
    if x is None or y is None:
        return
    deslocamento_sombra_x = (x - luz_x) * 0.08
    deslocamento_sombra_y = (y - luz_y) * 0.08
    sombra_x = int(x + deslocamento_sombra_x)
    sombra_y = int(y + deslocamento_sombra_y)
    cv2.circle(img, (sombra_x, sombra_y), int(raio * 1.1), (60, 60, 60), -1)
    cv2.circle(img, (x, y), raio, cor, -1)


print("=" * 60)
print(f"Gerando {num_trajetorias} vídeos ({num_trajetorias - 2} originais + 2 desafios)")
print("=" * 60)

luz_x, luz_y = largura // 2, -100

for traj_idx in range(num_trajetorias):
    nome_arquivo = os.path.join(pasta_saida, nomes_trajetorias[traj_idx])
    out = cv2.VideoWriter(nome_arquivo, fourcc, fps, (largura, altura))
    
    if traj_idx < 5: prefixo = "ANÁLISE"
    elif traj_idx < 7: prefixo = "TESTE"
    else: prefixo = "DESAFIO"
        
    print(f"[{traj_idx + 1}/{num_trajetorias}] {prefixo}: {nomes_trajetorias[traj_idx]}")
    
    for frame in range(frames_por_traj):
        t = frame / frames_por_traj
        
        # Fundos das cenas
        if traj_idx == 7:
            imagem = np.full((altura, largura, 3), 160, dtype=np.uint8) # Cinza médio
        elif traj_idx == 8:
            imagem = np.full((altura, largura, 3), 220, dtype=np.uint8) # Cinza claro (para a sombra destacar)
        else:
            imagem = np.zeros((altura, largura, 3), dtype=np.uint8) # Preto
            
        x, y = obter_posicao(traj_idx, frame, frames_por_traj)
        
        # --- DESAFIO 8: Objetos com sombras ---
        if traj_idx == 7:
            d1_x = int(largura/2 + 200 * math.cos(t * 3 * math.pi))
            d1_y = int(altura/2 + 200 * math.sin(t * 3 * math.pi))
            desenhar_com_sombra(imagem, d1_x, d1_y, (255, 0, 0), raio_objeto, luz_x, luz_y)
            
            d2_x = int(50 + math.fabs(math.sin(t * 4 * math.pi)) * (largura - 100))
            d2_y = int(50 + math.fabs(math.sin(t * 2 * math.pi)) * (altura - 100))
            desenhar_com_sombra(imagem, d2_x, d2_y, (0, 200, 0), raio_objeto, luz_x, luz_y)
            
            d3_x = int(largura/2 + 150 * math.sin(t * 19 * math.pi))
            d3_y = int(altura/2 + 100 * math.cos(t * 13 * math.pi))
            desenhar_com_sombra(imagem, d3_x, d3_y, (200, 0, 150), raio_objeto, luz_x, luz_y)
            
            desenhar_com_sombra(imagem, x, y, cor_alvo, raio_objeto, luz_x, luz_y)

        # --- DESAFIO 9: Formas diferentes e zona de sombra severa ---
        elif traj_idx == 8:
            # 1. Desenha o Alvo (Círculo Laranja, Raio 5)
            if x is not None and y is not None:
                cv2.circle(imagem, (x, y), raio_objeto, cor_alvo, -1)
                
            # 2. Desenha Distrator 1 (Quadrado Azul, Tamanho Maior)
            q_x = int(50 + t * (largura - 100))
            q_y = int(altura/4 + 100 * math.sin(t * 8 * math.pi))
            lado = 15
            cv2.rectangle(imagem, (q_x - lado, q_y - lado), (q_x + lado, q_y + lado), (255, 50, 50), -1)
            
            # 3. Desenha Distrator 2 (Triângulo Verde, Tamanho Ainda Maior)
            tr_x = int(largura - 50 - t * (largura - 100))
            tr_y = int(altura*3/4 + 80 * math.cos(t * 6 * math.pi))
            tamanho_tr = 20
            # Definindo os pontos do triângulo usando NumPy
            pts_triangulo = np.array([
                [tr_x, tr_y - tamanho_tr], 
                [tr_x - tamanho_tr, tr_y + tamanho_tr], 
                [tr_x + tamanho_tr, tr_y + tamanho_tr]
            ], np.int32)
            pts_triangulo = pts_triangulo.reshape((-1, 1, 2))
            cv2.fillPoly(imagem, [pts_triangulo], (0, 180, 0))

            # 4. Aplica a Sombra Semi-Transparente na Metade Direita da tela
            # Criamos uma cópia da imagem para servir de camada de mistura (overlay)
            overlay = imagem.copy()
            
            # Desenhamos um retângulo totalmente preto na metade direita do overlay
            cv2.rectangle(overlay, (largura // 2, 0), (largura, altura), (0, 0, 0), -1)
            
            # Mesclamos o overlay com a imagem original
            # Onde o overlay é preto, a imagem resultante ficará com 40% do brilho original
            # Onde o overlay é igual à imagem (metade esquerda), ela permanece inalterada (0.6 + 0.4 = 1.0)
            cv2.addWeighted(overlay, 0.6, imagem, 0.4, 0, imagem)

        # --- OUTROS VÍDEOS ---
        else:
            if x is not None and y is not None:
                cv2.circle(imagem, (x, y), raio_objeto, cor_alvo, -1)
                
        out.write(imagem)
    
    out.release()
    print("  ✓ Salvo com sucesso\n")

print("=" * 60)
print(f"✅ Todos os {num_trajetorias} vídeos foram gerados com sucesso!")
print("=" * 60)



