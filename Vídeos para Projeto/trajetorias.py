import cv2
import numpy as np
import math

# --- Configurações do Vídeo ---
largura, altura = 640, 480
fps = 30
duracao_por_trajetoria = 10  # segundos por trajetória
frames_por_traj = fps * duracao_por_trajetoria

# --- Configurações do Objeto ---
raio_objeto = 15
cor_objeto = (0, 165, 255)  # BGR: Laranja para detectar melhor

# --- Nomes de Saída dos Arquivos ---
# 5 para análise + 2 para teste
nomes_trajetorias = [
    # === PARA ANÁLISE (5) ===
    "01_linear_simples.mp4",           # Movimento linear direto
    "02_acelerado.mp4",                # Começa devagar, acelera
    "03_mudancas_direcao.mp4",         # Zigzag/mudanças rápidas
    "04_oclusao.mp4",                  # Linear com frames perdidos no meio
    "05_complexo.mp4",                 # Combinação: linear + aceleração + curvas
    # === PARA TESTE (2) ===
    "06_circular_suave.mp4",           # Movimento circular perfeito
    "07_aleatorio_caotico.mp4"         # Movimento aleatório/errático
]

num_trajetorias = len(nomes_trajetorias)

# Codec mp4v para salvar em .mp4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

def obter_posicao(indice, frame_atual, total_frames):
    """
    Retorna a coordenada (x, y) do objeto com base na trajetória selecionada.
    't' varia de 0.0 a 1.0 ao longo do vídeo.
    """
    t = frame_atual / total_frames 
    cx, cy = largura / 2, altura / 2
    
    if indice == 0:  # 1. LINEAR SIMPLES
        # Reta de (50, 50) até (590, 430)
        x = 50 + t * (largura - 100)
        y = 50 + t * (altura - 100)
        
    elif indice == 1:  # 2. ACELERADO
        # Começa devagar, depois acelera (quadrático)
        t_mod = t ** 2
        x = 50 + t_mod * (largura - 100)
        y = cy + 100 * math.sin(t * math.pi)  # Pequena oscilação
        
    elif indice == 2:  # 3. MUDANÇAS DE DIREÇÃO (Zigzag)
        # Movimento em zigzag - muda de direção a cada 25% do tempo
        if t < 0.25:
            x = 50 + t * 4 * (largura - 100)
            y = 100
        elif t < 0.5:
            x = largura - 50
            y = 100 + (t - 0.25) * 4 * (altura - 200)
        elif t < 0.75:
            x = largura - 50 - (t - 0.5) * 4 * (largura - 100)
            y = altura - 100
        else:
            x = 50
            y = altura - 100 - (t - 0.75) * 4 * (altura - 200)
            
    elif indice == 3:  # 4. COM OCLUSÃO
        # Movimento linear, mas com "desaparecimento" no meio (frames 100-130 aprox)
        x = 50 + t * (largura - 100)
        y = cy
        
        # Calcula se está na zona de oclusão (entre 30% e 50% do tempo)
        if 0.30 < t < 0.50:
            # Retorna None para simular oclusão (será tratado no desenho)
            return None, None
        
    elif indice == 4:  # 5. COMPLEXO (Combinação)
        # Primeira metade: movimento linear rápido (0.0 - 0.5)
        if t < 0.5:
            x = 50 + t * 2 * (largura - 100)
            y = 50 + t * 2 * 150  # Só desce um pouco
        # Segunda metade: movimento com aceleração + curva (0.5 - 1.0)
        else:
            t_seg = (t - 0.5) * 2  # Renormaliza para 0-1
            x = largura - 50 - t_seg * (largura - 100)  # Volta
            y = 200 + t_seg ** 2 * (altura - 200)  # Acelera caindo
            # Adiciona oscilação
            y += 50 * math.sin(t_seg * 6 * math.pi)
        
    elif indice == 5:  # 6. CIRCULAR SUAVE
        # Círculo perfeito com velocidade angular constante
        r = 150
        x = cx + r * math.cos(t * 2 * math.pi)
        y = cy + r * 0.7 * math.sin(t * 2 * math.pi)  # Um pouco achatado
        
    elif indice == 6:  # 7. ALEATÓRIO/CAÓTICO
        # Usa harmônicos não-sincronizados para criar movimento errático
        x = cx + 200 * math.sin(t * 13.7 * math.pi) * math.cos(t * 5.3 * math.pi)
        y = cy + 150 * math.cos(t * 11.1 * math.pi) * math.sin(t * 7.9 * math.pi)
        
    else:
        x, y = cx, cy

    # Se retornar None (oclusão), deixa como está
    if x is None or y is None:
        return None, None
    
    # Limita as coordenadas
    x = max(raio_objeto, min(largura - raio_objeto, x))
    y = max(raio_objeto, min(altura - raio_objeto, y))

    return int(x), int(y)


print("=" * 60)
print("Gerando 7 vídeos de teste (5 para análise + 2 para teste)")
print("=" * 60)
print(f"Resolução: {largura}×{altura} | FPS: {fps} | Duração: {duracao_por_trajetoria}s\n")

# Loop principal: renderizando cada trajetória
for traj_idx in range(num_trajetorias):
    nome_arquivo = nomes_trajetorias[traj_idx]
    nome_arquivo = "Vídeos para Projeto/"+nome_arquivo
    # Inicia o VideoWriter
    out = cv2.VideoWriter(nome_arquivo, fourcc, fps, (largura, altura))
    
    prefixo = "ANÁLISE" if traj_idx < 5 else "TESTE"
    print(f"[{traj_idx + 1}/7] {prefixo}: {nome_arquivo}")
    
    for frame in range(frames_por_traj):
        # Cria frame preto (fundo)
        imagem = np.zeros((altura, largura, 3), dtype=np.uint8)
        
        # Calcula posição
        x, y = obter_posicao(traj_idx, frame, frames_por_traj)
        
        # Se não está em oclusão, desenha o objeto
        if x is not None and y is not None:
            cv2.circle(imagem, (x, y), raio_objeto, cor_objeto, -1)
        # Se está em oclusão (trajetória 4), frame fica preto (objeto invisível)
        
        # Salva o frame
        out.write(imagem)
    
    # Libera arquivo
    out.release()
    print(f"  ✓ Salvo com sucesso\n")

print("=" * 60)
print("✅ Todos os 7 vídeos foram gerados com sucesso!")
print("=" * 60)
print("\nArquivos criados:")
print("  ANÁLISE (5):")
print("    - 01_linear_simples.mp4")
print("    - 02_acelerado.mp4")
print("    - 03_mudancas_direcao.mp4")
print("    - 04_oclusao.mp4")
print("    - 05_complexo.mp4")
print("  TESTE (2):")
print("    - 06_circular_suave.mp4")
print("    - 07_aleatorio_caotico.mp4")
print("\nLocalização: c:\\Users\\saulo\\Desktop\\PASKalman\\Vídeos para Projeto\\")
print("=" * 60)