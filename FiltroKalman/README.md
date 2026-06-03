# 🎯 GUIA COMPLETO: Filtro de Kalman - Sistema de Rastreamento com GUI

**Sistemas Adaptativos de Sinais | UFCG**  
_Um projeto educacional completo sobre filtragem, processamento de vídeo e interface gráfica._

---

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Arquitetura e Design](#arquitetura-e-design)
3. [Tecnologias Utilizadas](#tecnologias-utilizadas)
4. [Estrutura do Projeto](#estrutura-do-projeto)
5. [Como Funciona - Guia Detalhado](#como-funciona---guia-detalhado)
6. [Implementação Matemática](#implementação-matemática)
7. [Como Modificar e Estender](#como-modificar-e-estender)
8. [Troubleshooting Avançado](#troubleshooting-avançado)
9. [Melhorias Recomendadas](#melhorias-recomendadas)

---

## 🎬 Visão Geral

Este projeto implementa um **sistema completo de processamento de vídeo com Filtro de Kalman** em Python. A aplicação:

- **Carrega vídeos** em formatos comuns (MP4, MKV, AVI)
- **Detecta objetos** usando processamento de imagem customizável
- **Filtra trajetórias** com Kalman 4D ou 6D para suavizar ruído
- **Processa em thread** sem travar a interface
- **Reproduz com sincronização** entre vídeo e análise de métricas
- **Exibe métricas** em tempo real (RMS, trajetória, comparação medida vs filtrada)

### Características Principais

| Feature | Descrição |
|---------|-----------|
| 🎥 **Processamento Offline** | Carrega vídeo, processa frame por frame, salva resultado |
| 🤖 **Filtro Adaptativo** | Kalman 4D (posição+velocidade) ou 6D (+ aceleração) |
| 📊 **Análise em Tempo Real** | 3 gráficos sincronizados durante playback |
| 🎨 **Interface Minimalista** | Design limpo branco/cinza (sem distração) |
| 🔧 **Detector Customizável** | Pasta `src/detec/` para seu código de detecção |
| ⚡ **Performance Otimizada** | Gráficos throttled (a cada 3 frames), thread-safe |

---

## 🏗️ Arquitetura e Design

### Padrão de Arquitetura: MVC + Threading

```
┌─────────────────────────────────────────────┐
│         MODEL (Lógica de Processamento)     │
│  ┌──────────────┐  ┌──────────────────┐    │
│  │   Kalman     │  │  Video Processing │    │
│  │   Classes    │  │  (detect, filter) │   │
│  └──────────────┘  └──────────────────┘    │
└─────────────────────────────────────────────┘
              ↑              ↑
┌─────────────────────────────────────────────┐
│          VIEW (Interface Gráfica)            │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐    │
│  │  Panels  │ │ Controls │ │ Graphs  │    │
│  └──────────┘ └──────────┘ └─────────┘    │
└─────────────────────────────────────────────┘
              ↓              ↓
┌─────────────────────────────────────────────┐
│      THREADING (Não bloqueia GUI)           │
│  ┌─────────────────────────────────────┐   │
│  │ _process_video() em Thread Daemon   │   │
│  │ Deixa GUI responsiva durante proc.  │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

### Decisões de Design

**Por que Threading?**
- Processamento de vídeo é CPU-bound (lento)
- Thread separada impede que a GUI congele
- Usuário pode interagir enquanto processa

**Por que Matplotlib para gráficos?**
- Melhor que tkinter nativo para plotagem científica
- Integração com Tkinter via `FigureCanvasTkAgg`
- Suporte a múltiplas linhas e customização avançada

**Por que Kalman em vez de outro filtro?**
- Ótimo para objeto com velocidade aproximadamente constante
- Fornece estimativas de velocidade (não apenas posição)
- Bem-definido matematicamente (6 linhas de código = uma predição+atualização)
- Padrão em sistemas reais (GPS, radar, navegação)

---

## 📦 Tecnologias Utilizadas

### Core (Backend)

| Lib | Versão | Razão |
|-----|--------|-------|
| **numpy** | ≥1.20 | Operações matriciais eficientes (Kalman trabalha com matrizes) |
| **opencv-python** | ≥4.5 | Captura/gravação de vídeo + processamento de imagem |
| **scipy** | ≥1.6 | (Opcional) Operações numéricas avançadas |

### GUI e Visualização

| Lib | Versão | Razão |
|-----|--------|-------|
| **tkinter** | Nativa | Interface gráfica leve, multiplataforma |
| **Pillow** | ≥8.0 | Conversão BGR→RGB e escalamento de imagens |
| **matplotlib** | ≥3.3 | Plotagem científica integrada com Tkinter |

### Padrão de Código

- **Python 3.7+**: Type hints, async/await ready
- **PEP 8**: Código limpo e legível
- **Modularização**: Separação clara de responsabilidades

---

## 📁 Estrutura do Projeto

```
FiltroKalman/
├── main.py                          (1) Entrypoint - inicializa GUI
│
├── requirements.txt                 (2) Dependências pip
│
├── README.md                        (3) Este guia
│
└── src/
    ├── __init__.py
    │
    ├── kalman/                      (4) LÓGICA DO FILTRO
    │   ├── __init__.py
    │   ├── filter.py                → KalmanFilter2D (base 4D)
    │   │                               - predict()
    │   │                               - update()
    │   │                               - get_position()
    │   │
    │   └── models.py                → KalmanModel4D/6D (wrappers)
    │                                   - Parser de parâmetros Q/R
    │                                   - Inicialização de matrizes
    │
    ├── gui/                         (5) INTERFACE GRÁFICA
    │   ├── __init__.py
    │   ├── app.py                   → KalmanApp (classe principal 900+ linhas)
    │   │                               - Layout: 3 painéis
    │   │                               - _process_video() (thread worker)
    │   │                               - _poll_playback() (loop 30 FPS)
    │   │                               - _update_metrics_plots() (gráficos)
    │   │
    │   └── viewers.py               → VideoViewer (widget customizado)
    │                                   - display_image()
    │                                   - Letterboxing (sem distorção)
    │
    ├── detec/                       (6) DETECTOR CUSTOMIZÁVEL
    │   ├── __init__.py
    │   ├── detector.py              → Sua função detect_centroid()
    │   │                               - Edite para seu próprio detector!
    │   │
    │   └── README.md                → Exemplo (HSV, Blob, Template)
    │
    └── data/                        (7) SAÍDA
        └── output.mp4               → Vídeo processado com anotações
```

### Fluxo de Dados

```
                USER
                 ↓
        1. CARGA [Load Button]
                 ↓
        ┌──────────────────┐
        │  src/gui/app.py  │
        │  load_video()    │  → Lê vídeo, mostra frame 0
        └──────────────────┘
                 ↓
        2. CONFIGURAÇÃO PARÂMETROS
        ┌──────────────────┐
        │ Modelo (4D/6D)   │
        │ Q, R (Kalman)    │
        │ Debug (checkbox) │
        └──────────────────┘
                 ↓
        3. PROCESSAMENTO [EXEC Button]
        ┌──────────────────────────────────┐
        │ _process_video() em THREAD DAEMON │
        │                                  │
        │ Loop para cada frame:            │
        │  ├─ src/detec/detector.py       │
        │  │  detect_centroid(frame)      │
        │  │                              │
        │  ├─ src/kalman/models.py        │
        │  │  Kalman.predict()            │
        │  │  Kalman.update(measurement)  │
        │  │  filtra trajetória           │
        │  │                              │
        │  ├─ Calcula métricas            │
        │  │  (RMS, sqerr_x, sqerr_y)    │
        │  │                              │
        │  ├─ Desenha anotações           │
        │  │  (se debug ativado)          │
        │  │                              │
        │  └─ VideoWriter.write()         │
        │     Salva em src/data/          │
        └──────────────────────────────────┘
                 ↓
        4. PLAYBACK [Play Button]
        ┌──────────────────────────────────┐
        │ _poll_playback() a cada 33ms     │
        │                                  │
        │ ├─ current_frame_idx += 1       │
        │ ├─ _display_current_frame()     │
        │ │  Mostra frame anotado         │
        │ │                              │
        │ └─ _update_metrics_plots()      │
        │    Sincroniza gráficos          │
        │    (RMS, trajetória até idx)   │
        └──────────────────────────────────┘
```

---

## 🔍 Como Funciona - Guia Detalhado

### 1️⃣ Entrypoint: `main.py`

```python
from src.gui.app import run_app

if __name__ == "__main__":
    run_app()
```

**O que faz:**
- Importa `KalmanApp` de `app.py`
- Cria janela Tkinter
- Passa controle para o loop de eventos

**Modificações possíveis:**
- Carregar configurações padrão de arquivo
- Passar argumentos CLI (vídeo pré-selecionado)

---

### 2️⃣ GUI Principal: `src/gui/app.py`

**Classe: `KalmanApp` (~900 linhas)**

#### **Estrutura da Janela**

```
┌─────────────────────────────────────────────────┐
│  window (tamanho mín 1400×750, maximiza)        │
├────────┬──────────────────────┬─────────────────┤
│        │                      │                 │
│ LEFT   │    CENTER PANEL      │   RIGHT PANEL   │
│ 300px  │    (flex 750px)      │     450px       │
│        │                      │                 │
│        │                      │     RMS X       │
│ ┌────┐ │ ┌────────────────┐   │  (gráfico)      │
│ │Load│ │ │   Video Info   │   │                 │
│ └────┘ │ │  (tamanho,fps) │   │     RMS Y       │
│        │ ├────────────────┤   │  (gráfico)      │
│ ┌────┐ │ │   Viewer       │   │                 │
│ │Mod │ │ │ (640×480)      │   │  Trajectória    │
│ │Q:R │ │ ├────────────────┤   │  (gráfico)      │
│ └────┘ │ │  Time: 01:23   │   │                 │
│        │ ├────────────────┤   │                 │
│ ┌────┐ │ │ ◄ Play ►       │   │                 │
│ │Deb │ │ └────────────────┘   │                 │
│ │ug  │ │ Debug Legend:        │                 │
│ └────┘ │ 🔴 Detection         │                 │
│        │ 🟦 Kalman            │                 │
│ ┌────┐ │ 🟩 Trajectory        │                 │
│ │EXEC│ │                      │                 │
│ └────┘ │                      │                 │
└────────┴──────────────────────┴─────────────────┘
```

#### **Painel Esquerdo (Controls)**

```python
# Seção 1: Load Video
self.load_btn = tk.Button(...)  # Abre file dialog

# Seção 2: Filter Options
self.model_var = tk.StringVar()  # "4D" ou "6D"
self.q_entry = tk.Entry()        # Ex: "1e-2,1e-2,1e-1,1e-1"
self.r_entry = tk.Entry()        # Ex: "1e-1,1e-1"

# Seção 3: Debug Options
self.show_traj = tk.BooleanVar()      # Desenhar trajetória?
self.show_detect = tk.BooleanVar()    # Desenhar detecção?
self.show_kalman = tk.BooleanVar()    # Desenhar Kalman?
self.show_window = tk.BooleanVar()    # Reservado

# Seção 4: Execute
self.exec_btn = tk.Button(...)   # Inicia processamento
self.status_lbl = tk.Label(...)  # "Processando..." ou "Pronto"
```

**Por que 4 seções?**
- Separação clara de tarefas
- Usuário segue fluxo: Load → Configure → Debug → Execute

#### **Painel Central (Video Display)**

```python
# Info banner (atualizado ao carregar vídeo)
self.video_info_lbl.config(text=
    f"Arquivo: {nome} ({ext}) | Tamanho: {MB:.1f} | "
    f"FPS: {fps:.1f} | Frames: {total} | Taxa: {w}×{h}"
)

# Viewer (constrói frame com letterboxing)
self.tela_viewer = VideoViewer(...)
self.tela_viewer.display_image(frame)

# Time display
self.time_info_lbl.config(text="00:00 / 05:47")

# Playback controls
self.prev_btn   # ◄ Anterior
self.play_btn   # ⏵ Play/Pause
self.next_btn   # ► Próximo

# Debug legend
# 🔴 | 🟦 | 🟩 explicando cores
```

#### **Painel Direito (Metrics)**

```python
# Cada gráfico é uma Figure matplotlib integrada em Tkinter

# 1. RMS X (altura fixa 140px)
self.rmsx_fig = Figure(figsize=(4.2, 1.0), ...)
self.rmsx_ax.plot([], [], color="#333333", linewidth=2)
self.rmsx_canvas = FigureCanvasTkAgg(self.rmsx_fig, ...)

# 2. RMS Y (similar)
self.rmsy_fig = Figure(...)

# 3. Trajetória (flex, ocupa resto do espaço)
self.traj_fig = Figure(figsize=(4.2, 3.2), ...)
self.traj_ax.plot([], [], "r.-", label="medido")    # vermelho
self.traj_ax.plot([], [], "#333333", label="filtrado")  # preto
```

---

### 3️⃣ Loading & Configuration

#### **load_video() - Passos**

```python
def load_video(self):
    1. File dialog (selecionar arquivo)
       path = filedialog.askopenfilename()
    
    2. Abrir com OpenCV
       cap = cv2.VideoCapture(path)
    
    3. Extrair propriedades
       total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
       fps = cap.get(cv2.CAP_PROP_FPS)
       width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
       height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
       file_size = os.path.getsize(path)
    
    4. Ler primeiro frame
       ret, frame = cap.read()
    
    5. Exibir no VideoViewer
       self.tela_viewer.display_image(frame)
    
    6. Atualizar info banner
       "Arquivo: video.mp4 | Tamanho: 125.5 MB | FPS: 30.0 | ..."
    
    7. Habilitar EXEC button
       self.exec_btn.config(state="normal")
```

**Por que em passos?**
- Cada passo pode falhar (arquivo corrompido, sem codec, etc)
- Fácil de debugar/modificar

---

### 4️⃣ Processing: _process_video() (700+ linhas)

**Este é o coração do sistema!**

```python
def _process_video(self):
    # Rodando em THREAD DAEMON (não bloqueia GUI)
    
    # SETUP
    cap = cv2.VideoCapture(self.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH/HEIGHT)
    
    # Criar VideoWriter (output.mp4)
    out = cv2.VideoWriter("src/data/output.mp4", fourcc, fps, (w, h))
    
    # Criar Kalman filter (4D ou 6D)
    if model == "6D":
        kf = KalmanModel6D(dt=1/fps, q_diag=q_vals, r_diag=r_vals)
    else:
        kf = KalmanModel4D(dt=1/fps, q_diag=q_vals, r_diag=r_vals)
    
    # MAIN LOOP
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. DETECÇÃO (seu código em src/detec/detector.py)
        meas = detect_centroid(frame)
        
        # 2. PREDIÇÃO KALMAN
        kf.predict()
        
        # 3. ATUALIZAÇÃO KALMAN
        if meas is not None:
            kf.update(meas)
        
        # 4. OBTER POSIÇÃO ESTIMADA
        est = kf.get_position()  # [x, y]
        
        # 5. ARMAZENAR DADOS (para gráficos)
        self.meas_pts.append(meas or None)
        self.filt_pts.append((est[0], est[1]))
        
        # Calcular erro (para RMS)
        if meas:
            self.sqerr_x.append((est[0] - meas[0])**2)
            self.sqerr_y.append((est[1] - meas[1])**2)
        else:
            self.sqerr_x.append(np.nan)
            self.sqerr_y.append(np.nan)
        
        # 6. ANOTAÇÕES (desenhar no frame)
        ann = frame.copy()
        
        if self.show_detect.get() and len(self.meas_pts) >= 2:
            # Desenhar trajetória vermelha (medida)
            meas_poly = [tuple(np.array(p, dtype=int)) 
                        for p in self.meas_pts if p is not None]
            cv2.polylines(ann, [np.array(meas_poly)], False, (0, 0, 255), 2)
        
        if self.show_kalman.get() and len(self.filt_pts) >= 2:
            # Desenhar trajetória azul (filtrada)
            cv2.polylines(ann, [np.array(self.filt_pts, dtype=int)], False, (255, 0, 0), 2)
        
        if self.show_traj.get() and meas:
            # Desenhar ponto de detecção
            cv2.circle(ann, tuple(meas), 5, (0, 255, 0), -1)
        
        # 7. ESCREVER FRAME ANOTADO
        out.write(ann)
        
        frame_count += 1
        self.status_lbl.config(text=f"Processado: {frame_count}/{total_frames}")
        self.root.update_idletasks()  # Atualizar label sem bloquear
    
    # CLEANUP
    cap.release()
    out.release()
    
    # Recarregar processado para playback
    self._on_processing_complete()
```

**Detalhes Importantes:**

1. **Thread Daemon**
   ```python
   self.worker = threading.Thread(target=self._process_video, daemon=True)
   self.worker.start()
   ```
   - `daemon=True` = thread termina quando main encerra
   - Não bloqueia GUI

2. **detect_centroid() Importado**
   ```python
   from src.detec.detector import detect_centroid
   meas = detect_centroid(frame)  # Sua função!
   ```
   - Permite customização sem modificar app.py

3. **Kalman Update Loop**
   ```
   Para cada frame:
   ├─ predict()      → estima posição nova
   ├─ update(meas)   → corrige com medição
   └─ get_position() → retorna [x, y] filtrado
   ```

---

### 5️⃣ Playback: _poll_playback()

```python
def _poll_playback(self):
    """Executado a cada 33ms (~30 FPS)"""
    
    if self.playing and self.current_frame_idx < self.total_frames - 1:
        # Avançar frame
        self.current_frame_idx += 1
        self._display_current_frame()
    
    elif self.playing and self.current_frame_idx >= self.total_frames - 1:
        # Fim do vídeo
        self.playing = False
        self.play_btn.config(text="⏵")
    
    # Agendar próxima execução
    self.root.after(33, self._poll_playback)
```

**Por que 33ms?**
- 1000ms / 30 FPS ≈ 33ms por frame
- Suave para vídeos em 30 FPS

#### **_display_current_frame()**

```python
def _display_current_frame(self):
    """Mostra frame no viewer e atualiza gráficos"""
    
    # 1. Ler frame no índice atual
    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
    ret, frame = self.cap.read()
    
    if ret:
        # 2. Exibir
        self.tela_viewer.display_image(frame)
        
        # 3. Atualizar tempo
        current_time = self.current_frame_idx / self.video_fps
        total_time = self.total_frames / self.video_fps
        self.time_info_lbl.config(text=
            f"{self._format_time(current_time)} / "
            f"{self._format_time(total_time)}"
        )
        
        # 4. Sincronizar gráficos (KEY POINT!)
        self._update_metrics_plots()
```

#### **_update_metrics_plots() - A Sincronização**

```python
def _update_metrics_plots(self):
    """Sincroniza gráficos com frame atual"""
    
    # Throttle: atualiza a cada 3 frames
    self.metrics_update_counter += 1
    if self.metrics_update_counter % 3 != 0:
        return
    
    # IMPORTANTE: slice até current_frame_idx!
    current_idx = self.current_frame_idx + 1
    
    # RMS X - apenas dados até agora
    sx = np.array(self.sqerr_x[:current_idx] if not nan)
    if sx.size > 0:
        running_mean_x = np.cumsum(sx) / np.arange(1, len(sx)+1)
        running_rms_x = np.sqrt(running_mean_x)
        self.rmsx_line.set_data(range(len(running_rms_x)), running_rms_x)
    
    # RMS Y - similar
    
    # Trajetória - slice dos pontos
    xs_meas = [p[0] for p in self.meas_pts[:current_idx] if p]
    ys_meas = [p[1] for p in self.meas_pts[:current_idx] if p]
    xs_filt = [p[0] for p in self.filt_pts[:current_idx]]
    ys_filt = [p[1] for p in self.filt_pts[:current_idx]]
    
    self.traj_line_meas.set_data(xs_meas, ys_meas)
    self.traj_line_filt.set_data(xs_filt, ys_filt)
    
    # Redesenha canvas
    self.rmsx_canvas.draw_idle()
    self.rmsy_canvas.draw_idle()
    self.traj_canvas.draw_idle()
```

**Por que `[:current_idx]`?**
- Simula a trajetória "crescendo" durante playback
- Mostra apenas o que já foi "processado"
- Sincroniza vídeo com análise

---

### 6️⃣ Custom Video Viewer: `src/gui/viewers.py`

```python
class VideoViewer(tk.Frame):
    def display_image(self, cv_image):
        """Mostra imagem BGR (OpenCV) em Tkinter"""
        
        # 1. Converter BGR → RGB (Tkinter espera RGB)
        frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # 2. Redimensionar COM letterboxing
        frame = self._resize_letterbox(frame, self.width, self.height)
        
        # 3. Converter para PIL Image
        img = Image.fromarray(frame)
        
        # 4. Converter para PhotoImage (formato Tkinter)
        img_tk = ImageTk.PhotoImage(img)
        
        # 5. Atualizar label
        self.display_label.config(image=img_tk)
        self.display_label.image = img_tk  # Manter referência!
    
    def _resize_letterbox(self, img, target_w, target_h):
        """Redimensiona mantendo aspect ratio (com bordas cinzas)"""
        h, w = img.shape[:2]
        
        # Calcular scaling
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w*scale), int(h*scale)
        
        # Criar canvas cinzento
        canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 64
        
        # Centralizar imagem no canvas
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        # Redimensionar e colar
        resized = cv2.resize(img, (new_w, new_h))
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
```

**Por que letterboxing?**
- Vídeos pode ter proporção 16:9, 4:3, 1:1, etc
- Sem letterboxing = distorção
- Com letterboxing = preserva proporção original

---

## 📐 Implementação Matemática

### Kalman Filter 4D

**Estado:** x = [px, py, vx, vy]ᵀ
- px, py: posição em pixels
- vx, vy: velocidade em pixels/frame

#### **Predição**

```
Dinâmica: x_pred = F @ x + w
          onde F assume velocidade constante

        ┌           ┐
        │ 1  0  dt  0 │
    F = │ 0  1  0  dt│
        │ 0  0  1  0 │
        │ 0  0  0  1 │
        └           ┘

    dt = 1.0/fps (tempo entre frames)

Significado:
    px_new = px_old + vx * dt
    py_new = py_old + vy * dt
    vx_new = vx_old (velocidade constante)
    vy_new = vy_old
```

#### **Observação (Medição)**

```
Medimos apenas posição:
z = H @ x
onde H extrai [px, py]

        ┌     ┐
    H = │ 1 0 0 0 │
        │ 0 1 0 0 │
        └     ┘
```

#### **Atualização (Incorporate Measurement)**

```
Kalman Gain:
    K = P_pred @ Hᵀ @ inv(H @ P_pred @ Hᵀ + R)

Correção de estado:
    x_new = x_pred + K @ (z - H @ x_pred)
       ^       ^         ^   ^
       |       |         |   └─ inovação (medição - predição)
       |       |         └────── ganho Kalman
       |       └─ predição
       └─ nova estimativa

Correção de covariância:
    P_new = (I - K @ H) @ P_pred
```

#### **Parâmetros Q e R**

```python
Q = diag([q0, q1, q2, q3])  # Ruído de PROCESSO
Quanto VOCÊ desconfia da predição:
- Valores ALTOS   = "Não confio na predição, mude o modelo"
- Valores BAIXOS  = "A predição é boa, mantém"

R = diag([r0, r1])           # Ruído de MEDIÇÃO  
Quanto VOCÊ desconfia da medição:
- Valores ALTOS   = "Sensor é ruim, ignore medições"
- Valores BAIXOS  = "Sensor é bom, siga-o"
```

**Exemplo intuitivo:**
```
Medição ruidosa (Q=alto, R=baixo):
    └─ Kalman vai confiar na medição mesmo com ruído
    └─ Trajetória vai "pular" muito

Predição confiável (Q=baixo, R=alto):
    └─ Kalman vai confiar na predição
    └─ Trajetória vai suave
```

#### **Código em Python**

```python
import numpy as np

class KalmanFilter2D:
    def __init__(self, dt, process_var, meas_var):
        # F = transição de estado (velocidade constante)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)
        
        # H = matriz de observação (mede posição)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)
        
        # Q = covariância do ruído de processo
        self.Q = np.diag(process_var)
        
        # R = covariância do ruído de medição
        self.R = np.diag(meas_var)
        
        # Estados iniciais
        self.x = np.array([0, 0, 0, 0], dtype=float)  # [px, py, vx, vy]
        self.P = np.eye(4)  # Covariância do estado
    
    def predict(self):
        """Prediz próximo estado"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, z):
        """Atualiza com medição z=[px, py]"""
        z = np.array([z[0], z[1]], dtype=float)
        
        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Atualizar estado
        self.x = self.x + K @ (z - self.H @ self.x)
        
        # Atualizar covariância
        self.P = (np.eye(4) - K @ self.H) @ self.P
    
    def get_position(self):
        """Retorna posição estimada [px, py]"""
        return [self.x[0], self.x[1]]
```

### Kalman Filter 6D (Com Aceleração)

**Semelhante a 4D, mas com aceleração:**

```
Estado: x = [px, py, vx, vy, ax, ay]ᵀ

Dinâmica (aceleração constante por frame):
    px_new = px + vx*dt + 0.5*ax*dt²
    py_new = py + vy*dt + 0.5*ay*dt²
    vx_new = vx + ax*dt
    vy_new = vy + ay*dt
    ax_new = ax
    ay_new = ay
```

**(A implementação é similar, apenas matriz F é maior)**

---

## 🔧 Como Modificar e Estender

### 1️⃣ Modificar Detector (src/detec/detector.py)

**Atual (Threshold + maior contorno):**
```python
# detector.py
def detect_centroid(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
```

**Customizações comuns:**

a) **Detector por Cor (HSV)**
```python
def detect_centroid(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([120, 50, 50])    # Ajuste para sua cor
    upper = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    # ... resto do código (contours)
```

b) **Blob Detector**
```python
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea, maxArea = 50, 10000
detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(frame)
# Extrair (x, y) do maior blob
```

c) **Template Matching**
```python
template = cv2.imread("template.png", cv2.IMREAD_GRAYSCALE)
result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
# max_loc é o canto superior-esquerdo do match
```

d) **YOLO (Deep Learning)**
```python
import torch
net = torch.hub.load('ultralytics/yolov5', 'yolov5s')
results = net(frame)
# Extrair bbox do primeiro objeto
```

**Como o sistema usa automaticamente:**
```python
# app.py importa sua função
from src.detec.detector import detect_centroid

# No _process_video():
meas = detect_centroid(frame)  # Chama SUA implementação!
```

---

### 2️⃣ Modificar Parâmetros Kalman em Runtime

**Arquivo:** `src/gui/app.py`, método `_parse_config()`

**Atual:**
```python
def _parse_config(self):
    qtxt = self.q_entry.get()  # "1e-2,1e-2,1e-1,1e-1"
    rtxt = self.r_entry.get()  # "1e-1,1e-1"
    self.config_Q = [float(x) for x in qtxt.split(",")]
    self.config_R = [float(x) for x in rtxt.split(",")]
```

**Para adicionar validação:**
```python
def _parse_config(self):
    try:
        qtxt = self.q_entry.get()
        rtxt = self.r_entry.get()
        
        Q = [float(x) for x in qtxt.split(",")]
        R = [float(x) for x in rtxt.split(",")]
        
        # Validação
        assert len(Q) in [4, 6], "Q deve ter 4 ou 6 valores"
        assert len(R) == 2, "R deve ter 2 valores"
        assert all(x > 0 for x in Q + R), "Valores devem ser positivos"
        
        self.config_Q = Q
        self.config_R = R
        messagebox.showinfo("OK", "Configuração válida!")
    except Exception as e:
        messagebox.showerror("Erro", f"Parâmetros inválidos: {e}")
```

---

### 3️⃣ Adicionar Novo Gráfico

**Exemplo: Velocidade Estimada (vx, vy)**

```python
# Em __init__(), add Seção 5 no panel direito:
speed_frame = tk.LabelFrame(self.right_frame, text="Velocidade", ...)
self.speed_fig = Figure(figsize=(4.2, 1.0), facecolor="white")
self.speed_ax = self.speed_fig.add_subplot(111)
self.speed_line, = self.speed_ax.plot([], [], "green", linewidth=2)
self.speed_canvas = FigureCanvasTkAgg(self.speed_fig, master=speed_frame)
speed_frame.pack(fill="x", pady=(0, 4))

# Em _process_video(), armazenar velocidades:
vel_magnitude = np.sqrt(kf.x[2]**2 + kf.x[3]**2)  # ||[vx, vy]||
self.vel_data.append(vel_magnitude)

# Em _update_metrics_plots(), desenhar:
vels = self.vel_data[:current_idx]
if vels:
    self.speed_line.set_data(range(len(vels)), vels)
    self.speed_ax.set_xlim(0, len(vels))
    self.speed_ax.set_ylim(0, max(vels) * 1.2)
    self.speed_canvas.draw_idle()
```

---

### 4️⃣ Modificar Cores (Tema)

**Arquivo:** `src/gui/app.py`, basta procurar por hex colors

**Cores atuais (minimalista):**
```python
"#ffffff"  # Branco puros backgrounds
"#f9f9f9"  # Quase branco
"#f5f5f5"  # Cinza muito claro
"#e8e8e8"  # Cinza claro (info banner)
"#4a4a4a"  # Cinza escuro (botões)
"#333333"  # Quase preto (texto)
```

**Para tema ESCURO (Dark Mode):**
```python
# Fundo geral
bg="#1a1a1a"  # Muito escuro
secondary_bg="#2a2a2a"
tertiary_bg="#3a3a3a"

# Texto
fg="#ffffff"  # Branco puro
secondary_fg="#cccccc"  # Cinza claro

# Acento
accent="#00bcd4"  # Cyan
```

---

### 5️⃣ Adicionar Checkpoint (Salvar Progresso)

```python
import json

def save_state(self):
    """Salva data processada"""
    state = {
        "video_path": self.video_path,
        "total_frames": self.total_frames,
        "fps": self.video_fps,
        "meas_pts": self.meas_pts,
        "filt_pts": self.filt_pts,
        "sqerr_x": [float(x) if not np.isnan(x) else None for x in self.sqerr_x],
        "sqerr_y": [float(x) if not np.isnan(x) else None for x in self.sqerr_y],
    }
    with open("checkpoint.json", "w") as f:
        json.dump(state, f)
    messagebox.showinfo("OK", "Estado salvo!")

def load_state(self):
    """Carrega data processada"""
    try:
        with open("checkpoint.json", "r") as f:
            state = json.load(f)
        self.video_path = state["video_path"]
        self.total_frames = state["total_frames"]
        self.video_fps = state["fps"]
        self.meas_pts = state["meas_pts"]
        self.filt_pts = state["filt_pts"]
        self.sqerr_x = np.array(state["sqerr_x"])
        self.sqerr_y = np.array(state["sqerr_y"])
        messagebox.showinfo("OK", "Estado carregado!")
    except Exception as e:
        messagebox.showerror("Erro", f"Não conseguiu carregar: {e}")
```

---

## 🐛 Troubleshooting Avançado

### Problema 1: Detector não funciona

**Sintomas:** Kalman fica no lugar, sem movimento

**Solução:**
```python
# Em detect_centroid(), add debug:
def detect_centroid(frame):
    # ... código ...
    if result is None:
        print(f"[DEBUG] Frame {frame_idx}: nenhum objeto detectado")
    else:
        print(f"[DEBUG] Frame {frame_idx}: ({result[0]}, {result[1]})")
    return result
```

**Testar manualmente:**
```python
import cv2
from src.detec.detector import detect_centroid

cap = cv2.VideoCapture("seu_video.mp4")
for i in range(10):
    ret, frame = cap.read()
    result = detect_centroid(frame)
    print(f"Frame {i}: {result}")
```

---

### Problema 2: Kalman "lag" (demora para seguir)

**Sintomas:** Trajetória filtrada fica muito atrás da medida

**Causa:** Q muito baixo (confia demais na predição)

**Solução:** Aumentar Q

```
Antes: Q = 1e-2, 1e-2, 1e-1, 1e-1 (muito baixo)
Depois: Q = 1e-1, 1e-1, 1e-0, 1e-0  (mais confiável)

Intuição: "Meu modelo de velocidade constante pode estar errado,
          então deixo o filtro se adaptar mais rápido"
```

---

### Problema 3: Trajetória muito oscilante (jitter)

**Sintomas:** Linha salta muito de um lado pro outro

**Causa:** R muito alto (não confia na medição)

**Solução:** Diminuir R ou melhorar detector

```
Antes: R = 1e-1, 1e-1 (desconfia muito)
Depois: R = 1e-2, 1e-2 (confia mais no sensor)

Ou: Melhorar detector (HSV tunned, pré-filtro gaussian)
```

---

### Problema 4: Processamento muito lento (30 FPS → 5 FPS)

**Causas possíveis:**
1. Detector lento (ex: YOLO não otimizado)
2. Vídeo muito grande
3. CPU fraca

**Soluções:**
```python
# 1. Processar em escala reduzida
frame_small = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
meas = detect_centroid(frame_small)
meas = (meas[0]*2, meas[1]*2)  # Re-escalar coordenadas

# 2. Skipframe: processar a cada 2 frames
if frame_count % 2 == 0:  # Processa a cada 2 frames
    meas = detect_centroid(frame)

# 3. Usar GPU para detector (CUDA)
# yolo = torch.hub.load(..., pretrained=True, device='cuda')
```

---

### Problema 5: GUI congela durante processamento

**Causa:** Code NOT em thread ou thread ativa loop GUI

**Verificar:**
```python
# app.py, deve ter:
self.worker = threading.Thread(target=self._process_video, daemon=True)
self.worker.start()

# E _process_video() DEVE chamar:
self.root.update_idletasks()  # Em vez de update()
```

---

## 🚀 Melhorias Recomendadas

### 1️⃣ Detector Robusto (YOLO ou Detectron2)

**Vantagem:** Funciona com múltiplos objetos, diferentes ângulos

```python
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

def detect_centroid(frame):
    results = model(frame)
    detections = results.xyxy[0]  # [x1, y1, x2, y2, conf, class]
    
    if len(detections) == 0:
        return None
    
    # Pegar primeiro objeto confiante
    best = detections[0]
    cx = (best[0] + best[2]) / 2
    cy = (best[1] + best[3]) / 2
    return (int(cx), int(cy))
```

---

### 2️⃣ Multiple Object Tracking (MOT)

**Para rastrear N objetos:**

```python
class MultiObjectTracker:
    def __init__(self):
        self.tracks = {}  # id → KalmanFilter
        self.next_id = 0
    
    def update(self, detections):
        """
        detections: lista de (x, y) detectados
        Retorna: dict de id → posição filtrada
        """
        # Data association (Hungarian algorithm)
        # Associar detecções a tracks existentes
        # Criar novos tracks para detecções não associadas
        pass
```

---

### 3️⃣ Adaptive Q/R

**Kalman com parâmetros dinâmicos:**

```python
def adaptive_Q(error_magnitude, window_size=20):
    """Q aumenta se o modelo está errando muito"""
    recent_errors = error_magnitude[-window_size:]
    mean_error = np.mean(recent_errors)
    
    # Q começa baixo
    Q_base = 1e-2
    # Aumenta linearmente com erro
    Q_adaptive = Q_base + mean_error * 10
    return Q_adaptive
```

---

### 4️⃣ Saved Sessions

```python
import pickle

def save_session(self):
    """Salva tudo para continuar depois"""
    session = {
        "meas_pts": self.meas_pts,
        "filt_pts": self.filt_pts,
        "sqerr_x": self.sqerr_x,
        "sqerr_y": self.sqerr_y,
        "config_Q": self.config_Q,
        "config_R": self.config_R,
    }
    with open("session.pkl", "wb") as f:
        pickle.dump(session, f)

def load_session(self):
    """Usa dados salvos"""
    with open("session.pkl", "rb") as f:
        session = pickle.load(f)
    self.meas_pts = session["meas_pts"]
    # ...
```

---

### 5️⃣ Advanced Metrics

```python
def calculate_metrics(self):
    """Análise mais profunda"""
    meas_xy = np.array([p for p in self.meas_pts if p is not None])
    filt_xy = np.array(self.filt_pts)
    
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(meas_xy - filt_xy))
    
    # Eficiência de filtragem
    noise_before = np.std(meas_xy[:, 0])  # Ruído original
    noise_after = np.std(meas_xy[:, 0] - filt_xy[:len(meas_xy), 0])
    reduction = (1 - noise_after/noise_before) * 100
    
    # Suavidade (curvatura da trajetória)
    diffs = np.diff(filt_xy, axis=0)
    curvature = np.sum(np.sqrt(np.sum(np.diff(diffs, axis=0)**2, axis=1)))
    
    return {
        "mae": mae,
        "noise_reduction": reduction,
        "smoothness": curvature,
    }
```

---

## 📚 Recursos e Referências

| Tópico | Recurso |
|--------|---------|
| **Kalman Filter** | [Tutorial MIT](https://ocw.mit.edu) - Kalman filtering and tracking |
| **OpenCV** | [docs.opencv.org](https://docs.opencv.org/) |
| **Object Tracking** | [MOT Challenge](https://motchallenge.net/) - Dataset + baseline |
| **Deep Learning Detection** | [YOLOv5](https://github.com/ultralytics/yolov5) |
| **Python Threading** | [Real Python Threading](https://realpython.com/intro-to-python-threading/) |
| **Matplotlib em Tkinter** | [Embedding Matplotlib](https://matplotlib.org/stable/gallery/user_interfaces/embedding_in_tk_agg.html) |

---

## 🎓 Conceitos-Chave Explicados

### Predição vs Atualização

```
Predição (sem medição):
├─ Usa modelo físico
├─ Prevê onde objeto vai
└─ Pode estar errado (sem sensor)

Atualização (com medição):
├─ Recebe sensor
├─ Compara predição vs medição
└─ Ajusta estimativa
```

### Matriz de Covariância P

```
P = "incerteza nas nossas estimativas"
    
Começar com P = I (alta incerteza)
Durante filtro:
├─ Predição aumenta P (acumula incerteza)
└─ Atualização reduz P (ganha informação)

Padrão: P → diminui, depois oscila
```

### Gain de Kalman K

```
K = "quanto confiar na medição?"

Se R é PEQUENO (sensor bom):
└─ K é GRANDE → confia muito na medição
└─ Trajetória segue sensor de perto

Se Q é PEQUENO (modelo confiável):
└─ K é PEQUENO → confia na predição
└─ Trajetória é suave

K começa alto (aprende) e diminui (estabiliza)
```

---

## ✅ Checklist de Implementação

Você tem tudo implementado:

- ✅ GUI minimalista (branco/cinza)
- ✅ 3 painéis (controles, viewer, gráficos)
- ✅ Kalman 4D e 6D
- ✅ Processamento em thread
- ✅ Playback sincronizado
- ✅ Detector customizável em src/detec/
- ✅ Gráficos atualizando em tempo real
- ✅ Informações de vídeo (tamanho, FPS, frames)
- ✅ Tema profissional
- ✅ Documentação completa

---

## 👨‍🏫 Conclusão

Este projeto é um **laboratório completo** de:
- Processamento de sinais adaptativos (Kalman)
- Programação orientada a objetos
- Desenvolvimento de GUI
- Threading e performance
- Integração de múltiplas bibliotecas

**Próximos passos:**
1. Experimente diferentes detectores (HSV, YOLO, ...)
2. Ajuste Q/R para seu vídeo
3. Adicione novos gráficos de análise
4. Implemente MOT (múltiplos objetos)
5. Publique em GitHub!

---

**Versão:** 1.0  
**Última atualização:** Abril 2026  
**Licença:** Educacional - UFCG
