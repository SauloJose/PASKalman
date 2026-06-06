import threading
import time
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv 

# Importando as classes corrigidas diretamente do seu arquivo world.py
from src.models.world import World, Entidy
from src.gui.viewers import VideoViewer
from src.detec.detector import detect_centroid

import traceback

# Colors (B,G,R)
RED_COLOR     = (0,0,255) 
BLUE_COLOR    = (255,0,0)
GREEN_COLOR   = (0,255,0)
YELLOW_COLOR  = (255,255,0)

class KalmanApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Filtro de Kalman - Demo")
        
        # Video display size
        self.video_width = 640
        self.video_height = 480

        # Set minimum window size
        self.root.minsize(1400, 750)
        
        # Dimensões reais desejadas
        self.max_x = 150 #metros
        self.max_y = 150 #metros

        self.min_window_m = 2.5 #metros

        # Maximize window
        try:
            self.root.state("zoomed")
        except Exception:
            try:
                self.root.attributes("-fullscreen", True)
            except Exception:
                pass
        self.root.bind("<Escape>", self._on_escape)

        # Application state
        self.video_path = None
        self.processed_video_path = None
        self.cap = None
        self.worker = None
        self.running = False
        self.processing = False
        self.playing = False
        self.paused = False
        self.current_frame_idx = 0
        self.total_frames = 0
        self.detection_rate = 0.0  # Métrica de % de detecção estável
        self.meas_inside_roi = 0 

        # Metrics data (Agora armazenados em METROS para os gráficos)
        self.meas_pts = []
        self.filt_pts = []
        self.sqerr_x = []
        self.sqerr_y = []
        self.kalman_windows = []  # Armazena as matrizes P de covariância
        self.pred_pts = []          # posição predita (antes do update)
        self.innovations = []       # diferença z - H*x_pred (2D)
        self.nis_values = []        # inovação normalizada escalar
        self.prior_covs = []        # covariância predita P_pred
        self.measurements_raw = []  # medições em metros (para gráficos de dispersão)

        # Visualization toggle states
        self.show_traj = tk.BooleanVar(value=True)
        self.show_detect = tk.BooleanVar(value=True)
        self.show_kalman = tk.BooleanVar(value=True)
        self.show_window = tk.BooleanVar(value=False)

        # Main layout: left (300) | center (750) | right (450) = proportion 2:5:3
        self.root.grid_columnconfigure(0, weight=0, minsize=300)  
        self.root.grid_columnconfigure(1, weight=1, minsize=750)  
        self.root.grid_columnconfigure(2, weight=0, minsize=450)  
        self.root.grid_rowconfigure(0, weight=1)

        # --- LEFT PANEL: 4 Sections ---
        self.left_frame = tk.Frame(self.root, bg="white", width=300)
        self.left_frame.grid(row=0, column=0, sticky="nsew")
        self.left_frame.grid_propagate(False)

        # Title
        title_lbl = tk.Label(self.left_frame, text="Filtro de Kalman", 
                            font=("Segoe UI", 14, "bold"), bg="white", fg="#333333")
        title_lbl.pack(fill="x", padx=10, pady=(12, 6))
        
        # Separator
        sep1 = tk.Frame(self.left_frame, bg="#e0e0e0", height=1)
        sep1.pack(fill="x", padx=10, pady=4)

        # ===== SECTION 1: Carregar o vídeo =====
        entrada_lbl_frame = tk.LabelFrame(self.left_frame, text="📄 Carregar o vídeo", 
                                         font=("Segoe UI", 10, "bold"), 
                                         bg="#f5f5f5", fg="#333333", padx=10, pady=10)
        entrada_lbl_frame.pack(fill="x", padx=8, pady=(4, 4))

        ttk.Label(entrada_lbl_frame, text="Vídeo carregado:", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.video_path_label = tk.Label(entrada_lbl_frame, text="Nenhum vídeo", 
                                         wraplength=260, font=("Segoe UI", 8), 
                                         fg="#666666", bg="#f5f5f5")
        self.video_path_label.pack(anchor="w", pady=(4, 8))

        self.load_btn = tk.Button(entrada_lbl_frame, text="📁 Load", command=self.load_video, 
                                 font=("Segoe UI", 9, "bold"), bg="#4a4a4a", fg="white", 
                                 relief="flat", padx=10, pady=1, cursor="hand2")
        self.load_btn.pack(fill="x", pady=4)

        # ===== SECTION 2: Opções do Filtro (Modelo 6D Fixo) =====
        config_lbl_frame = tk.LabelFrame(self.left_frame, text="⚙ Opções do Filtro & Sensor", 
                                        font=("Segoe UI", 10, "bold"), 
                                        bg="#f5f5f5", fg="#333333", padx=10, pady=10)
        config_lbl_frame.pack(fill="x", padx=8, pady=(4, 4))

        # Fixed model label
        ttk.Label(config_lbl_frame, text="Modelo: Entidy 6D [x, y, vx, vy, ax, ay]", 
                 font=("Segoe UI", 8, "bold"), foreground="#0066cc").pack(anchor="w", pady=(0, 6))

        #Erro do detector de centroide (Simulação de ruído do sensor)
        ttk.Label(config_lbl_frame, text="Erro do Detector (Ruído em Pixels):", 
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(4, 2))
        self.detector_noise_entry = ttk.Entry(config_lbl_frame, width=12)
        self.detector_noise_entry.insert(0, "1.0")  # Valor padrão de ruído de pixel
        self.detector_noise_entry.pack(anchor="w", pady=(0, 8))

        # Q matrix diagonal inputs (6 values)
        ttk.Label(config_lbl_frame, text="Q - Diagonal (Ruído de Processo):", 
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(4, 4))
        
        q_frame = tk.Frame(config_lbl_frame, bg="#f5f5f5")
        q_frame.pack(fill="x", pady=(0, 6))
        
        q_labels = ["Q[0,0] (x)", "Q[1,1] (y)", "Q[2,2] (vx)", "Q[3,3] (vy)", "Q[4,4] (ax)", "Q[5,5] (ay)"]
        
        # Valores padrão baseados na cinemática: Pos(1e-4), Vel(1e-2), Acel(1.0)
        default_q_vals = ["5e-1", "5e-1", "5e-1", "5e-1", "5e-1", "5e-1"]
        
        self.q_entries = []
        for i, label in enumerate(q_labels):
            col = i % 3
            row = i // 3
            lbl = tk.Label(q_frame, text=label, font=("Segoe UI", 7), bg="#f5f5f5")
            lbl.grid(row=row*2, column=col, sticky="w", padx=2, pady=(2, 0))
            
            entry = ttk.Entry(q_frame, width=8)
            # Insere o valor padrão correspondente ao índice atual
            entry.insert(0, default_q_vals[i])
            entry.grid(row=row*2+1, column=col, sticky="ew", padx=2, pady=(0, 4))
            
            self.q_entries.append(entry)
            
        q_frame.columnconfigure(0, weight=1)
        q_frame.columnconfigure(1, weight=1)
        q_frame.columnconfigure(2, weight=1)

        # R matrix diagonal inputs (2 values)
        ttk.Label(config_lbl_frame, text="R - Diagonal (Ruído de Medição):", 
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(4, 4))
        
        r_frame = tk.Frame(config_lbl_frame, bg="#f5f5f5")
        r_frame.pack(fill="x", pady=(0, 2))
        
        r_labels = ["R[0,0] (x)", "R[1,1] (y)"]
        
        # Valores padrão baseados na variância do erro em metros: 0.1m^2 = 0.01 = 1e-2
        default_r_vals = ["1e-2", "1e-2"]
        
        self.r_entries = []
        for i, label in enumerate(r_labels):
            col = i % 2 
            row = i // 2
            lbl = tk.Label(r_frame, text=label, font=("Segoe UI", 7), bg="#f5f5f5")
            lbl.grid(row=row*2, column=col, sticky="w", padx=2, pady=(2, 0))
            
            entry = ttk.Entry(r_frame, width=8)
            entry.insert(0, default_r_vals[i])
            entry.grid(row=row*2+1, column=col, sticky="ew", padx=2, pady=(0, 4))
            
            self.r_entries.append(entry)
            
        r_frame.columnconfigure(0, weight=1)
        r_frame.columnconfigure(1, weight=1)

        # ===== SECTION 3: Opções de Debug =====
        debug_lbl_frame = tk.LabelFrame(self.left_frame, text="🔍 Opções de Debug", 
                                       font=("Segoe UI", 10, "bold"), 
                                       bg="#f5f5f5", fg="#333333", padx=10, pady=10)
        debug_lbl_frame.pack(fill="x", padx=8, pady=(4, 4))

        ttk.Checkbutton(debug_lbl_frame, text="Desenhar trajetória", variable=self.show_traj).pack(anchor="w", pady=2)
        ttk.Checkbutton(debug_lbl_frame, text="Desenhar detecção", variable=self.show_detect).pack(anchor="w", pady=2)
        ttk.Checkbutton(debug_lbl_frame, text="Desenhar Kalman", variable=self.show_kalman).pack(anchor="w", pady=2)
        ttk.Checkbutton(debug_lbl_frame, text="Desenhar Janela Kalman", variable=self.show_window).pack(anchor="w", pady=2)

        # ===== SECTION 4: EXEC Button =====
        exec_lbl_frame = tk.Frame(self.left_frame, bg="#f5f5f5", padx=10, pady=10)
        exec_lbl_frame.pack(fill="x", padx=8, pady=(4, 8))

        self.exec_btn = tk.Button(exec_lbl_frame, text="▶ EXEC", command=self.execute_processing, 
                                 font=("Segoe UI", 11, "bold"), bg="#333333", fg="white", 
                                 relief="flat", padx=20, pady=1, cursor="hand2", state="disabled")
        self.exec_btn.pack(fill="x")

        # Status label
        self.status_lbl = tk.Label(exec_lbl_frame, text="Status: Aguardando", 
                                  font=("Segoe UI", 8), fg="#666666", bg="#f5f5f5")
        self.status_lbl.pack(pady=(6, 4))

        # SAVE RESULTS Button
        self.save_btn = tk.Button(exec_lbl_frame, text="💾 Salvar Resultados", command=self.save_results, 
                                 font=("Segoe UI", 10, "bold"), bg="#666666", fg="white", 
                                 relief="flat", padx=15, pady=6, cursor="hand2", state="disabled")
        self.save_btn.pack(fill="x", pady=(4, 0))

        # --- CENTER PANEL: Single Viewer ---
        self.center_frame = tk.Frame(self.root, bg="#f9f9f9")
        self.center_frame.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)

        viewer_title = tk.Label(self.center_frame, text="🎬 Rastreamento com Filtro de Kalman", 
                               font=("Segoe UI", 11, "bold"), bg="#f9f9f9", fg="#333333")
        viewer_title.pack(fill="x", padx=0, pady=(0, 10))
        
        # INFO BANNER: Video details
        info_frame = tk.Frame(self.center_frame, bg="#e8e8e8", height=60)
        info_frame.pack(fill="x", pady=(0, 8), padx=0)
        info_frame.pack_propagate(False)
        
        self.video_info_lbl = tk.Label(info_frame, 
                                       text="Arquivo: — | Tamanho: — | FPS: — | Frames: — | Taxa: —",
                                       font=("Segoe UI", 8), fg="#555555", bg="#e8e8e8", 
                                       justify="left", wraplength=700)
        self.video_info_lbl.pack(anchor="w", padx=10, pady=8)
        
        # VIEWER CONTAINER
        viewer_container = tk.Frame(self.center_frame, bg="#f9f9f9")
        viewer_container.pack(fill="both", expand=True, pady=(0, 8))
        
        self.tela_viewer = VideoViewer(viewer_container, width=self.video_width, 
                                       height=self.video_height, bg="white")
        self.tela_viewer.pack()

        # TIME DISPLAY (centered)
        self.time_info_lbl = tk.Label(self.center_frame, text="00:00 / 00:00", 
                                      font=("Segoe UI", 9, "bold"), fg="#333333", bg="#f9f9f9")
        self.time_info_lbl.pack(pady=(0, 6))

        # PLAYBACK CONTROLS (below viewer)
        controls_frame = tk.Frame(self.center_frame, bg="#f9f9f9")
        controls_frame.pack(fill="x", pady=(0, 8))
        
        controls_inner = tk.Frame(controls_frame, bg="#f9f9f9")
        controls_inner.pack(anchor="center")
        
        self.prev_btn = tk.Button(controls_inner, text="◄ Anterior", command=self.prev_frame, 
                                 font=("Segoe UI", 9, "bold"), bg="#4a4a4a", fg="white", 
                                 relief="flat", padx=12, pady=6, state="disabled", cursor="hand2",
                                 activebackground="#666666")
        self.prev_btn.pack(side="left", padx=6)
        
        self.play_btn = tk.Button(controls_inner, text="⏵ Play", command=self.toggle_playback, 
                                 font=("Segoe UI", 9, "bold"), bg="#4a4a4a", fg="white", 
                                 relief="flat", padx=12, pady=6, state="disabled", cursor="hand2",
                                 activebackground="#666666")
        self.play_btn.pack(side="left", padx=6)
        
        self.next_btn = tk.Button(controls_inner, text="Próximo ►", command=self.next_frame, 
                                 font=("Segoe UI", 9, "bold"), bg="#4a4a4a", fg="white", 
                                 relief="flat", padx=12, pady=6, state="disabled", cursor="hand2",
                                 activebackground="#666666")
        self.next_btn.pack(side="left", padx=6)

        # DEBUG LEGEND
        legend_frame = tk.Frame(self.center_frame, bg="#f5f5f5", relief="solid", borderwidth=1)
        legend_frame.pack(fill="x", pady=(0, 0), padx=0)
        
        legend_title = tk.Label(legend_frame, text="📋 Legenda de Debug", 
                               font=("Segoe UI", 9, "bold"), bg="#f5f5f5", fg="#333333")
        legend_title.pack(anchor="w", padx=10, pady=(6, 4))
        
        legend_text = tk.Label(legend_frame, 
                              text="🔴 Detecção (vermelho) = centroide medido pelo detector\n"
                                   "🟦 Kalman (azul) = posição estimada pelo filtro\n"
                                   "🟩 Trajetória (verde) = caminho filtrado\n"
                                   "🟦 Janela Kalman (cyan) = região de incerteza dinâmica (±3σ mapeada de metros para pixels)",
                              font=("Segoe UI", 8), fg="#666666", bg="#f5f5f5", 
                              justify="left")
        legend_text.pack(anchor="w", padx=10, pady=(0, 6))


        # --- RIGHT PANEL: Metrics (Modificados para Metros) ---
        self.right_frame = tk.Frame(self.root, bg="white")
        self.right_frame.grid(row=0, column=2, sticky="nsew", padx=6, pady=6)

        # RMS X
        rms_x_frame = tk.LabelFrame(self.right_frame, text="RMS de X (metros)", 
                                   font=("Segoe UI", 10, "bold"), bg="white", fg="#333333",
                                   padx=4, pady=4, borderwidth=1, relief="solid")
        rms_x_frame.pack(fill="x", pady=(0, 4), expand=False)
        rms_x_frame.pack_propagate(False)
        rms_x_frame.configure(height=140)
        
        self.rmsx_fig = Figure(figsize=(4.2, 1.0), tight_layout=True, facecolor="white")
        self.rmsx_ax = self.rmsx_fig.add_subplot(111)
        self.rmsx_ax.set_facecolor("#f5f5f5")
        self.rmsx_ax.tick_params(colors="#666666", labelsize=8)
        self.rmsx_ax.spines["bottom"].set_color("#999999")
        self.rmsx_ax.spines["left"].set_color("#999999")
        self.rmsx_ax.spines["top"].set_visible(False)
        self.rmsx_ax.spines["right"].set_visible(False)
        self.rmsx_line, = self.rmsx_ax.plot([], [], "#333333", linewidth=2)
        self.rmsx_canvas = FigureCanvasTkAgg(self.rmsx_fig, master=rms_x_frame)
        self.rmsx_canvas.get_tk_widget().pack(fill="both", expand=True)

        # RMS Y
        rms_y_frame = tk.LabelFrame(self.right_frame, text="RMS de Y (metros)", 
                                   font=("Segoe UI", 10, "bold"), bg="white", fg="#333333",
                                   padx=4, pady=4, borderwidth=1, relief="solid")
        rms_y_frame.pack(fill="x", pady=(0, 4), expand=False)
        rms_y_frame.pack_propagate(False)
        rms_y_frame.configure(height=140)
        
        self.rmsy_fig = Figure(figsize=(4.2, 1.0), tight_layout=True, facecolor="white")
        self.rmsy_ax = self.rmsy_fig.add_subplot(111)
        self.rmsy_ax.set_facecolor("#f5f5f5")
        self.rmsy_ax.tick_params(colors="#666666", labelsize=8)
        self.rmsy_ax.spines["bottom"].set_color("#999999")
        self.rmsy_ax.spines["left"].set_color("#999999")
        self.rmsy_ax.spines["top"].set_visible(False)
        self.rmsy_ax.spines["right"].set_visible(False)
        self.rmsy_line, = self.rmsy_ax.plot([], [], "#333333", linewidth=2)
        self.rmsy_canvas = FigureCanvasTkAgg(self.rmsy_fig, master=rms_y_frame)
        self.rmsy_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Trajetória
        traj_frame = tk.LabelFrame(self.right_frame, text="Gráfico da Trajetória (Mundo Real - Metros)", 
                                  font=("Segoe UI", 10, "bold"), bg="white", fg="#333333",
                                  padx=4, pady=4, borderwidth=1, relief="solid")
        traj_frame.pack(fill="both", expand=True, pady=(0, 0))
        
        self.traj_fig = Figure(figsize=(4.2, 3.2), tight_layout=True, facecolor="white")
        self.traj_ax = self.traj_fig.add_subplot(111)
        self.traj_ax.set_facecolor("#f5f5f5")
        self.traj_ax.tick_params(colors="#666666", labelsize=8)
        self.traj_ax.spines["bottom"].set_color("#999999")
        self.traj_ax.spines["left"].set_color("#999999")
        self.traj_ax.spines["top"].set_visible(False)
        self.traj_ax.spines["right"].set_visible(False)
        self.traj_line_meas, = self.traj_ax.plot([], [], "r.-", label="medido", linewidth=2)
        self.traj_line_filt, = self.traj_ax.plot([], [], "#333333", marker=".", label="filtrado", linewidth=2)
        legend = self.traj_ax.legend(facecolor="#f5f5f5", edgecolor="#999999", labelcolor="#333333")
        self.traj_canvas = FigureCanvasTkAgg(self.traj_fig, master=traj_frame)
        self.traj_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Store config
        self.config_Q = None
        self.config_R = None
        self.config_detector_noise = 1.0
        self.video_fps = 30.0
        
        # Pre-calculate metrics for optimization
        self.rmsx_cached = None
        self.rmsy_cached = None
        self.metrics_update_counter = 0

        # Playback poll
        self.root.after(33, self._poll_playback)

    def _on_escape(self, event=None):
        try:
            self.root.state("normal")
        except Exception:
            try:
                self.root.attributes("-fullscreen", False)
            except Exception:
                pass

    def load_video(self):
        """Load a video and show first frame."""
        path = filedialog.askopenfilename(
            title="Selecione um vídeo", 
            filetypes=[("Vídeos", "*.mp4;*.mkv;*.avi;*.mov"), ("All", "*")]
        )
        if not path:
            return
        
        self.video_path = path
        self.video_path_label.config(text=os.path.basename(path))
        
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        
        # Get video properties
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame_idx = 0
        
        # Calculate file size
        file_size_bytes = os.path.getsize(self.video_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        # Get file extension
        file_ext = os.path.splitext(self.video_path)[1].strip(".").upper()
        
        # Calculate total duration
        total_time = self.total_frames / self.video_fps
        total_time_str = self._format_time(total_time)
        
        cap.release()
        
        if not ret:
            messagebox.showerror("Erro", "Não foi possível ler o vídeo")
            return
        
        # Display first frame
        self.tela_viewer.display_image(frame)
        self.time_info_lbl.config(text=f"00:00 / {total_time_str}")
        
        # Update video info banner
        info_text = (f"Arquivo: {os.path.basename(path)} ({file_ext}) | "
                    f"Tamanho: {file_size_mb:.1f} MB | "
                    f"FPS: {self.video_fps:.1f} | "
                    f"Frames: {self.total_frames} | "
                    f"Taxa: {width}×{height} ({total_time_str})")
        self.video_info_lbl.config(text=info_text)
        
        # Enable EXEC button
        self.exec_btn.config(state="normal")
        self.status_lbl.config(text="Status: Pronto para executar")

    def execute_processing(self):
        """Process video and save to src/data/"""
        if not self.video_path or self.processing:
            return
        
        try:
            self._parse_config()
        except Exception as e:
            error_msg = traceback.format_exc()
            self.root.after(0, lambda: messagebox.showerror("Erro", f"Erro nas configurações: {error_msg}"))
            self.processing = False
            return  # Para a execução IMEDIATAMENTE se houver erro!
        
        # Inicia o processamento apenas se o try acima deu certo
        self.processing = True
        self.exec_btn.config(state="disabled")
        self.load_btn.config(state="disabled")
        self.status_lbl.config(text="Status: Processando...")
        
        self.worker = threading.Thread(target=self._process_video, daemon=True)
        self.worker.start()

    def _parse_config(self):
        """Parse Q, R and detector noise from UI."""
        # Parse Q from individual entries
        q_vals = []
        for entry in self.q_entries:
            val_str = entry.get().strip()
            try:
                q_vals.append(float(val_str))
            except ValueError:
                q_vals.append(1e-2)
        
        # Parse R from individual entries
        r_vals = []
        for entry in self.r_entries:
            val_str = entry.get().strip()
            try:
                r_vals.append(float(val_str))
            except ValueError:
                r_vals.append(1e-1)
        
        # Parse detector noise value
        try:
            self.config_detector_noise = float(self.detector_noise_entry.get().strip())
        except ValueError:
            self.config_detector_noise = 4.0

        self.config_Q = q_vals[:6]  
        self.config_R = r_vals[:2]  

    def _process_video(self):
        """Process video with Kalman filter and save to src/data/."""
        cap = None 
        out = None
        try:
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.fps = fps 
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            os.makedirs("FiltroKalman/src/data", exist_ok=True)
            output_path = "FiltroKalman/src/data/output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
            self.world_m = World(dimPX=w, dimPY=h, dimPOX=self.max_x, dimPOY=self.max_y)
            dt = 1.0 / fps
            
            q_vals = self.config_Q if len(self.config_Q) >= 6 else self.config_Q + [1e-1] * (6 - len(self.config_Q))
            r_vals = self.config_R[:2] if len(self.config_R) >= 2 else self.config_R + [1e-1]

            kf = Entidy(dt=dt, q_diag=q_vals[:6], r_diag=r_vals[:2])
            
            self.saved_Q = kf.Q if hasattr(kf, 'Q') else np.diag(q_vals[:6])
            self.saved_R = kf.R if hasattr(kf, 'R') else np.diag(r_vals[:2])
            self.saved_Qd = getattr(kf, 'Qd', getattr(kf, 'Q_discrete', getattr(kf, 'Q_d', None)))

            frame_count = 0
            frames_with_meas = 0        
            self.meas_inside_roi = 0         

            self.meas_pts = []
            self.filt_pts = []
            self.sqerr_x = []
            self.sqerr_y = []
            self.kalman_windows = []
            self.nis_vals = []
            self.innov_x = []           
            self.innov_y = []           

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                meas_px = detect_centroid(frame, noise_std=self.config_detector_noise)
                meas_m = self.world_m.img2world(meas_px[0], meas_px[1]) if meas_px is not None else None
                
                # 1. PREDIÇÃO (A Priori)
                kf.predict()
                pos_pred = kf.get_position()  
                P_pred = kf.P.copy() if hasattr(kf, 'P') else None

                # Calcula o ROI Previsto (Matriz S) independentemente de haver medição
                # Isso permite desenhar o ROI (área de busca) durante o vídeo todo
                if P_pred is not None:
                    S = P_pred[:2, :2] + np.diag(r_vals[:2])
                    std_innov_x_m = max(self.min_window_m, np.sqrt(S[0, 0]) * 3)
                    std_innov_y_m = max(self.min_window_m, np.sqrt(S[1, 1]) * 3)
                else:
                    std_innov_x_m = self.min_window_m
                    std_innov_y_m = self.min_window_m

                if meas_m is not None:
                    innov = np.array([meas_m[0] - pos_pred[0], meas_m[1] - pos_pred[1]])
                    self.innov_x.append(innov[0])
                    self.innov_y.append(innov[1])

                    if P_pred is not None:
                        try:
                            invS = np.linalg.inv(S)
                            nis = innov @ invS @ innov
                        except np.linalg.LinAlgError:
                            nis = np.nan
                    else:
                        nis = np.nan
                        
                    self.nis_vals.append(nis)

                    # --- VALIDAÇÃO DA MEDIÇÃO (GATING) ---
                    dist_x = abs(innov[0])
                    dist_y = abs(innov[1])

                    if dist_x <= std_innov_x_m and dist_y <= std_innov_y_m:
                        self.meas_inside_roi += 1
                        
                    # 2. ATUALIZAÇÃO (A Posteriori)
                    kf.update(meas_m)
                    frames_with_meas += 1

                else:
                    self.innov_x.append(np.nan)
                    self.innov_y.append(np.nan)
                    self.nis_vals.append(np.nan)

                # 3. GUARDAR ESTADOS FINAIS (A Posteriori)
                est_m = kf.get_position()
                P_mat = kf.P.copy() if hasattr(kf, 'P') else None
                self.kalman_windows.append(P_mat)

                if meas_m is None:
                    self.meas_pts.append(None)
                    self.sqerr_x.append(np.nan)
                    self.sqerr_y.append(np.nan)
                else:
                    mx, my = float(meas_m[0]), float(meas_m[1])
                    ex, ey = float(est_m[0]), float(est_m[1])
                    self.meas_pts.append((mx, my))
                    dx, dy = ex - mx, ey - my
                    self.sqerr_x.append(dx ** 2)
                    self.sqerr_y.append(dy ** 2)

                self.filt_pts.append((ex, ey))

                # --- RENDERIZAÇÃO GRÁFICA ---
                ann = frame.copy()
                if self.show_traj.get() and len(self.filt_pts) >= 2:
                    filt_poly = [self.world_m.world2img(p[0], p[1]) for p in self.filt_pts if p is not None]
                    if len(filt_poly) >= 2:
                        cv2.polylines(ann, [np.array(filt_poly, dtype=np.int32)], False, YELLOW_COLOR, 2)

                if self.show_detect.get() and len(self.meas_pts) >= 2:
                    meas_poly = [self.world_m.world2img(p[0], p[1]) for p in self.meas_pts if p is not None]
                    if len(meas_poly) >= 2:
                        cv2.polylines(ann, [np.array(meas_poly, dtype=np.int32)], False, GREEN_COLOR, 1)

                # DESENHA A JANELA DE KALMAN (ROI PREVISTO PARA ESTE FRAME) E O CENTRO
                if self.show_window.get() and pos_pred is not None:
                    # Usa a predição e std_innov (S) para desenhar a área esperada
                    px1, py1 = self.world_m.world2img(pos_pred[0] - std_innov_x_m, pos_pred[1] + std_innov_y_m)
                    px2, py2 = self.world_m.world2img(pos_pred[0] + std_innov_x_m, pos_pred[1] - std_innov_y_m)
                    
                    # Retângulo Verde
                    cv2.rectangle(ann, 
                                  (max(0, min(w, int(px1))), max(0, min(h, int(py1)))),
                                  (max(0, min(w, int(px2))), max(0, min(h, int(py2)))),
                                  (0, 255, 0), 2)
                    
                    # Ponto verde bem pequeno no centro do ROI (Predição)
                    cx, cy = self.world_m.world2img(pos_pred[0], pos_pred[1])
                    cv2.circle(ann, (int(cx), int(cy)), 2, (0, 255, 0), -1)

                # Desenha o Kalman Atualizado (A Posteriori) como bolinha azul
                est_px = self.world_m.world2img(est_m[0], est_m[1])
                if self.show_kalman.get():
                    cv2.circle(ann, (int(est_px[0]), int(est_px[1])), 6, BLUE_COLOR, -1)

                # Desenha a Detecção como bolinha (Ciano ou vermelha se estiver fora, opcional no futuro)
                if self.show_detect.get() and meas_m is not None:
                    valid_px = self.world_m.world2img(meas_m[0], meas_m[1])
                    cv2.circle(ann, (int(valid_px[0]), int(valid_px[1])), 6, (255, 0, 0), -1)

                out.write(ann)
                frame_count += 1

            self.total_frames = frame_count
            
            self.detection_rate = (frames_with_meas / frame_count * 100.0) if frame_count > 0 else 0.0
            self.inlier_rate = (self.meas_inside_roi / frames_with_meas * 100.0) if frames_with_meas > 0 else 0.0
            
            self.sensor_detection_rate = self.detection_rate
            self.roi_accuracy_rate = self.inlier_rate

            self.processed_video_path = output_path
            self.root.after(0, self._on_processing_complete)
        except Exception as e:
            errorMsg = str(e)
            self.root.after(0, lambda: messagebox.showerror("Erro", f"Erro ao processar: {errorMsg}"))
        finally:
            self.processing = False
            if cap is not None: cap.release()
            if out is not None: out.release()

    def _find_convergence_frame(self, running_rms, final_val, tol=0.05, min_stable=10):
        """
        Retorna o índice (em frames válidos) em que o running RMS fica consistentemente
        <= (final_val * (1 + tol)) por pelo menos min_stable pontos consecutivos.
        """
        if final_val is None or len(running_rms) < min_stable:
            return None
        threshold = final_val * (1 + tol)
        start = max(1, len(running_rms) // 5)  # ignora os primeiros 20%
        for i in range(start, len(running_rms) - min_stable + 1):
            if all(v <= threshold for v in running_rms[i:i+min_stable]):
                return i
        return None

    def save_results(self):
        """Salva gráficos detalhados, relatório e dados brutos em CSV em src/results/"""
        if not self.filt_pts or not self.meas_pts:
            messagebox.showwarning("Aviso", "Nenhum resultado para salvar.")
            return

        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            video_name = os.path.splitext(os.path.basename(self.video_path))[0] if self.video_path else "results"
            save_dir = f"FiltroKalman/src/results/{video_name}"
            os.makedirs(save_dir, exist_ok=True)

            # ========== Preparação de dados ==========
            signed_dx = []
            signed_dy = []
            for m_pt, f_pt in zip(self.meas_pts, self.filt_pts):
                if m_pt is not None:
                    signed_dx.append(f_pt[0] - m_pt[0])
                    signed_dy.append(f_pt[1] - m_pt[1])

            valid_nis = [n for n in self.nis_vals if not np.isnan(n)]

            sx = np.array([v for v in self.sqerr_x if not np.isnan(v)])
            sy = np.array([v for v in self.sqerr_y if not np.isnan(v)])
            run_rms_x = np.sqrt(np.cumsum(sx) / np.arange(1, sx.size + 1)) if sx.size > 0 else []
            run_rms_y = np.sqrt(np.cumsum(sy) / np.arange(1, sy.size + 1)) if sy.size > 0 else []

            # ===== GRÁFICO 1: TRAJETÓRIA =====
            fig1 = Figure(figsize=(12, 8), tight_layout=True, dpi=150)
            ax1 = fig1.add_subplot(111)
            xs_meas = [p[0] for p in self.meas_pts if p is not None]
            ys_meas = [p[1] for p in self.meas_pts if p is not None]
            xs_filt = [p[0] for p in self.filt_pts]
            ys_filt = [p[1] for p in self.filt_pts]
            ax1.plot(xs_meas, ys_meas, "r.-", label="Medido", linewidth=1, markersize=4, alpha=0.5)
            ax1.plot(xs_filt, ys_filt, "#333333", label="Kalman Filtrado", linewidth=2.5, alpha=0.9)
            ax1.set_xlabel("Posição X (metros)", fontweight="bold")
            ax1.set_ylabel("Posição Y (metros)", fontweight="bold")
            ax1.set_title(f"Trajetória Espacial | Detecções: {self.detection_rate:.1f}% | Inliers: {self.inlier_rate:.1f}%", fontweight="bold")
            ax1.legend()
            ax1.grid(True, alpha=0.3, linestyle="--")
            fig1.savefig(f"{save_dir}/{video_name}_1_traj.png")

            # ===== GRÁFICO 2: RMS ACUMULADO + CONVERGÊNCIA =====
            fig2 = Figure(figsize=(12, 6), tight_layout=True, dpi=150)
            ax2 = fig2.add_subplot(111)
            if len(run_rms_x) > 0:
                ax2.plot(run_rms_x, label="RMS X", color="blue")
                ax2.plot(run_rms_y, label="RMS Y", color="orange")
                final_x = run_rms_x[-1] if len(run_rms_x) > 0 else None
                final_y = run_rms_y[-1] if len(run_rms_y) > 0 else None
                conv_idx_x = self._find_convergence_frame(run_rms_x, final_x)
                conv_idx_y = self._find_convergence_frame(run_rms_y, final_y)
                if conv_idx_x is not None:
                    ax2.axvline(conv_idx_x, color='blue', linestyle=':', alpha=0.7,
                                label=f"Conv. X: {conv_idx_x/self.fps:.2f}s")
                if conv_idx_y is not None:
                    ax2.axvline(conv_idx_y, color='orange', linestyle=':', alpha=0.7,
                                label=f"Conv. Y: {conv_idx_y/self.fps:.2f}s")
            ax2.set_xlabel("Frames", fontweight="bold")
            ax2.set_ylabel("Erro RMS Acumulado (metros)", fontweight="bold")
            ax2.set_title("Evolução do Erro RMS", fontweight="bold")
            ax2.legend()
            ax2.grid(True, alpha=0.3, linestyle="--")
            fig2.savefig(f"{save_dir}/{video_name}_2_rms.png")

            # ===== GRÁFICO 3: DISPERSÃO (ERROS ASSINADOS) =====
            fig3 = Figure(figsize=(8, 8), tight_layout=True, dpi=150)
            ax3 = fig3.add_subplot(111)
            ax3.scatter(signed_dx, signed_dy, alpha=0.5, c='purple', edgecolors='k', s=20)
            ax3.axhline(0, color='black', linewidth=1)
            ax3.axvline(0, color='black', linewidth=1)
            ax3.set_xlabel("Erro X (m)", fontweight="bold")
            ax3.set_ylabel("Erro Y (m)", fontweight="bold")
            ax3.set_title("Dispersão dos Erros de Estimação (assinados)", fontweight="bold")
            ax3.grid(True, alpha=0.3, linestyle="--")
            fig3.savefig(f"{save_dir}/{video_name}_3_scatter.png")

            # ===== GRÁFICO 4: NIS =====
            fig4 = Figure(figsize=(12, 6), tight_layout=True, dpi=150)
            ax4 = fig4.add_subplot(111)
            frames_nis = [i for i, n in enumerate(self.nis_vals) if not np.isnan(n)]
            if valid_nis:
                ax4.plot(frames_nis, valid_nis, 'm-', alpha=0.7, label="NIS Calculado")
                ax4.axhline(5.99, color='r', linestyle='--', linewidth=2, label="Limite 95% Confiança (χ²)")
                ax4.set_ylim(0, max(15, np.percentile(valid_nis, 95) * 1.5))
            ax4.set_xlabel("Frames", fontweight="bold")
            ax4.set_ylabel("Valor NIS", fontweight="bold")
            ax4.set_title("Teste de Consistência NIS (covariância corrigida)", fontweight="bold")
            ax4.legend()
            ax4.grid(True, alpha=0.3, linestyle="--")
            fig4.savefig(f"{save_dir}/{video_name}_4_nis.png")

            # ===== GRÁFICO 5: HISTOGRAMA DOS ERROS =====
            fig5 = Figure(figsize=(10, 6), tight_layout=True, dpi=150)
            ax5 = fig5.add_subplot(111)
            ax5.hist(signed_dx, bins=30, alpha=0.5, color='blue', label='Erros X (m)')
            ax5.hist(signed_dy, bins=30, alpha=0.5, color='orange', label='Erros Y (m)')
            ax5.set_xlabel("Erro (metros)", fontweight="bold")
            ax5.set_ylabel("Frequência", fontweight="bold")
            ax5.set_title("Histograma dos Erros de Estado (assinados)", fontweight="bold")
            ax5.legend()
            ax5.grid(True, alpha=0.3, linestyle="--")
            fig5.savefig(f"{save_dir}/{video_name}_5_hist.png")

            plt.close('all')

            # ========== CÁLCULO DE MÉTRICAS ==========
            rmse_x_total = np.sqrt(np.nanmean(self.sqerr_x)) if self.sqerr_x else 0.0
            rmse_y_total = np.sqrt(np.nanmean(self.sqerr_y)) if self.sqerr_y else 0.0
            mean_signed_dx = np.mean(signed_dx) if signed_dx else 0.0
            mean_signed_dy = np.mean(signed_dy) if signed_dy else 0.0
            std_dx = np.std(signed_dx) if signed_dx else 0.0
            std_dy = np.std(signed_dy) if signed_dy else 0.0
            max_err_x = np.max(np.abs(signed_dx)) if signed_dx else 0.0
            max_err_y = np.max(np.abs(signed_dy)) if signed_dy else 0.0

            mean_nis = np.mean(valid_nis) if valid_nis else 0.0
            nis_above_95 = sum(1 for n in valid_nis if n > 5.99)
            nis_pct_above = (nis_above_95 / len(valid_nis)) * 100 if valid_nis else 0.0

            conv_idx_x = self._find_convergence_frame(run_rms_x, run_rms_x[-1] if len(run_rms_x) else None)
            conv_idx_y = self._find_convergence_frame(run_rms_y, run_rms_y[-1] if len(run_rms_y) else None)
            conv_time_x = conv_idx_x / self.fps if conv_idx_x is not None else None
            conv_time_y = conv_idx_y / self.fps if conv_idx_y is not None else None

            # ========== RELATÓRIO TXT ==========
            txt_path = f"{save_dir}/{video_name}_metrics.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("====================================================\n")
                f.write("      RESUMO DE MÉTRICAS - FILTRO DE KALMAN         \n")
                f.write("====================================================\n\n")
                f.write(f"Arquivo Fonte: {self.video_path}\n")
                f.write(f"Data da Análise: {time.strftime('%d/%m/%Y %H:%M:%S')}\n")
                f.write(f"Total de Frames: {self.total_frames}\n")
                f.write(f"FPS: {self.fps:.2f}\n\n")

                fmt_opts = {'precision': 6, 'suppress_small': True, 'separator': '  '}
                f.write("--- PARÂMETROS DO FILTRO ---\n")
                if hasattr(self, 'saved_Q') and self.saved_Q is not None:
                    f.write("Matriz de Ruído de Processo Contínuo (Q):\n")
                    f.write(f"{np.array2string(np.array(self.saved_Q), **fmt_opts)}\n\n")
                if hasattr(self, 'saved_Qd') and self.saved_Qd is not None:
                    f.write("Matriz de Ruído de Processo Discretizada (Qd):\n")
                    f.write(f"{np.array2string(np.array(self.saved_Qd), **fmt_opts)}\n\n")
                else:
                    f.write("Matriz de Ruído de Processo Discretizada (Qd): [Não Disponível]\n\n")
                if hasattr(self, 'saved_R') and self.saved_R is not None:
                    f.write("Matriz de Ruído de Medição (R):\n")
                    f.write(f"{np.array2string(np.array(self.saved_R), **fmt_opts)}\n\n")
                f.write("----------------------------------------------------\n\n")

                f.write("--- TAXAS DE DETECÇÃO E CONSISTÊNCIA ---\n")
                f.write(f"Taxa de Detecção (frames com medição): {self.detection_rate:.2f}%\n")
                f.write(f"Taxa de Inliers (medições dentro da janela 3σ): {self.inlier_rate:.2f}%\n\n")

                f.write("--- ERROS DE ESTIMAÇÃO (METROS) ---\n")
                f.write(f"RMSE X: {rmse_x_total:.4f} m\n")
                f.write(f"RMSE Y: {rmse_y_total:.4f} m\n")
                f.write(f"Erro Médio (viés) em X: {mean_signed_dx:+.4f} m\n")
                f.write(f"Erro Médio (viés) em Y: {mean_signed_dy:+.4f} m\n")
                f.write(f"Desvio Padrão Erro X: {std_dx:.4f} m\n")
                f.write(f"Desvio Padrão Erro Y: {std_dy:.4f} m\n")
                f.write(f"Erro Máximo Absoluto X: {max_err_x:.4f} m\n")
                f.write(f"Erro Máximo Absoluto Y: {max_err_y:.4f} m\n\n")

                f.write("--- CONVERGÊNCIA DO RMS ---\n")
                if conv_time_x is not None:
                    f.write(f"RMS X convergiu em {conv_idx_x} frames ({conv_time_x:.2f} s)\n")
                else:
                    f.write("RMS X não atingiu convergência dentro do vídeo.\n")
                if conv_time_y is not None:
                    f.write(f"RMS Y convergiu em {conv_idx_y} frames ({conv_time_y:.2f} s)\n")
                else:
                    f.write("RMS Y não atingiu convergência dentro do vídeo.\n")
                f.write("(Critério: erro RMS ≤ 5% do valor final por 10 frames consecutivos)\n\n")

                f.write("--- AVALIAÇÃO DE CONSISTÊNCIA (NIS) ---\n")
                f.write(f"NIS Médio (ideal ≈ 2): {mean_nis:.4f}\n")
                f.write(f"Percentual acima do limite 95% (5.99): {nis_pct_above:.2f}%\n")
                f.write(" * Nota: O NIS avalia se a covariância reflete a real incerteza do modelo.\n")
                f.write("   Uma porcentagem acima de ~5% no limite indica que o filtro está subestimando\n")
                f.write("   o ruído ou divergindo levemente.\n\n")
                f.write("====================================================\n")

            # ========== CSV ==========
            csv_path = f"{save_dir}/{video_name}_positions.csv"
            with open(csv_path, mode='w', newline='', encoding='utf-8') as f_csv:
                writer = csv.writer(f_csv, delimiter=',')
                writer.writerow(["Frame", "Meas_X(m)", "Meas_Y(m)", "Filt_X(m)", "Filt_Y(m)"])
                for frame_idx, (m_pt, f_pt) in enumerate(zip(self.meas_pts, self.filt_pts)):
                    mx = f"{m_pt[0]:.4f}" if m_pt is not None else ""
                    my = f"{m_pt[1]:.4f}" if m_pt is not None else ""
                    fx = f"{f_pt[0]:.4f}" if f_pt is not None else ""
                    fy = f"{f_pt[1]:.4f}" if f_pt is not None else ""
                    writer.writerow([frame_idx, mx, my, fx, fy])

            messagebox.showinfo("Sucesso", f"Análise completa e arquivo .csv salvos em:\n{save_dir}/")

        except Exception as e:
            error_msg = traceback.format_exc()
            self.root.after(0, lambda: messagebox.showerror("Erro", f"Erro ao gerar relatórios: {error_msg}"))
        finally:
            self.processing = False
            
    def _on_processing_complete(self):
        """Seletor de interface chamado ao finalizar o processamento."""
        self.status_lbl.config(text=f"Status: Concluído | Detecção: {self.inlier_rate:.1f}%")
        self.exec_btn.config(state="normal")
        self.load_btn.config(state="normal")

        if self.processed_video_path:
            self.cap = cv2.VideoCapture(self.processed_video_path)
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame_idx = 0

            ret, frame = self.cap.read()
            if ret:
                self.tela_viewer.display_image(frame)
                total_time = self.total_frames / self.video_fps
                self.time_info_lbl.config(text=f"00:00 / {self._format_time(total_time)}")

            self.prev_btn.config(state="normal")
            self.play_btn.config(state="normal")
            self.next_btn.config(state="normal")
            self.save_btn.config(state="normal")

            self._update_metrics_plots()
    def toggle_playback(self):
        if not self.cap:
            return
        self.playing = not self.playing
        self.play_btn.config(text="⏸" if self.playing else "⏵")

    def next_frame(self):
        if not self.cap:
            return
        self.playing = False
        self.play_btn.config(text="⏵")
        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self._display_current_frame()

    def prev_frame(self):
        if not self.cap:
            return
        self.playing = False
        self.play_btn.config(text="⏵")
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self._display_current_frame()

    def _display_current_frame(self):
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.tela_viewer.display_image(frame)
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            current_str = self._format_time(self.current_frame_idx / fps)
            total_str = self._format_time(self.total_frames / fps)
            self.time_info_lbl.config(text=f"{current_str} / {total_str}")
            self._update_metrics_plots()

    def _poll_playback(self):
        if self.playing and self.cap and self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self._display_current_frame()
        elif self.playing and self.current_frame_idx >= self.total_frames - 1:
            self.playing = False
            self.play_btn.config(text="⏵")
        self.root.after(33, self._poll_playback)
    
    def _format_time(self, seconds):
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

    def _update_metrics_plots(self):
        """Atualiza dinamicamente as curvas de telemetria baseando-se no espaço métrico (m)."""
        self.metrics_update_counter += 1
        if self.metrics_update_counter % 3 != 0:  
            return
        
        current_idx = self.current_frame_idx + 1
        
        # RMS X (metros)
        sx = np.array([v for v in self.sqerr_x[:current_idx] if not (v is None or np.isnan(v))], dtype=float)
        if sx.size > 0:
            running_mean_x = np.cumsum(sx) / np.arange(1, sx.size + 1)
            running_rms_x = np.sqrt(running_mean_x)
            self.rmsx_line.set_data(range(len(running_rms_x)), running_rms_x)
            self.rmsx_ax.set_xlim(0, max(10, len(running_rms_x)))
            self.rmsx_ax.set_ylim(0, max(0.1, running_rms_x.max() * 1.2))
        
        # RMS Y (metros)
        sy = np.array([v for v in self.sqerr_y[:current_idx] if not (v is None or np.isnan(v))], dtype=float)
        if sy.size > 0:
            running_mean_y = np.cumsum(sy) / np.arange(1, sy.size + 1)
            running_rms_y = np.sqrt(running_mean_y)
            self.rmsy_line.set_data(range(len(running_rms_y)), running_rms_y)
            self.rmsy_ax.set_xlim(0, max(10, len(running_rms_y)))
            self.rmsy_ax.set_ylim(0, max(0.1, running_rms_y.max() * 1.2))
        
        # Trajetória Cartesiana em Metros (Orientação padrão: y cresce para cima)
        xs_meas = [p[0] for p in self.meas_pts[:current_idx] if p is not None]
        ys_meas = [p[1] for p in self.meas_pts[:current_idx] if p is not None]
        xs_filt = [p[0] for p in self.filt_pts[:current_idx]]
        ys_filt = [p[1] for p in self.filt_pts[:current_idx]]
        
        if xs_meas and ys_meas:
            self.traj_line_meas.set_data(xs_meas, ys_meas)
        self.traj_line_filt.set_data(xs_filt, ys_filt)
        
        if xs_filt and ys_filt:
            x_max = max(xs_filt)
            y_max = max(ys_filt)
            self.traj_ax.set_xlim(0, max(2.0, x_max * 1.1))
            self.traj_ax.set_ylim(0, max(2.0, y_max * 1.1)) # Removida a inversão de tela no gráfico Matplotlib!
        
        self.rmsx_canvas.draw_idle()
        self.rmsy_canvas.draw_idle()
        self.traj_canvas.draw_idle()

def run_app():
    root = tk.Tk()
    app = KalmanApp(root)
    root.mainloop()

if __name__ == "__main__":
    run_app()