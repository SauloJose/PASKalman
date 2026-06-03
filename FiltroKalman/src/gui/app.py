import threading
import time
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.kalman.models import SixDModel
from src.gui.viewers import VideoViewer
from src.detec.detector import detect_centroid


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

        # Metrics data
        self.meas_pts = []
        self.filt_pts = []
        self.sqerr_x = []
        self.sqerr_y = []
        self.kalman_windows = []  # Store P matrices for window visualization

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
                                 relief="flat", padx=10, pady=6, cursor="hand2")
        self.load_btn.pack(fill="x", pady=4)

        # ===== SECTION 2: Opções do Filtro (Modelo 6D Fixo) =====
        config_lbl_frame = tk.LabelFrame(self.left_frame, text="⚙ Opções do Filtro", 
                                        font=("Segoe UI", 10, "bold"), 
                                        bg="#f5f5f5", fg="#333333", padx=10, pady=10)
        config_lbl_frame.pack(fill="x", padx=8, pady=(4, 4))

        # Fixed model label
        ttk.Label(config_lbl_frame, text="Modelo: 6D [x, y, vx, vy, ax, ay]", 
                 font=("Segoe UI", 8, "bold"), foreground="#0066cc").pack(anchor="w", pady=(0, 8))

        # Q matrix diagonal inputs (6 values)
        ttk.Label(config_lbl_frame, text="Q - Diagonal (Ruído de Processo):", 
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(6, 4))
        
        q_frame = tk.Frame(config_lbl_frame, bg="#f5f5f5")
        q_frame.pack(fill="x", pady=(0, 6))
        
        q_labels = ["Q[0,0]\n(x)", "Q[1,1]\n(y)", "Q[2,2]\n(vx)", "Q[3,3]\n(vy)", "Q[4,4]\n(ax)", "Q[5,5]\n(ay)"]
        self.q_entries = []
        for i, label in enumerate(q_labels):
            col = i % 3
            row = i // 3
            lbl = tk.Label(q_frame, text=label, font=("Segoe UI", 7), bg="#f5f5f5")
            lbl.grid(row=row*2, column=col, sticky="w", padx=2, pady=(2, 0))
            entry = ttk.Entry(q_frame, width=8)
            entry.insert(0, "1e-2" if i < 2 else "1e-1")
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
        
        r_labels = ["R[0,0]\n(x)", "R[1,1]\n(y)"]
        self.r_entries = []
        for i, label in enumerate(r_labels):
            lbl = tk.Label(r_frame, text=label, font=("Segoe UI", 7), bg="#f5f5f5")
            lbl.grid(row=0, column=i, sticky="w", padx=2, pady=(2, 0))
            entry = ttk.Entry(r_frame, width=10)
            entry.insert(0, "1e-1")
            entry.grid(row=1, column=i, sticky="ew", padx=2, pady=(0, 2))
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
                                 relief="flat", padx=20, pady=8, cursor="hand2", state="disabled")
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
                                   "🟦 Janela Kalman (cyan) = região de incerteza (±3σ da covariância P)",
                              font=("Segoe UI", 8), fg="#666666", bg="#f5f5f5", 
                              justify="left")
        legend_text.pack(anchor="w", padx=10, pady=(0, 6))


        # --- RIGHT PANEL: Metrics ---
        self.right_frame = tk.Frame(self.root, bg="white")
        self.right_frame.grid(row=0, column=2, sticky="nsew", padx=6, pady=6)

        # RMS X
        rms_x_frame = tk.LabelFrame(self.right_frame, text="RMS de X", 
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
        rms_y_frame = tk.LabelFrame(self.right_frame, text="RMS de Y", 
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
        traj_frame = tk.LabelFrame(self.right_frame, text="Gráfico da Trajetória", 
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
        
        if self.config_Q is None or self.config_R is None:
            try:
                self._parse_config()
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao processar config: {e}")
                return
        
        self.processing = True
        self.exec_btn.config(state="disabled")
        self.load_btn.config(state="disabled")
        self.status_lbl.config(text="Status: Processando...")
        
        self.worker = threading.Thread(target=self._process_video, daemon=True)
        self.worker.start()

    def _parse_config(self):
        """Parse Q and R from UI."""
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
        
        self.config_Q = q_vals[:6]  # Ensure 6 values for 6D model
        self.config_R = r_vals[:2]  # Ensure 2 values for measurements

    def _process_video(self):
        """Process video with Kalman filter and save to src/data/."""
        try:
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create output directory
            os.makedirs("FiltroKalman/src/data", exist_ok=True)
            
            # Output video
            output_path = "FiltroKalman/src/data/output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
            dt = 1.0 / fps
            
            # Always use 6D model
            q_vals = self.config_Q if len(self.config_Q) >= 6 else self.config_Q + [1e-1] * (6 - len(self.config_Q))
            r_vals = self.config_R[:2] if len(self.config_R) >= 2 else self.config_R + [1e-1]
            kf = SixDModel(dt=dt, q_diag=q_vals[:6], r_diag=r_vals[:2])
            
            frame_count = 0
            self.meas_pts = []
            self.filt_pts = []
            self.sqerr_x = []
            self.sqerr_y = []
            self.kalman_windows = []  # Store P matrices for window visualization
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                meas = detect_centroid(frame) ########## ESSA É A LINHA EM QUE EU MODIFICO O CÓDIGO DE DETECÇÃO
                kf.predict()
                if meas is not None:
                    kf.update(meas)
                est = kf.get_position()
                
                # Store P matrix (posteriori covariance) for window visualization
                self.kalman_windows.append(kf.P.copy() if hasattr(kf, 'P') else None)
                
                if meas is None:
                    self.meas_pts.append(None)
                    self.sqerr_x.append(np.nan)
                    self.sqerr_y.append(np.nan)
                else:
                    mx, my = float(meas[0]), float(meas[1])
                    ex, ey = float(est[0]), float(est[1])
                    self.meas_pts.append((mx, my))
                    self.sqerr_x.append((ex - mx) ** 2)
                    self.sqerr_y.append((ey - my) ** 2)
                self.filt_pts.append((float(est[0]), float(est[1])))
                
                # Draw annotations
                ann = frame.copy()
                
                # Draw trajectory
                if self.show_traj.get() and len(self.filt_pts) >= 2:
                    filt_pts_scaled = [tuple(np.array(p, dtype=np.int32)) for p in self.filt_pts]
                    filt_poly = [p for p in filt_pts_scaled if p is not None]
                    if len(filt_poly) >= 2:
                        cv2.polylines(ann, [np.array(filt_poly, dtype=np.int32)], False, YELLOW_COLOR, 2)
                
                # Draw detection
                if self.show_detect.get() and len(self.meas_pts) >= 2:
                    meas_pts_scaled = [tuple(np.array(p, dtype=np.int32)) if p else None for p in self.meas_pts]
                    meas_poly = [p for p in meas_pts_scaled if p is not None]
                    if len(meas_poly) >= 2:
                        cv2.polylines(ann, [np.array(meas_poly, dtype=np.int32)], False, GREEN_COLOR, 1)
                
                # Draw Kalman window (rectangle based on P covariance matrix)
                if self.show_window.get() and frame_count < len(self.kalman_windows) and self.kalman_windows[frame_count] is not None:
                    P_mat = self.kalman_windows[frame_count]
                    # Extract variances from diagonal (posteriori covariance)
                    var_x = float(P_mat[0, 0]) if P_mat.shape[0] > 0 else 0
                    var_y = float(P_mat[1, 1]) if P_mat.shape[1] > 1 else 0
                    
                    # Calculate window dimensions (3-sigma confidence ellipse)
                    std_x = max(np.sqrt(var_x) * 3, 1)
                    std_y = max(np.sqrt(var_y) * 3, 1)
                    
                    # Draw rectangle centered at Kalman estimate
                    if self.filt_pts:
                        center_x = int(self.filt_pts[-1][0])
                        center_y = int(self.filt_pts[-1][1])
                        
                        # Rectangle from (center - std) to (center + std)
                        x1 = max(0, int(center_x - std_x))
                        y1 = max(0, int(center_y - std_y))
                        x2 = min(w, int(center_x + std_x))
                        y2 = min(h, int(center_y + std_y))
                        
                        # Draw semi-transparent rectangle
                        cv2.rectangle(ann, (x1, y1), (x2, y2), (255, 200, 0), 2)  # Cyan color
                
                # Draw Kalman circles
                if self.show_kalman.get() and self.filt_pts:
                    est_pt = tuple(np.array(self.filt_pts[-1], dtype=np.int32))
                    cv2.circle(ann, est_pt, 6, BLUE_COLOR, -1)
                
                if self.show_detect.get() and self.meas_pts and self.meas_pts[-1] is not None:
                    meas_pt = tuple(np.array(self.meas_pts[-1], dtype=np.int32))
                    cv2.circle(ann, meas_pt, 6, (255, 0, 0), -1)
                
                out.write(ann)
                frame_count += 1
            
            cap.release()
            out.release()
            
            self.processed_video_path = output_path
            self.root.after(0, self._on_processing_complete)
        
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Erro", f"Erro ao processar: {e}"))
        
        finally:
            self.processing = False

    def _on_processing_complete(self):
        """Called when video processing is complete."""
        self.status_lbl.config(text="Status: Processamento concluído")
        self.exec_btn.config(state="normal")
        self.load_btn.config(state="normal")
        
        # Load processed video for playback
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
            
            # Enable playback controls
            self.prev_btn.config(state="normal")
            self.play_btn.config(state="normal")
            self.next_btn.config(state="normal")
            
            # Enable save results button
            self.save_btn.config(state="normal")
            
            # Update metrics plots
            self._update_metrics_plots()

    def save_results(self):
        """Salva gráficos detalhados em src/results/"""
        if not self.filt_pts or not self.meas_pts:
            messagebox.showwarning("Aviso", "Nenhum resultado para salvar. Execute o processamento primeiro.")
            return
        
        try:
            os.makedirs("FiltroKalman/src/results", exist_ok=True)
            
            # Gerar timestamp para nomes únicos
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            video_name = os.path.splitext(os.path.basename(self.video_path))[0] if self.video_path else "results"
            
            # ===== GRÁFICO 1: TRAJETÓRIA DETALHADA =====
            fig1 = Figure(figsize=(14, 10), tight_layout=True, dpi=150)
            ax1 = fig1.add_subplot(111)
            
            xs_meas = [p[0] for p in self.meas_pts if p is not None]
            ys_meas = [p[1] for p in self.meas_pts if p is not None]
            xs_filt = [p[0] for p in self.filt_pts]
            ys_filt = [p[1] for p in self.filt_pts]
            
            ax1.plot(xs_meas, ys_meas, "r.-", label="Medido (Detector)", linewidth=2, markersize=4, alpha=0.7)
            ax1.plot(xs_filt, ys_filt, "#333333", label="Filtrado (Kalman)", linewidth=2.5, alpha=0.9)
            
            ax1.set_xlabel("Posição X (pixels)", fontsize=12, fontweight="bold")
            ax1.set_ylabel("Posição Y (pixels)", fontsize=12, fontweight="bold")
            ax1.set_title("Trajetória: Medida vs Filtrada", fontsize=14, fontweight="bold", pad=20)
            ax1.legend(fontsize=11, loc="best", framealpha=0.95)
            ax1.grid(True, alpha=0.3, linestyle="--")
            ax1.set_facecolor("#f5f5f5")
            
            path1 = f"src/results/{video_name}_traj_{timestamp}.png"
            fig1.savefig(path1, dpi=150, bbox_inches="tight")
            
            # ===== GRÁFICO 2: RMS X =====
            fig2 = Figure(figsize=(14, 7), tight_layout=True, dpi=150)
            ax2 = fig2.add_subplot(111)
            
            sx = np.array([v for v in self.sqerr_x if not (v is None or np.isnan(v))], dtype=float)
            if sx.size > 0:
                running_mean_x = np.cumsum(sx) / np.arange(1, sx.size + 1)
                running_rms_x = np.sqrt(running_mean_x)
                frames_x = range(len(running_rms_x))
                
                ax2.plot(frames_x, running_rms_x, "#333333", linewidth=2.5, label="RMS X")
                ax2.fill_between(frames_x, running_rms_x, alpha=0.2, color="#cccccc")
            
            ax2.set_xlabel("Índice do Frame", fontsize=12, fontweight="bold")
            ax2.set_ylabel("Erro Quadrático Médio (pixels)", fontsize=12, fontweight="bold")
            ax2.set_title("RMS de X: Erro Ao Longo do Tempo", fontsize=14, fontweight="bold", pad=20)
            ax2.legend(fontsize=11, loc="best")
            ax2.grid(True, alpha=0.3, linestyle="--")
            ax2.set_facecolor("#f5f5f5")
            
            path2 = f"src/results/{video_name}_rms_x_{timestamp}.png"
            fig2.savefig(path2, dpi=150, bbox_inches="tight")
            
            # ===== GRÁFICO 3: RMS Y =====
            fig3 = Figure(figsize=(14, 7), tight_layout=True, dpi=150)
            ax3 = fig3.add_subplot(111)
            
            sy = np.array([v for v in self.sqerr_y if not (v is None or np.isnan(v))], dtype=float)
            if sy.size > 0:
                running_mean_y = np.cumsum(sy) / np.arange(1, sy.size + 1)
                running_rms_y = np.sqrt(running_mean_y)
                frames_y = range(len(running_rms_y))
                
                ax3.plot(frames_y, running_rms_y, "#333333", linewidth=2.5, label="RMS Y")
                ax3.fill_between(frames_y, running_rms_y, alpha=0.2, color="#cccccc")
            
            ax3.set_xlabel("Índice do Frame", fontsize=12, fontweight="bold")
            ax3.set_ylabel("Erro Quadrático Médio (pixels)", fontsize=12, fontweight="bold")
            ax3.set_title("RMS de Y: Erro Ao Longo do Tempo", fontsize=14, fontweight="bold", pad=20)
            ax3.legend(fontsize=11, loc="best")
            ax3.grid(True, alpha=0.3, linestyle="--")
            ax3.set_facecolor("#f5f5f5")
            
            path3 = f"src/results/{video_name}_rms_y_{timestamp}.png"
            fig3.savefig(path3, dpi=150, bbox_inches="tight")
            
            # Fechar figuras para liberar memória
            import matplotlib.pyplot as plt
            plt.close(fig1)
            plt.close(fig2)
            plt.close(fig3)
            
            # Mensagem de sucesso
            msg = (f"Resultados salvos em src/results/:\n\n"
                   f"✅ {os.path.basename(path1)}\n"
                   f"✅ {os.path.basename(path2)}\n"
                   f"✅ {os.path.basename(path3)}")
            messagebox.showinfo("Sucesso", msg)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao salvar resultados: {e}")

    def toggle_playback(self):
        """Toggle play/pause."""
        if not self.cap:
            return
        
        self.playing = not self.playing
        if self.playing:
            self.play_btn.config(text="⏸")
        else:
            self.play_btn.config(text="⏵")

    def next_frame(self):
        """Next frame."""
        if not self.cap:
            return
        self.playing = False
        self.play_btn.config(text="⏵")
        
        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self._display_current_frame()

    def prev_frame(self):
        """Previous frame."""
        if not self.cap:
            return
        self.playing = False
        self.play_btn.config(text="⏵")
        
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self._display_current_frame()

    def _display_current_frame(self):
        """Display frame at current_frame_idx."""
        if not self.cap:
            return
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        
        if ret:
            self.tela_viewer.display_image(frame)
            # Update time display
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            current_time = self.current_frame_idx / fps
            total_time = self.total_frames / fps
            current_str = self._format_time(current_time)
            total_str = self._format_time(total_time)
            self.time_info_lbl.config(text=f"{current_str} / {total_str}")
            # Update metrics
            self._update_metrics_plots()

    def _poll_playback(self):
        """Poll for playback updates."""
        if self.playing and self.cap and self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self._display_current_frame()
        elif self.playing and self.current_frame_idx >= self.total_frames - 1:
            self.playing = False
            self.play_btn.config(text="⏵")
        
        self.root.after(33, self._poll_playback)  # ~30 FPS
    
    def _format_time(self, seconds):
        """Convert seconds to MM:SS format."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    def _update_metrics_plots(self):
        """Update all metrics plots synced to current frame."""
        self.metrics_update_counter += 1
        if self.metrics_update_counter % 3 != 0:  # Update every 3 frames for performance
            return
        
        # Get data up to current frame index (synchronized playback)
        current_idx = self.current_frame_idx + 1
        
        # RMS X - only use data up to current frame
        sx = np.array([v for v in self.sqerr_x[:current_idx] if not (v is None or np.isnan(v))], dtype=float)
        if sx.size > 0:
            running_mean_x = np.cumsum(sx) / np.arange(1, sx.size + 1)
            running_rms_x = np.sqrt(running_mean_x)
            self.rmsx_line.set_data(range(len(running_rms_x)), running_rms_x)
            self.rmsx_ax.set_xlim(0, max(10, len(running_rms_x)))
            self.rmsx_ax.set_ylim(0, max(1e-6, running_rms_x.max() * 1.2))
        
        # RMS Y - only use data up to current frame
        sy = np.array([v for v in self.sqerr_y[:current_idx] if not (v is None or np.isnan(v))], dtype=float)
        if sy.size > 0:
            running_mean_y = np.cumsum(sy) / np.arange(1, sy.size + 1)
            running_rms_y = np.sqrt(running_mean_y)
            self.rmsy_line.set_data(range(len(running_rms_y)), running_rms_y)
            self.rmsy_ax.set_xlim(0, max(10, len(running_rms_y)))
            self.rmsy_ax.set_ylim(0, max(1e-6, running_rms_y.max() * 1.2))
        
        # Trajectory - only use data up to current frame
        xs_meas = [p[0] for p in self.meas_pts[:current_idx] if p is not None]
        ys_meas = [p[1] for p in self.meas_pts[:current_idx] if p is not None]
        xs_filt = [p[0] for p in self.filt_pts[:current_idx]]
        ys_filt = [p[1] for p in self.filt_pts[:current_idx]]
        
        if xs_meas and ys_meas:
            self.traj_line_meas.set_data(xs_meas, ys_meas)
        self.traj_line_filt.set_data(xs_filt, ys_filt)
        
        if xs_filt and ys_filt:
            x_max = max(xs_filt) if xs_filt else self.video_width
            y_max = max(ys_filt) if ys_filt else self.video_height
            y_min = min(ys_filt) if ys_filt else 0
            
            self.traj_ax.set_xlim(0, x_max + 10)
            self.traj_ax.set_ylim(max(y_max + 10, self.video_height), min(y_min - 10, 0))
        
        # Draw only every 3 frames
        self.rmsx_canvas.draw_idle()
        self.rmsy_canvas.draw_idle()
        self.traj_canvas.draw_idle()




def run_app():
    root = tk.Tk()
    app = KalmanApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_app()
