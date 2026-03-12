"""
PyQt6 GUI for Multispectral Drone Detection.
Displays RGB, LWIR, UV, and Fused streams in a 4-panel grid.
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                                 QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
                                 QPushButton, QCheckBox, QComboBox, QFileDialog, QFormLayout, QLineEdit, QProgressBar)
    from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
    from PyQt6.QtGui import QImage, QPixmap, QFont
except ImportError:
    print("PyQt6 is not installed. Please install it with: pip install PyQt6")
    sys.exit(1)

import time
import psutil

from ultralytics import YOLO

# Local imports
from infer import load_config, FrameSource, draw_boxes, add_banner, to_bgr, _COLOR
from fusion import filter_rgb, filter_lwir, fuse

class InferenceWorker(QThread):
    frames_ready = pyqtSignal(dict, int, float)
    finished_stream = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.running = False

    def run(self):
        self.running = True
        cfg = self.cfg
        
        try:
            # Load models
            rgb_model  = YOLO(cfg.get("rgb_model")) if cfg.get("rgb_model") else None
            lwir_model = YOLO(cfg.get("lwir_model")) if cfg.get("lwir_model") else None
            uv_model   = YOLO(cfg.get("uv_model")) if cfg.get("uv_model") else None

            rgb_src  = FrameSource(cfg["rgb_source"]) if cfg.get("rgb_source") else None
            lwir_src = FrameSource(cfg["lwir_source"]) if cfg.get("lwir_source") else None
            uv_source_path = cfg.get("uv_source")
            uv_src   = FrameSource(uv_source_path) if uv_source_path else None

            do_fusion   = cfg.get("do_fusion", True)
            method      = cfg.get("fusion_method", "wbf")
            iou_thr     = float(cfg.get("iou_thr",     0.55))
            skip_thr    = float(cfg.get("skip_thr",    0.01))
            conf_thr    = float(cfg.get("conf_thr",    0.25))
            imgsz       = int(cfg.get("imgsz",       960))
            device      = cfg.get("device",      "cpu")
            max_frames  = cfg.get("max_frames",    None)

            model_conf   = float(cfg.get("model_conf", conf_thr))
            infer_kwargs = dict(imgsz=imgsz, device=device, verbose=False, conf=model_conf)

            frame_idx = 0
            
            while self.running:
                if max_frames is not None and frame_idx >= max_frames:
                    break

                ok_r, rgb_frame  = rgb_src.read() if rgb_src else (False, None)
                ok_l, lwir_frame = lwir_src.read() if lwir_src else (False, None)
                
                # If neither RGB nor LWIR is available, finish
                if not ok_r and not ok_l:
                    if rgb_src and lwir_src:
                        break
                    # if only one source is selected and finishes
                    if not rgb_src and not ok_l: break
                    if not lwir_src and not ok_r: break

                ok_u, uv_frame = uv_src.read() if uv_src else (False, None)

                # Inference
                rgb_res  = rgb_model(rgb_frame,   **infer_kwargs)[0] if rgb_model and ok_r else None
                lwir_res = lwir_model(lwir_frame, **infer_kwargs)[0] if lwir_model and ok_l else None
                uv_res   = uv_model(uv_frame,     **infer_kwargs)[0] if uv_model and ok_u else None

                # Filter
                boxes_r, scores_r = filter_rgb(rgb_res) if rgb_res else (np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32))
                boxes_l, scores_l = filter_lwir(lwir_res) if lwir_res else (np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32))
                boxes_u, scores_u = filter_lwir(uv_res) if uv_res else (np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32))

                # Fuse
                if do_fusion:
                    fused_boxes, fused_scores = fuse(
                        boxes_r, scores_r, boxes_l, scores_l,
                        method=method, iou_thr=iou_thr, skip_thr=skip_thr,
                        boxes3=boxes_u if uv_res else None,
                        scores3=scores_u if uv_res else None,
                    )
                    if len(fused_scores):
                        keep = fused_scores >= conf_thr
                        fused_boxes  = fused_boxes[keep]
                        fused_scores = fused_scores[keep]
                else:
                    fused_boxes, fused_scores = np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)

                emitted_frames = {}

                # Render individual streams with boxes
                if ok_r:
                    out_rgb = draw_boxes(to_bgr(rgb_frame), boxes_r, scores_r, label="rgb", color=_COLOR["rgb"])
                    out_rgb = add_banner(out_rgb, f"RGB  f:{frame_idx}", color=_COLOR["rgb"])
                    emitted_frames["rgb"] = out_rgb
                    
                    if do_fusion:
                        out_fused = draw_boxes(to_bgr(rgb_frame.copy()), fused_boxes, fused_scores, label="fused", color=_COLOR["fused"])
                        fused_streams = []
                        if ok_r: fused_streams.append("rgb")
                        if ok_l: fused_streams.append("lwir")
                        if ok_u: fused_streams.append("uv")
                        fusion_label = f"{'·'.join(s.upper() for s in fused_streams)}  {method.upper()}"
                        out_fused = add_banner(out_fused, f"{fusion_label}  f:{frame_idx}", color=_COLOR["fused"])
                        emitted_frames["fused"] = out_fused

                if ok_l:
                    out_lwir = draw_boxes(to_bgr(lwir_frame), boxes_l, scores_l, label="lwir", color=_COLOR["lwir"])
                    out_lwir = add_banner(out_lwir, f"LWIR  f:{frame_idx}", color=_COLOR["lwir"])
                    emitted_frames["lwir"] = out_lwir

                if ok_u:
                    out_uv = draw_boxes(to_bgr(uv_frame), boxes_u, scores_u, label="uv", color=_COLOR["uv"])
                    out_uv = add_banner(out_uv, f"UV  f:{frame_idx}", color=_COLOR["uv"])
                    emitted_frames["uv"] = out_uv

                # FPS Calculation
                if frame_idx == 0:
                    self.start_time = time.time()
                    fps = 0.0
                else:
                    elapsed = time.time() - self.start_time
                    fps = frame_idx / elapsed

                # Emit frames to GUI
                self.frames_ready.emit(emitted_frames, frame_idx, fps)
                frame_idx += 1

                # Yield to let event loop catch up
                self.msleep(10)

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.running = False
            if rgb_src: rgb_src.release()
            if lwir_src: lwir_src.release()
            if uv_src: uv_src.release()
            self.finished_stream.emit()

    def stop(self):
        self.running = False
        self.quit()


class VideoPanel(QLabel):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #0d0d0d; color: #888888; border: 1px solid #2d2d2d; border-radius: 8px;")
        self._title = title
        self.has_frame = False
        self.setText(f"Waiting for Stream: {self._title}")
        
        # Prevent the label from enforcing a size hint based on the pixmap
        from PyQt6.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(100, 100)
    
    def set_title(self, title):
        self._title = title
        if not self.has_frame:
            self.setText(f"Waiting for Stream: {self._title}")

    def set_frame(self, frame_bgr):
        h, w, c = frame_bgr.shape
        bytes_per_line = c * w
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.has_frame = True
        self.setPixmap(scaled_pixmap)
        
    def clear_frame(self):
        self.has_frame = False
        self.setText(f"Waiting for Stream: {self._title}")


class MainWindow(QMainWindow):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.worker = None
        self.setWindowTitle("Multispectral Drone Detection - Live View")
        self.resize(1300, 800)

        self.setup_ui()
        self.populate_config()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # High-Tech Header
        header_label = QLabel("MULTISPECTRAL DRONE DETECTION INTELLIGENCE")
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_label.setStyleSheet("""
            QLabel {
                font-size: 26px;
                font-weight: 900;
                color: #58a6ff;
                letter-spacing: 3px;
                padding: 10px;
                background-color: rgba(31, 111, 235, 0.1);
                border: 1px solid rgba(88, 166, 255, 0.4);
                border-radius: 6px;
                margin-bottom: 10px;
            }
        """)
        main_layout.addWidget(header_label)

        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)

        # Left side: Control Panel
        control_group = QGroupBox("Configuration")
        control_group.setFixedWidth(350)
        control_layout = QVBoxLayout()
        control_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Form layout for file selectors
        form_layout = QFormLayout()
        
        # RGB Model String
        self.rgb_model_input = QLineEdit()
        self.rgb_btn = QPushButton("Browse")
        self.rgb_btn.clicked.connect(lambda: self.browse_file(self.rgb_model_input, "Select RGB Model", "*.pt"))
        rgb_hz = QHBoxLayout()
        rgb_hz.addWidget(self.rgb_model_input)
        rgb_hz.addWidget(self.rgb_btn)
        form_layout.addRow("RGB Model:", rgb_hz)

        # LWIR Model String
        self.lwir_model_input = QLineEdit()
        self.lwir_btn = QPushButton("Browse")
        self.lwir_btn.clicked.connect(lambda: self.browse_file(self.lwir_model_input, "Select LWIR Model", "*.pt"))
        lwir_hz = QHBoxLayout()
        lwir_hz.addWidget(self.lwir_model_input)
        lwir_hz.addWidget(self.lwir_btn)
        form_layout.addRow("LWIR Model:", lwir_hz)

        # UV Model String
        self.uv_model_input = QLineEdit()
        self.uv_btn = QPushButton("Browse")
        self.uv_btn.clicked.connect(lambda: self.browse_file(self.uv_model_input, "Select UV Model", "*.pt"))
        uv_hz = QHBoxLayout()
        uv_hz.addWidget(self.uv_model_input)
        uv_hz.addWidget(self.uv_btn)
        form_layout.addRow("UV Model:", uv_hz)
        
        # Sources
        self.rgb_src_input = QLineEdit()
        form_layout.addRow("RGB Source:", self.rgb_src_input)
        
        self.lwir_src_input = QLineEdit()
        form_layout.addRow("LWIR Source:", self.lwir_src_input)
        
        self.uv_src_input = QLineEdit()
        form_layout.addRow("UV Source:", self.uv_src_input)

        # Fusion Checkbox
        self.fusion_check = QCheckBox("Enable Fusion")
        self.fusion_check.setChecked(True)
        form_layout.addRow("Fusion:", self.fusion_check)

        # Fusion Algorithm Selection
        self.fusion_algo_combo = QComboBox()
        self.fusion_algo_combo.addItems(["wbf", "bayesian"])
        form_layout.addRow("Fusion Algo:", self.fusion_algo_combo)

        control_layout.addLayout(form_layout)
        control_layout.addSpacing(10)

        # Dynamic Grid Configuration
        grid_group = QGroupBox("Grid Layout Selection")
        grid_layout = QFormLayout()
        
        self.tl_combo = QComboBox()
        self.tr_combo = QComboBox()
        self.bl_combo = QComboBox()
        self.br_combo = QComboBox()
        
        opts = ["None", "RGB", "LWIR", "UV", "Fused"]
        for cb in [self.tl_combo, self.tr_combo, self.bl_combo, self.br_combo]:
            cb.addItems(opts)
            cb.currentTextChanged.connect(self._on_grid_combo_changed)
            
        self.tl_combo.setCurrentText("RGB")
        self.tr_combo.setCurrentText("LWIR")
        self.bl_combo.setCurrentText("UV")
        self.br_combo.setCurrentText("Fused")
        
        grid_layout.addRow("Top Left:", self.tl_combo)
        grid_layout.addRow("Top Right:", self.tr_combo)
        grid_layout.addRow("Bottom Left:", self.bl_combo)
        grid_layout.addRow("Bottom Right:", self.br_combo)
        
        grid_group.setLayout(grid_layout)
        control_layout.addWidget(grid_group)

        # Playback Controls
        play_layout = QHBoxLayout()
        self.start_btn = QPushButton("▶ Start")
        self.start_btn.clicked.connect(self.start_sim)
        self.stop_btn = QPushButton("■ Stop")
        self.stop_btn.clicked.connect(self.stop_sim)
        self.stop_btn.setEnabled(False)
        self.reset_btn = QPushButton("↻ Reset")
        self.reset_btn.clicked.connect(self.reset_sim)
        
        play_layout.addWidget(self.start_btn)
        play_layout.addWidget(self.stop_btn)
        play_layout.addWidget(self.reset_btn)

        control_layout.addSpacing(20)
        control_layout.addLayout(play_layout)
        
        # Status Label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: gray;")
        control_layout.addSpacing(20)
        control_layout.addWidget(self.status_label)
        
        control_group.setLayout(control_layout)
        content_layout.addWidget(control_group)

        # Middle: Video Panels
        self.video_layout = QGridLayout()
        self.video_layout.setSpacing(5)
        
        self.panel_tl = VideoPanel(self.tl_combo.currentText())
        self.panel_tr = VideoPanel(self.tr_combo.currentText())
        self.panel_bl = VideoPanel(self.bl_combo.currentText())
        self.panel_br = VideoPanel(self.br_combo.currentText())

        self.video_layout.addWidget(self.panel_tl, 0, 0)
        self.video_layout.addWidget(self.panel_tr, 0, 1)
        self.video_layout.addWidget(self.panel_bl, 1, 0)
        self.video_layout.addWidget(self.panel_br, 1, 1)

        self.video_layout.setColumnStretch(0, 1)
        self.video_layout.setColumnStretch(1, 1)
        self.video_layout.setRowStretch(0, 1)
        self.video_layout.setRowStretch(1, 1)

        content_layout.addLayout(self.video_layout, 1)

        # Right side: Metrics area
        metrics_group = QGroupBox("System Telemetry")
        metrics_group.setFixedWidth(250)
        metrics_layout = QVBoxLayout()
        
        # CPU
        self.cpu_label = QLabel("CPU Usage: 0%")
        self.cpu_bar = QProgressBar()
        self.cpu_bar.setRange(0, 100)
        self.cpu_bar.setTextVisible(False)
        metrics_layout.addWidget(self.cpu_label)
        metrics_layout.addWidget(self.cpu_bar)
        
        # RAM
        self.ram_label = QLabel("Memory Usage: 0%")
        self.ram_bar = QProgressBar()
        self.ram_bar.setRange(0, 100)
        self.ram_bar.setTextVisible(False)
        metrics_layout.addWidget(self.ram_label)
        metrics_layout.addWidget(self.ram_bar)
        
        metrics_layout.addSpacing(20)
        
        # FPS
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #4daafc;")
        metrics_layout.addWidget(self.fps_label)
        
        metrics_layout.addStretch()
        metrics_group.setLayout(metrics_layout)

        content_layout.addWidget(metrics_group)

        # Setup Timer for System Metrics
        self.metrics_timer = QTimer(self)
        self.metrics_timer.timeout.connect(self.update_system_metrics)
        self.metrics_timer.start(1000)

    def _on_grid_combo_changed(self):
        # Update dummy titles when not playing and toggle visibility
        if hasattr(self, "panel_tl"):
            combos = [self.tl_combo, self.tr_combo, self.bl_combo, self.br_combo]
            panels = [self.panel_tl, self.panel_tr, self.panel_bl, self.panel_br]
            
            for combo, panel in zip(combos, panels):
                text = combo.currentText()
                panel.set_title(text)
                
                # Hide the panel completely to give space to others if "None"
                if text == "None":
                    panel.setVisible(False)
                else:
                    panel.setVisible(True)

            # Dynamically adjust grid stretches to reclaim empty space
            row0_visible = self.panel_tl.isVisible() or self.panel_tr.isVisible()
            row1_visible = self.panel_bl.isVisible() or self.panel_br.isVisible()
            col0_visible = self.panel_tl.isVisible() or self.panel_bl.isVisible()
            col1_visible = self.panel_tr.isVisible() or self.panel_br.isVisible()

            self.video_layout.setRowStretch(0, 1 if row0_visible else 0)
            self.video_layout.setRowStretch(1, 1 if row1_visible else 0)
            self.video_layout.setColumnStretch(0, 1 if col0_visible else 0)
            self.video_layout.setColumnStretch(1, 1 if col1_visible else 0)

    def update_system_metrics(self):
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        self.cpu_label.setText(f"CPU Usage: {cpu}%")
        self.cpu_bar.setValue(int(cpu))
        self.ram_label.setText(f"Memory Usage: {ram}%")
        self.ram_bar.setValue(int(ram))

    def populate_config(self):
        # Set values from CLI/Config
        self.rgb_model_input.setText(self.cfg.get("rgb_model", ""))
        self.lwir_model_input.setText(self.cfg.get("lwir_model", ""))
        self.uv_model_input.setText(self.cfg.get("uv_model", ""))
        
        self.rgb_src_input.setText(self.cfg.get("rgb_source", ""))
        self.lwir_src_input.setText(self.cfg.get("lwir_source", ""))
        self.uv_src_input.setText(self.cfg.get("uv_source", ""))
        
        algo = self.cfg.get("fusion_method", "wbf")
        index = self.fusion_algo_combo.findText(algo)
        if index >= 0:
            self.fusion_algo_combo.setCurrentIndex(index)

    def browse_file(self, line_edit, title, filter_str):
        path, _ = QFileDialog.getOpenFileName(self, title, "", filter_str)
        if path:
            line_edit.setText(path)

    def _sync_ui_to_cfg(self):
        self.cfg["rgb_model"] = self.rgb_model_input.text().strip()
        self.cfg["lwir_model"] = self.lwir_model_input.text().strip()
        self.cfg["uv_model"] = self.uv_model_input.text().strip()
        self.cfg["rgb_source"] = self.rgb_src_input.text().strip()
        self.cfg["lwir_source"] = self.lwir_src_input.text().strip()
        self.cfg["uv_source"] = self.uv_src_input.text().strip()
        self.cfg["do_fusion"] = self.fusion_check.isChecked()
        self.cfg["fusion_method"] = self.fusion_algo_combo.currentText()

    def start_sim(self):
        if self.worker is not None and self.worker.isRunning():
            return
        
        self._sync_ui_to_cfg()
        
        self.status_label.setText("Status: Loading models & starting...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.worker = InferenceWorker(self.cfg)
        self.worker.frames_ready.connect(self.update_frames)
        self.worker.finished_stream.connect(self.on_finished)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()

    def stop_sim(self):
        if self.worker is not None:
            self.worker.stop()
            # Do not call .wait() here as it blocks the main thread
            # Let the thread finish naturally
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Status: Stopped")

    def reset_sim(self):
        self.stop_sim()
        self.panel_tl.clear_frame()
        self.panel_tr.clear_frame()
        self.panel_bl.clear_frame()
        self.panel_br.clear_frame()
        self.status_label.setText("Status: Reset and ready.")
        self.fps_label.setText("FPS: --")

    def update_frames(self, frames_dict, frame_idx, fps):
        self.status_label.setText(f"Status: Running... Frame {frame_idx}")
        self.fps_label.setText(f"FPS: {fps:.1f}")
        
        combos = [self.tl_combo, self.tr_combo, self.bl_combo, self.br_combo]
        panels = [self.panel_tl, self.panel_tr, self.panel_bl, self.panel_br]
        
        for combo, panel in zip(combos, panels):
            selected_stream = combo.currentText().lower()
            if selected_stream in frames_dict:
                # Need to respect fusion checkbox
                if selected_stream == "fused" and not self.fusion_check.isChecked():
                    panel.clear_frame()
                else:
                    panel.set_frame(frames_dict[selected_stream])
            else:
                panel.clear_frame()

    def on_finished(self):
        self.status_label.setText("Status: Stream Finished.")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_error(self, err_msg):
        self.status_label.setText(f"Status: Error - {err_msg}")
        self.status_label.setStyleSheet("color: red;")
        self.stop_sim()

    def closeEvent(self, event):
        print("Stopping inference...")
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(1000) # Give it 1 second to clean up before closing entirely
        event.accept()

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GUI for Multi-stream inference (RGB / LWIR / UV) with optional fusion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-c",  "--config",    default="configs/infer.yaml", help="YAML config file")
    p.add_argument("-r",  "--rgb",       metavar="PT",  help="RGB model weights")
    p.add_argument("-l",  "--lwir",      metavar="PT",  help="LWIR model weights")
    p.add_argument("-u",  "--uv",        metavar="PT",  help="UV model weights")
    p.add_argument("-rs", "--rgb-src",   metavar="SRC", help="RGB video source")
    p.add_argument("-ls", "--lwir-src",  metavar="SRC", help="LWIR video source")
    p.add_argument("-us", "--uv-src",    metavar="SRC", help="UV video source")
    p.add_argument("-d",  "--device",    metavar="DEV", help="mps | cpu | 0 (CUDA)")
    p.add_argument("-n",  "--max-frames", metavar="N", type=int, help="frame cap")
    return p

if __name__ == "__main__":
    args = _build_parser().parse_args()
    cfg  = load_config(args.config)

    # CLI overrides config
    if args.rgb:        cfg["rgb_model"]   = args.rgb
    if args.lwir:       cfg["lwir_model"]  = args.lwir
    if args.uv:         cfg["uv_model"]    = args.uv
    if args.rgb_src:    cfg["rgb_source"]  = args.rgb_src
    if args.lwir_src:   cfg["lwir_source"] = args.lwir_src
    if args.uv_src:     cfg["uv_source"]   = args.uv_src
    if args.device:     cfg["device"]      = args.device
    if args.max_frames: cfg["max_frames"]  = args.max_frames

    app = QApplication(sys.argv)
    
    # Modern Dark Theme Stylesheet
    app.setStyleSheet("""
        QMainWindow {
            background-color: #0d1117;
        }
        QWidget {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            font-size: 13px;
            color: #c9d1d9;
        }
        QGroupBox {
            border: 1px solid #30363d;
            border-radius: 8px;
            margin-top: 24px;
            padding-top: 15px;
            background-color: rgba(22, 27, 34, 0.8);
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 5px 15px;
            background-color: #1f6feb;
            color: white;
            border-radius: 4px;
            font-weight: 800;
            font-size: 14px;
            letter-spacing: 1px;
        }
        QPushButton {
            background-color: #21262d;
            color: #c9d1d9;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #30363d;
            border-color: #8b949e;
        }
        QPushButton:pressed {
            background-color: #1f6feb;
            color: white;
            border: 1px solid #1f6feb;
        }
        QPushButton:disabled {
            background-color: #161b22;
            color: #484f58;
            border: 1px solid #21262d;
        }
        QLineEdit, QComboBox {
            background-color: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 6px 10px;
            color: #c9d1d9;
            selection-background-color: #1f6feb;
        }
        QLineEdit:focus, QComboBox:focus {
            border: 1px solid #58a6ff;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border-radius: 4px;
            border: 1px solid #30363d;
            background-color: #0d1117;
        }
        QCheckBox::indicator:checked {
            background-color: #1f6feb;
            border: 1px solid #1f6feb;
        }
        QLabel {
            color: #8b949e;
        }
        QProgressBar {
            background-color: #0d1117;
            border: 1px solid #30363d;
            border-radius: 4px;
            height: 12px;
        }
        QProgressBar::chunk {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #1f6feb, stop:1 #58a6ff);
            border-radius: 3px;
        }
    """)
    
    window = MainWindow(cfg)
    window.show()
    sys.exit(app.exec())
