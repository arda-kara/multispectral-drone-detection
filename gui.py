"""
PyQt6 GUI for Multispectral Drone Detection.
Displays RGB, LWIR, UV, and Fused streams in a 4-panel grid.
"""

import sys
import argparse
import cv2

try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                                 QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
                                 QPushButton, QCheckBox, QComboBox, QFileDialog, QFormLayout, QLineEdit,
                                 QScrollArea, QFrame)
    from PyQt6.QtCore import QThread, pyqtSignal, Qt
    from PyQt6.QtGui import QImage, QPixmap, QFont
except ImportError:
    print("PyQt6 is not installed. Please install it with: pip install PyQt6")
    sys.exit(1)

import time

from ultralytics import YOLO

# Local imports
from infer import (
    load_config,
    FrameSource,
    add_banner,
    draw_boxes,
    draw_fused_boxes,
    empty_detections,
    fuse_selected_streams,
    parse_fusion_streams,
    stream_title,
    to_bgr,
    _COLOR,
)
from fusion import filter_rgb, filter_lwir, load_homography, parse_offset, warp_boxes

# Maps grid combo text (lowercased) to emitted frames_dict keys
_COMBO_KEY = {
    "rgb": "rgb",
    "lwir": "lwir",
    "uv": "uv",
    "fused": "fused",
    "aligned lwir": "aligned_lwir",
    "aligned uv": "aligned_uv",
}
_FUSION_BANDS = ("rgb", "lwir", "uv")


def _format_offset(value) -> str:
    try:
        dx, dy = parse_offset(value)
    except ValueError:
        return "" if value is None else str(value)
    return f"{dx:g},{dy:g}"


def _align_label(H, offset_value) -> str:
    if H is None:
        return "identity"
    dx, dy = parse_offset(offset_value)
    if dx == 0.0 and dy == 0.0:
        return "H-loaded"
    return f"H+off ({dx:g},{dy:g})"

class InferenceWorker(QThread):
    frames_ready = pyqtSignal(dict, int, float, dict)
    finished_stream = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.running = False

    def run(self):
        import traceback as _tb
        self.running = True
        cfg = self.cfg
        rgb_src = lwir_src = uv_src = None  # ensure finally can always clean up

        try:
            # Load models
            rgb_model  = YOLO(cfg.get("rgb_model")) if cfg.get("rgb_model") else None
            lwir_model = YOLO(cfg.get("lwir_model")) if cfg.get("lwir_model") else None
            uv_model   = YOLO(cfg.get("uv_model")) if cfg.get("uv_model") else None

            rgb_src  = FrameSource(cfg["rgb_source"]) if cfg.get("rgb_source") else None
            lwir_src = FrameSource(cfg["lwir_source"]) if cfg.get("lwir_source") else None
            uv_source_path = cfg.get("uv_source")
            uv_src   = FrameSource(uv_source_path) if uv_source_path else None

            H_lwir = load_homography(cfg.get("lwir_homography"), cfg.get("lwir_homography_offset"))
            H_uv   = load_homography(cfg.get("uv_homography"), cfg.get("uv_homography_offset"))

            method      = cfg.get("fusion_method", "wbf")
            iou_thr     = float(cfg.get("iou_thr",     0.55))
            skip_thr    = float(cfg.get("skip_thr",    0.01))
            conf_thr    = float(cfg.get("conf_thr",    0.25))
            center_dist_px = float(cfg.get("center_dist_px", 60.0))
            fusion_margin  = float(cfg.get("fusion_margin", 0.10))
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
                ok_u, uv_frame   = uv_src.read() if uv_src else (False, None)
                if not ok_r and not ok_l and not ok_u:
                    break

                # Inference
                rgb_res  = rgb_model(rgb_frame,   **infer_kwargs)[0] if rgb_model and ok_r else None
                lwir_res = lwir_model(lwir_frame, **infer_kwargs)[0] if lwir_model and ok_l else None
                uv_res   = uv_model(uv_frame,     **infer_kwargs)[0] if uv_model and ok_u else None

                # Filter
                boxes_r, scores_r = filter_rgb(rgb_res) if rgb_res is not None else empty_detections()
                boxes_l_raw, scores_l = filter_lwir(lwir_res) if lwir_res is not None else empty_detections()
                boxes_u_raw, scores_u = filter_lwir(uv_res) if uv_res is not None else empty_detections()

                frames_dict = {}
                boxes_dict = {}
                scores_dict = {}
                if ok_r:
                    frames_dict["rgb"] = rgb_frame
                    boxes_dict["rgb"] = boxes_r
                    scores_dict["rgb"] = scores_r
                if ok_l:
                    frames_dict["lwir"] = lwir_frame
                    boxes_dict["lwir"] = boxes_l_raw
                    scores_dict["lwir"] = scores_l
                if ok_u:
                    frames_dict["uv"] = uv_frame
                    boxes_dict["uv"] = boxes_u_raw
                    scores_dict["uv"] = scores_u

                boxes_for_fusion = {band: boxes.copy() for band, boxes in boxes_dict.items()}

                # Warp LWIR and UV boxes into RGB pixel space before fusion
                if H_lwir is not None and ok_l and ok_r:
                    boxes_for_fusion["lwir"] = warp_boxes(
                        boxes_l_raw,
                        H_lwir,
                        (lwir_frame.shape[1], lwir_frame.shape[0]),
                        (rgb_frame.shape[1],  rgb_frame.shape[0]),
                    )
                if H_uv is not None and ok_u and ok_r:
                    boxes_for_fusion["uv"] = warp_boxes(
                        boxes_u_raw,
                        H_uv,
                        (uv_frame.shape[1], uv_frame.shape[0]),
                        (rgb_frame.shape[1], rgb_frame.shape[0]),
                    )

                do_fusion = bool(cfg.get("do_fusion", True))
                fused_boxes, fused_scores, fused_details, fused_streams, fusion_canvas_key = fuse_selected_streams(
                    frames_dict,
                    boxes_for_fusion,
                    scores_dict,
                    cfg.get("fusion_streams"),
                    do_fusion=do_fusion,
                    method=method,
                    iou_thr=iou_thr,
                    skip_thr=skip_thr,
                    conf_thr=conf_thr,
                    center_dist_px=center_dist_px,
                )

                emitted_frames = {}

                # Render individual streams with boxes
                if ok_r:
                    out_rgb = draw_boxes(to_bgr(rgb_frame), boxes_r, scores_r, label="rgb", color=_COLOR["rgb"])
                    out_rgb = add_banner(out_rgb, f"RGB  f:{frame_idx}", color=_COLOR["rgb"])
                    emitted_frames["rgb"] = out_rgb

                if ok_l:
                    out_lwir = draw_boxes(to_bgr(lwir_frame), boxes_l_raw, scores_l, label="lwir", color=_COLOR["lwir"])
                    out_lwir = add_banner(out_lwir, f"LWIR  f:{frame_idx}", color=_COLOR["lwir"])
                    emitted_frames["lwir"] = out_lwir

                if ok_u:
                    out_uv = draw_boxes(to_bgr(uv_frame), boxes_u_raw, scores_u, label="uv", color=_COLOR["uv"])
                    out_uv = add_banner(out_uv, f"UV  f:{frame_idx}", color=_COLOR["uv"])
                    emitted_frames["uv"] = out_uv

                if do_fusion and fused_streams and fusion_canvas_key in frames_dict:
                    fused_canvas = to_bgr(frames_dict[fusion_canvas_key].copy())
                    out_fused = draw_fused_boxes(
                        fused_canvas,
                        fused_boxes,
                        fused_scores,
                        details=fused_details,
                        margin_percent=fusion_margin,
                    )
                    fusion_label = f"{'·'.join(s.upper() for s in fused_streams)}  {method.upper()}"
                    out_fused = add_banner(
                        out_fused,
                        f"{fusion_label} on {stream_title(fusion_canvas_key)}  f:{frame_idx}",
                        color=_COLOR["fused"],
                    )
                    emitted_frames["fused"] = out_fused

                # Aligned frames: LWIR/UV warped to RGB canvas for visual verification
                if H_lwir is not None and ok_l and ok_r:
                    dsize = (rgb_frame.shape[1], rgb_frame.shape[0])
                    al = cv2.warpPerspective(to_bgr(lwir_frame), H_lwir, dsize,
                                            flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    al = draw_boxes(al, boxes_for_fusion["lwir"], scores_l, label="lwir-aligned", color=_COLOR["lwir"])
                    al = add_banner(al, f"LWIR Aligned  f:{frame_idx}", color=_COLOR["lwir"])
                    emitted_frames["aligned_lwir"] = al

                if H_uv is not None and ok_u and ok_r:
                    dsize = (rgb_frame.shape[1], rgb_frame.shape[0])
                    au = cv2.warpPerspective(to_bgr(uv_frame), H_uv, dsize,
                                            flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    au = draw_boxes(au, boxes_for_fusion["uv"], scores_u, label="uv-aligned", color=_COLOR["uv"])
                    au = add_banner(au, f"UV Aligned  f:{frame_idx}", color=_COLOR["uv"])
                    emitted_frames["aligned_uv"] = au

                # FPS Calculation
                if frame_idx == 0:
                    self.start_time = time.time()
                    fps = 0.0
                else:
                    elapsed = time.time() - self.start_time
                    fps = frame_idx / elapsed

                # Per-model detection metrics
                def _stream_metrics(scores):
                    if len(scores) == 0:
                        return {"count": 0, "conf_max": None, "conf_avg": None}
                    return {
                        "count": len(scores),
                        "conf_max": float(scores.max()),
                        "conf_avg": float(scores.mean()),
                    }

                detection_metrics = {
                    "rgb":   {**_stream_metrics(scores_r),     "align": "identity"},
                    "lwir":  {**_stream_metrics(scores_l),     "align": _align_label(H_lwir, cfg.get("lwir_homography_offset"))},
                    "uv":    {**_stream_metrics(scores_u),     "align": _align_label(H_uv, cfg.get("uv_homography_offset"))},
                    "fused": {**_stream_metrics(fused_scores), "align": "n/a"},
                }

                # Emit frames to GUI
                self.frames_ready.emit(emitted_frames, frame_idx, fps, detection_metrics)
                frame_idx += 1

                # Yield to let event loop catch up
                self.msleep(10)

        except Exception as e:
            _tb.print_exc()  # always visible in the terminal
            self.error_occurred.emit(f"{type(e).__name__}: {e}")
        finally:
            self.running = False
            for _src in (rgb_src, lwir_src, uv_src):
                if _src is not None:
                    try:
                        _src.release()
                    except Exception:
                        pass
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


class CollapsibleBox(QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.toggle_button = QPushButton(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)
        self.toggle_button.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 10px 12px;
                background-color: #21262d;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 6px;
                font-weight: 800;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #30363d;
            }
            QPushButton:checked {
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
                border-bottom: None;
            }
        """)
        
        self.toggle_button.toggled.connect(self.on_pressed)

        self.content_area = QFrame()
        self.content_area.setStyleSheet("""
            QFrame {
                background-color: rgba(22, 27, 34, 0.5);
                border: 1px solid #30363d;
                border-top: None;
                border-bottom-left-radius: 6px;
                border-bottom-right-radius: 6px;
            }
        """)
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(12, 12, 12, 12)
        
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.toggle_button)
        main_layout.addWidget(self.content_area)
        
        # Initialize arrow
        self.update_arrow(True)

    def on_pressed(self, checked):
        self.content_area.setVisible(checked)
        self.update_arrow(checked)

    def update_arrow(self, checked):
        title = self.toggle_button.text().replace("▼ ", "").replace("▶ ", "")
        if checked:
            self.toggle_button.setText(f"▼ {title}")
        else:
            self.toggle_button.setText(f"▶ {title}")

    def addWidget(self, widget):
        self.content_layout.addWidget(widget)
        
    def addLayout(self, layout):
        self.content_layout.addLayout(layout)


class MainWindow(QMainWindow):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.worker = None
        self._had_error = False
        self.setWindowTitle("Multispectral Drone Detection - Live View")
        self.resize(1300, 800)

        self.setup_ui()
        self.populate_config()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # High-Tech Header
        header_label = QLabel("MULTISPECTRAL DRONE DETECTION")
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
        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setFixedWidth(370)
        control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        control_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollArea > QWidget > QWidget {
                background-color: transparent;
            }
        """)
        
        control_widget = QWidget()
        control_scroll.setWidget(control_widget)
        
        control_layout = QVBoxLayout(control_widget)
        control_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        control_layout.setContentsMargins(0, 0, 10, 0)
        control_layout.setSpacing(12)

        # ====== PANEL 1: Models & Sources ======
        panel1 = CollapsibleBox("Models & Sources")
        p1_layout = QFormLayout()
        
        # RGB Model String
        self.rgb_model_input = QLineEdit()
        self.rgb_btn = QPushButton("Browse")
        self.rgb_btn.clicked.connect(lambda: self.browse_file(self.rgb_model_input, "Select RGB Model", "*.pt"))
        rgb_hz = QHBoxLayout()
        rgb_hz.addWidget(self.rgb_model_input)
        rgb_hz.addWidget(self.rgb_btn)
        p1_layout.addRow("RGB Model:", rgb_hz)

        # LWIR Model String
        self.lwir_model_input = QLineEdit()
        self.lwir_btn = QPushButton("Browse")
        self.lwir_btn.clicked.connect(lambda: self.browse_file(self.lwir_model_input, "Select LWIR Model", "*.pt"))
        lwir_hz = QHBoxLayout()
        lwir_hz.addWidget(self.lwir_model_input)
        lwir_hz.addWidget(self.lwir_btn)
        p1_layout.addRow("LWIR Model:", lwir_hz)

        # UV Model String
        self.uv_model_input = QLineEdit()
        self.uv_btn = QPushButton("Browse")
        self.uv_btn.clicked.connect(lambda: self.browse_file(self.uv_model_input, "Select UV Model", "*.pt"))
        uv_hz = QHBoxLayout()
        uv_hz.addWidget(self.uv_model_input)
        uv_hz.addWidget(self.uv_btn)
        p1_layout.addRow("UV Model:", uv_hz)
        
        # Sources
        self.rgb_src_input = QLineEdit()
        p1_layout.addRow("RGB Source:", self.rgb_src_input)
        
        self.lwir_src_input = QLineEdit()
        p1_layout.addRow("LWIR Source:", self.lwir_src_input)
        
        self.uv_src_input = QLineEdit()
        p1_layout.addRow("UV Source:", self.uv_src_input)
        
        panel1.addLayout(p1_layout)
        control_layout.addWidget(panel1)

        # ====== PANEL 2: Homography & Fusion ======
        panel2 = CollapsibleBox("Homography & Fusion")
        # Collapse panel 2 by default to easily see panel 1 and 3 if desired, or keep open based on preference. Let's keep it open initially.
        p2_layout = QFormLayout()

        # LWIR Homography (.npy)
        self.lwir_homo_input = QLineEdit()
        self.lwir_homo_btn = QPushButton("Browse")
        self.lwir_homo_btn.clicked.connect(
            lambda: self.browse_file(self.lwir_homo_input, "Select LWIR Homography", "*.npy")
        )
        lwir_homo_hz = QHBoxLayout()
        lwir_homo_hz.addWidget(self.lwir_homo_input)
        lwir_homo_hz.addWidget(self.lwir_homo_btn)
        p2_layout.addRow("LWIR Homo:", lwir_homo_hz)

        # UV Homography (.npy)
        self.uv_homo_input = QLineEdit()
        self.uv_homo_btn = QPushButton("Browse")
        self.uv_homo_btn.clicked.connect(
            lambda: self.browse_file(self.uv_homo_input, "Select UV Homography", "*.npy")
        )
        uv_homo_hz = QHBoxLayout()
        uv_homo_hz.addWidget(self.uv_homo_input)
        uv_homo_hz.addWidget(self.uv_homo_btn)
        p2_layout.addRow("UV Homo:", uv_homo_hz)

        self.lwir_offset_input = QLineEdit()
        self.lwir_offset_input.setPlaceholderText("dx,dy")
        p2_layout.addRow("LWIR Offset:", self.lwir_offset_input)

        self.uv_offset_input = QLineEdit()
        self.uv_offset_input.setPlaceholderText("dx,dy")
        p2_layout.addRow("UV Offset:", self.uv_offset_input)

        # Fusion Checkbox
        self.fusion_check = QCheckBox("Enable Fusion")
        self.fusion_check.setChecked(True)
        self.fusion_check.toggled.connect(self._sync_live_fusion_settings)
        p2_layout.addRow("Fusion:", self.fusion_check)

        self.fuse_rgb_check = QCheckBox("RGB")
        self.fuse_lwir_check = QCheckBox("LWIR")
        self.fuse_uv_check = QCheckBox("UV")
        for checkbox in (self.fuse_rgb_check, self.fuse_lwir_check, self.fuse_uv_check):
            checkbox.setChecked(True)
            checkbox.toggled.connect(self._sync_live_fusion_settings)
        fuse_streams_hz = QHBoxLayout()
        fuse_streams_hz.addWidget(self.fuse_rgb_check)
        fuse_streams_hz.addWidget(self.fuse_lwir_check)
        fuse_streams_hz.addWidget(self.fuse_uv_check)
        p2_layout.addRow("Fuse Streams:", fuse_streams_hz)

        # Fusion Algorithm Selection
        self.fusion_algo_combo = QComboBox()
        self.fusion_algo_combo.addItems(["spatial"])
        p2_layout.addRow("Fusion Algo:", self.fusion_algo_combo)

        self.center_dist_input = QLineEdit()
        p2_layout.addRow("Center Dist px:", self.center_dist_input)

        self.fusion_margin_input = QLineEdit()
        p2_layout.addRow("Box Margin %:", self.fusion_margin_input)
        
        panel2.addLayout(p2_layout)
        control_layout.addWidget(panel2)

        # ====== PANEL 3: Grid Layout & Cosmetics ======
        panel3 = CollapsibleBox("Grid Layout")
        p3_layout = QFormLayout()
        
        self.tl_combo = QComboBox()
        self.tr_combo = QComboBox()
        self.bl_combo = QComboBox()
        self.br_combo = QComboBox()
        
        opts = ["None", "RGB", "LWIR", "UV", "Fused", "Aligned LWIR", "Aligned UV"]
        for cb in [self.tl_combo, self.tr_combo, self.bl_combo, self.br_combo]:
            cb.addItems(opts)
            cb.currentTextChanged.connect(self._on_grid_combo_changed)
            
        self.tl_combo.setCurrentText("RGB")
        self.tr_combo.setCurrentText("LWIR")
        self.bl_combo.setCurrentText("UV")
        self.br_combo.setCurrentText("Fused")
        
        p3_layout.addRow("Top Left:", self.tl_combo)
        p3_layout.addRow("Top Right:", self.tr_combo)
        p3_layout.addRow("Bottom Left:", self.bl_combo)
        p3_layout.addRow("Bottom Right:", self.br_combo)
        
        panel3.addLayout(p3_layout)
        control_layout.addWidget(panel3)
        panel3.toggle_button.setChecked(False) # Default collapse to save space, user can open it. Or actually keeping it open is fine too. Let's keep it open.
        panel3.on_pressed(True)

        control_layout.addStretch()

        # Playback Controls Main Container (Static at bottom)
        playback_container = QWidget()
        playback_layout = QVBoxLayout(playback_container)
        playback_layout.setContentsMargins(0, 10, 10, 0)

        play_btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("▶ Start")
        self.start_btn.clicked.connect(self.start_sim)
        self.start_btn.setStyleSheet("background-color: #238636; color: white; font-weight: bold; border-radius: 6px; padding: 8px;")
        
        self.stop_btn = QPushButton("■ Stop")
        self.stop_btn.clicked.connect(self.stop_sim)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("background-color: #da3633; color: white; font-weight: bold; border-radius: 6px; padding: 8px;")
        
        self.reset_btn = QPushButton("↻ Reset")
        self.reset_btn.clicked.connect(self.reset_sim)
        self.reset_btn.setStyleSheet("background-color: #21262d; border: 1px solid #30363d; border-radius: 6px; padding: 8px;")
        
        play_btn_layout.addWidget(self.start_btn)
        play_btn_layout.addWidget(self.stop_btn)
        play_btn_layout.addWidget(self.reset_btn)

        playback_layout.addLayout(play_btn_layout)
        
        # Status Label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #8b949e; margin-top: 5px;")
        playback_layout.addWidget(self.status_label)
        
        # Combine scroll area and static playback block
        left_main_layout = QVBoxLayout()
        left_main_layout.addWidget(control_scroll, 1) # takes flexible space
        left_main_layout.addWidget(playback_container, 0) # static size
        
        content_layout.addLayout(left_main_layout)

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

        # Right side: Detection Metrics
        metrics_group = QGroupBox("Detection Metrics")
        metrics_group.setFixedWidth(250)
        metrics_layout = QVBoxLayout()
        metrics_layout.setSpacing(4)

        # FPS at the top
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #4daafc;")
        metrics_layout.addWidget(self.fps_label)
        metrics_layout.addSpacing(12)

        # Per-stream metric widgets: (stream_key, display_name, accent_color)
        stream_defs = [
            ("rgb",   "RGB",   "#4daafc"),
            ("lwir",  "LWIR",  "#f0883e"),
            ("uv",    "UV",    "#bc8cff"),
            ("fused", "FUSED", "#3fb950"),
        ]
        self._stream_metric_labels = {}
        for key, name, color in stream_defs:
            title = QLabel(name)
            title.setStyleSheet(f"font-size: 13px; font-weight: bold; color: {color}; margin-top: 6px;")
            metrics_layout.addWidget(title)

            det_lbl  = QLabel("  Detections: --")
            max_lbl  = QLabel("  Max Conf:   --")
            avg_lbl  = QLabel("  Avg Conf:   --")
            for lbl in (det_lbl, max_lbl, avg_lbl):
                lbl.setStyleSheet("font-size: 12px; color: #8b949e;")
                metrics_layout.addWidget(lbl)

            homo_lbl = QLabel("  Align: identity")
            homo_lbl.setStyleSheet("font-size: 11px; color: #484f58;")
            metrics_layout.addWidget(homo_lbl)

            self._stream_metric_labels[key] = (det_lbl, max_lbl, avg_lbl, homo_lbl)

        metrics_layout.addStretch()
        metrics_group.setLayout(metrics_layout)

        content_layout.addWidget(metrics_group)

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

    def populate_config(self):
        # Set values from CLI/Config
        self.rgb_model_input.setText(self.cfg.get("rgb_model", ""))
        self.lwir_model_input.setText(self.cfg.get("lwir_model", ""))
        self.uv_model_input.setText(self.cfg.get("uv_model", ""))
        
        self.rgb_src_input.setText(self.cfg.get("rgb_source", ""))
        self.lwir_src_input.setText(self.cfg.get("lwir_source", ""))
        self.uv_src_input.setText(self.cfg.get("uv_source", ""))
        self.lwir_homo_input.setText(self.cfg.get("lwir_homography") or "")
        self.uv_homo_input.setText(self.cfg.get("uv_homography") or "")
        self.lwir_offset_input.setText(_format_offset(self.cfg.get("lwir_homography_offset")))
        self.uv_offset_input.setText(_format_offset(self.cfg.get("uv_homography_offset")))
        self.center_dist_input.setText(str(self.cfg.get("center_dist_px", 60.0)))
        self.fusion_margin_input.setText(str(self.cfg.get("fusion_margin", 0.10)))
        self.fusion_check.setChecked(bool(self.cfg.get("do_fusion", True)))
        selected_fusion_streams = parse_fusion_streams(self.cfg.get("fusion_streams"))
        if selected_fusion_streams is None:
            selected_fusion_streams = list(_FUSION_BANDS)
        self.fuse_rgb_check.setChecked("rgb" in selected_fusion_streams)
        self.fuse_lwir_check.setChecked("lwir" in selected_fusion_streams)
        self.fuse_uv_check.setChecked("uv" in selected_fusion_streams)

        algo = self.cfg.get("fusion_method", "wbf")
        index = self.fusion_algo_combo.findText(algo)
        if index >= 0:
            self.fusion_algo_combo.setCurrentIndex(index)

    def browse_file(self, line_edit, title, filter_str):
        path, _ = QFileDialog.getOpenFileName(self, title, "", filter_str)
        if path:
            line_edit.setText(path)

    def _selected_fusion_streams_from_ui(self) -> list[str]:
        selected = []
        if self.fuse_rgb_check.isChecked():
            selected.append("rgb")
        if self.fuse_lwir_check.isChecked():
            selected.append("lwir")
        if self.fuse_uv_check.isChecked():
            selected.append("uv")
        return selected

    def _sync_live_fusion_settings(self):
        self.cfg["do_fusion"] = self.fusion_check.isChecked()
        self.cfg["fusion_streams"] = self._selected_fusion_streams_from_ui()

    def _sync_ui_to_cfg(self):
        self.cfg["rgb_model"] = self.rgb_model_input.text().strip()
        self.cfg["lwir_model"] = self.lwir_model_input.text().strip()
        self.cfg["uv_model"] = self.uv_model_input.text().strip()
        self.cfg["rgb_source"] = self.rgb_src_input.text().strip()
        self.cfg["lwir_source"] = self.lwir_src_input.text().strip()
        self.cfg["uv_source"] = self.uv_src_input.text().strip()
        self.cfg["do_fusion"] = self.fusion_check.isChecked()
        self.cfg["fusion_streams"] = self._selected_fusion_streams_from_ui()
        self.cfg["fusion_method"] = self.fusion_algo_combo.currentText()
        self.cfg["lwir_homography"] = self.lwir_homo_input.text().strip() or None
        self.cfg["uv_homography"]   = self.uv_homo_input.text().strip() or None
        self.cfg["lwir_homography_offset"] = self.lwir_offset_input.text().strip() or "0,0"
        self.cfg["uv_homography_offset"]   = self.uv_offset_input.text().strip() or "0,0"
        self.cfg["center_dist_px"] = self.center_dist_input.text().strip() or "60.0"
        self.cfg["fusion_margin"]  = self.fusion_margin_input.text().strip() or "0.10"

    def start_sim(self):
        if self.worker is not None and self.worker.isRunning():
            return

        self._had_error = False
        self.status_label.setStyleSheet("color: gray;")
        self._sync_ui_to_cfg()

        self.status_label.setText("Status: Loading models & starting...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.worker = InferenceWorker(self.cfg)
        self.worker.frames_ready.connect(self._on_frames_ready)
        self.worker.finished_stream.connect(self.on_finished)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()

    def stop_sim(self):
        if self.worker is not None:
            self.worker.stop()

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if not self._had_error:
            self.status_label.setText("Status: Stopped")

    def reset_sim(self):
        self._had_error = False
        self.status_label.setStyleSheet("color: gray;")
        self.stop_sim()
        self.panel_tl.clear_frame()
        self.panel_tr.clear_frame()
        self.panel_bl.clear_frame()
        self.panel_br.clear_frame()
        self.status_label.setText("Status: Reset and ready.")
        self.fps_label.setText("FPS: --")

    def _on_frames_ready(self, frames_dict, frame_idx, fps, detection_metrics):
        self.status_label.setText(f"Status: Running... Frame {frame_idx}")
        self.fps_label.setText(f"FPS: {fps:.1f}")

        combos = [self.tl_combo, self.tr_combo, self.bl_combo, self.br_combo]
        panels = [self.panel_tl, self.panel_tr, self.panel_bl, self.panel_br]

        for combo, panel in zip(combos, panels):
            stream_key = _COMBO_KEY.get(combo.currentText().lower(), combo.currentText().lower())
            if stream_key in frames_dict:
                if stream_key == "fused" and not self.fusion_check.isChecked():
                    panel.clear_frame()
                else:
                    panel.set_frame(frames_dict[stream_key])
            else:
                panel.clear_frame()

        # Refresh per-stream detection metric labels
        for key, (det_lbl, max_lbl, avg_lbl, homo_lbl) in self._stream_metric_labels.items():
            m = detection_metrics.get(key, {})
            count = m.get("count", 0)
            conf_max = m.get("conf_max")
            conf_avg = m.get("conf_avg")
            align = m.get("align", "identity")
            det_lbl.setText(f"  Detections: {count}")
            max_lbl.setText(f"  Max Conf:   {conf_max:.2f}" if conf_max is not None else "  Max Conf:   --")
            avg_lbl.setText(f"  Avg Conf:   {conf_avg:.2f}" if conf_avg is not None else "  Avg Conf:   --")
            homo_lbl.setText(f"  Align: {align}")
            homo_lbl.setStyleSheet(
                "font-size: 11px; color: #3fb950;" if align not in {"identity", "n/a"}
                else "font-size: 11px; color: #484f58;"
            )

    def on_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if not self._had_error:
            self.status_label.setText("Status: Stream Finished.")

    def on_error(self, err_msg):
        self._had_error = True
        self.status_label.setStyleSheet("color: red;")
        self.status_label.setText(f"Status: Error — {err_msg}")
        # Update buttons directly; do not call stop_sim() to avoid overwriting the message
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if self.worker is not None:
            self.worker.stop()

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            print("Stopping inference...")
            self.worker.stop()
            self.worker.wait(1000)  # Give it 1 second to clean up before closing entirely
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
    if args.max_frames is not None:
        cfg["max_frames"] = args.max_frames

    app = QApplication(sys.argv)
    app.setFont(QFont("Helvetica Neue"))
    
    # Modern Dark Theme Stylesheet
    app.setStyleSheet("""
        QMainWindow {
            background-color: #0d1117;
        }
        QWidget {
            font-family: "Helvetica Neue", Helvetica, Arial;
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
