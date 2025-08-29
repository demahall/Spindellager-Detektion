
import tkinter as tk
from tkinter import ttk,filedialog
import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from math import sqrt

# --- your modules ---
from RobustContrastNormalization import robust_contrast_normalization
from FittingEdge import fit_cage_edges, overlay_fits
from GUIHelper import ( fit_circle_from_points, ring_xy, build_annulus_mask,
    robust_normalize, canny_roi, fit_and_overlay)
from Evaluation import (evaluate_fit_quality,
    build_session_dict, save_session_json)


class GUI:
    def __init__(self, image):
        # ----- state -----
        self.image = image.copy()
        self.gray = (self.image if self.image.ndim == 2
                     else cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY))
        self.h,self.w = self.gray.shape
        # Initial step
        self.mode = 'outer_points'

        # State step 1
        self.outer_points = []
        self.center = None
        self.radius = None

        # State step 2
        self.ref_point = None
        self.ring_mask = None
        self.ring_inner = None
        self.ring_outer = None
        self.thresh_diam= None

        # State step 3
        self.robust_parameter = [0.5, 0.5]  # [low, high]
        self.image_norm_full = None

        # State step 4
        self.edges = None
        self.fit_results = None
        self.scatter = None
        self.circle_line = None
        self.ref_scatter = None

        # ----- Tk roots -----
        self.root = tk.Tk()
        self.root.title("Bearing Cage — Image & Log")

        # controls in separate window
        self.ctrl = tk.Toplevel(self.root)
        self.ctrl.title("Parameters & Steps")

        # ----- MAIN WINDOW (image + log) -----
        self.fig, self.ax = plt.subplots(figsize=(6.2, 6.2))
        self.ax.set_title("Step 1: click EXACTLY 5 points on outer ring → Confirm")
        self.im = self.ax.imshow(self.gray, cmap="gray", interpolation="nearest")
        self.ax.set_axis_on()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        self.ring_inner_line, = self.ax.plot([], [], lw=1)  # inner ring
        self.ring_outer_line, = self.ax.plot([], [], lw=1)  # outer ring

        # --- Zooming & panning state ---
        self._panning = False
        self._space_down = False
        self._pan_last_data = None  # (xdata, ydata) of previous motion during pan

        # Matplotlib event bindings
        self.canvas.mpl_connect("scroll_event", self._on_zoom)
        self.canvas.mpl_connect("button_press_event", self._start_pan)
        self.canvas.mpl_connect("button_release_event", self._end_pan)
        self.canvas.mpl_connect("motion_notify_event", self._do_pan)

        # Keyboard to toggle panning with space (bind on main window)
        self.root.bind("<space>", lambda e: setattr(self, "_space_down", True))
        self.root.bind("<KeyRelease-space>", lambda e: setattr(self, "_space_down", False))

        # log
        self.logbox = tk.Text(self.root, height=8, bg="#1f1f1f", fg="#eaeaea")
        self.logbox.pack(fill="both", expand=False, padx=6, pady=6)
        self.log("GUI started")

        # mpl events
        self.cid_click = self.canvas.mpl_connect("button_press_event", self.on_click)

        # ----- CONTROL WINDOW (two panes) -----
        wrap = ttk.Notebook(self.ctrl)
        wrap.pack(fill="both", expand=True)

        # each step as a tab

        self.nb = ttk.Notebook(self.ctrl)
        self.nb.pack(fill="both", expand=True)

        self.tab1 = tk.Frame(self.nb);
        self.nb.add(self.tab1, text="Step 1 — Outer Points")
        self.tab2 = tk.Frame(self.nb);
        self.nb.add(self.tab2, text="Step 2 — Reference & Δdiam")
        self.tab3 = tk.Frame(self.nb);
        self.nb.add(self.tab3, text="Step 3 — Robust Norm")
        self.tab4 = tk.Frame(self.nb);
        self.nb.add(self.tab4, text="Step 4 — Canny")
        self.tab5 = tk.Frame(self.nb);
        self.nb.add(self.tab5, text="Step 5 — Fit & Show")

        # --- Step 1 ---
        tk.Label(self.tab1, text="Click 5 points on the outer ring in the image window.").pack(anchor="w", padx=8,pady=(10, 2))
        row = tk.Frame(self.tab1);
        row.pack(anchor="w", padx=8, pady=6)
        self.btn_confirm5 = tk.Button(row, text="Confirm 5 points", command=self.on_confirm5)
        self.btn_reset_pts = tk.Button(row, text="Reset (Step 1)", command=self.on_reset_points)
        self.btn_next_1 = tk.Button(row, text="Next ▶", command=self.on_next_step1)
        self.btn_confirm5.pack(side="left", padx=4)
        self.btn_reset_pts.pack(side="left", padx=4)
        self.btn_next_1.pack(side="left", padx=8)

        # --- Step 2 ---
        tk.Label(self.tab2, text="Click ONE reference point, set Δdiam (px), then Confirm reference.").pack(anchor="w",padx=8,pady=(10,2))
        row = tk.Frame(self.tab2);
        row.pack(anchor="w", padx=8, pady=6)
        tk.Label(row, text="Δdiam (px):").pack(side="left")
        self.thr_var = tk.DoubleVar(value=20.0)
        self.thr_input = tk.Spinbox(row, from_=1, to=9999, width=6, textvariable=self.thr_var)
        self.thr_input.pack(side="left", padx=(4, 10))
        self.btn_confirm_ref = tk.Button(row, text="Confirm reference", command=self.on_confirm_reference)
        self.btn_reset_step2 = tk.Button(row, text="Reset (Step 2)", command=self.on_reset_step2)
        self.btn_next_2 = tk.Button(row, text="Next ▶", command=self.on_next_step2)
        self.btn_confirm_ref.pack(side="left", padx=4)
        self.btn_reset_step2.pack(side="left", padx=8)
        self.btn_next_2.pack(side="left", padx=8)

        # --- Step 3 (Robust) ---
        tk.Label(self.tab3, text="Robust normalization for FULL image (stretchlim).").pack(anchor="w", padx=8,
                                                                                           pady=(10, 2))
        row = tk.Frame(self.tab3);
        row.pack(anchor="w", padx=8, pady=6)
        tk.Label(row, text="Low %").pack(side="left")
        self.cut_low_var = tk.DoubleVar(value=0.5)
        self.cut_low = tk.Spinbox(row, from_=0.0, to=20.0, increment=0.1, width=6, textvariable=self.cut_low_var)
        self.cut_low.pack(side="left", padx=(4, 12))
        tk.Label(row, text="High %").pack(side="left")
        self.cut_high_var = tk.DoubleVar(value=0.5)
        self.cut_high = tk.Spinbox(row, from_=0.0, to=20.0, increment=0.1, width=6, textvariable=self.cut_high_var)
        self.cut_high.pack(side="left", padx=(4, 12))
        self.btn_confirm_norm = tk.Button(row, text="Confirm (apply robust)", command=self.on_apply_robust)
        self.btn_reset_step3 = tk.Button(row, text="Reset (Step 3)", command=self.on_reset_step3)
        self.btn_next_3 = tk.Button(row, text="Next ▶", command=self.on_next_step3)
        self.btn_confirm_norm.pack(side="left", padx=6)
        self.btn_reset_step3.pack(side="left", padx=8)
        self.btn_next_3.pack(side="left", padx=8)

        # --- Step 4 (Canny) ---
        tk.Label(self.tab4, text="Configure Canny, then Confirm.").pack(anchor="w", padx=8, pady=(10, 2))
        row = tk.Frame(self.tab4);
        row.pack(anchor="w", padx=8, pady=6)
        tk.Label(row, text="low").pack(side="left")
        self.s_low = tk.Scale(row, from_=0, to=255, orient="horizontal", length=250);
        self.s_low.set(60)
        self.s_low.pack(side="left", padx=(4, 12))
        tk.Label(row, text="high").pack(side="left")
        self.s_high = tk.Scale(row, from_=0, to=255, orient="horizontal", length=250);
        self.s_high.set(140)
        self.s_high.pack(side="left", padx=(4, 12))
        self.btn_confirm_canny = tk.Button(row, text="Confirm (apply Canny)", command=self.on_apply_canny)
        self.btn_reset_step4 = tk.Button(row, text="Reset (Step 4)", command=self.on_reset_step4)
        self.btn_next_4 = tk.Button(row, text="Next ▶", command=self.on_next_step4)
        self.btn_confirm_canny.pack(side="left", padx=6)
        self.btn_reset_step4.pack(side="left", padx=8)
        self.btn_next_4.pack(side="left", padx=8)

        # --- Step 5 (Fit) ---
        tk.Label(self.tab5, text="Confirm parameters (quality tol), Show results, or Reset ALL.").pack(anchor="w",
                                                                                                       padx=8,
                                                                                                       pady=(10, 2))
        row = tk.Frame(self.tab5);
        row.pack(anchor="w", padx=8, pady=6)
        tk.Label(row, text="Quality tol (px):").pack(side="left")
        self.tol_px_var = tk.DoubleVar(value=3.0)
        tk.Spinbox(row, from_=0.5, to=10.0, increment=0.5, width=6, textvariable=self.tol_px_var).pack(side="left",
                                                                                                       padx=(4, 12))
        self.btn_confirm_step5 = tk.Button(row, text="Confirm (Step 5)", command=self.on_confirm_step5)
        self.btn_fit_show = tk.Button(row, text="Show results", command=self.on_show_results)
        self.btn_reset_all2 = tk.Button(row, text="Reset ALL", command=self.on_reset_all)
        self.btn_save = tk.Button(row, text="Save session (JSON)", command=self.on_save_json)
        self.btn_confirm_step5.pack(side="left", padx=4)
        self.btn_fit_show.pack(side="left", padx=8)
        self.btn_reset_all2.pack(side="left", padx=8)
        self.btn_save.pack(side="left", padx=8)

    # ================== UI helpers ==================
    def log(self, msg):
        self.logbox.insert("end", msg + "\n"); self.logbox.see("end")

    def _draw_circle(self, xc, yc, r):
        ang = np.linspace(0, 2*np.pi, 400)
        xs, ys = xc + r*np.cos(ang), yc + r*np.sin(ang)
        if self.circle_line is None:
            self.circle_line, = self.ax.plot(xs, ys, 'c-', lw=2)
        else:
            self.circle_line.set_data(xs, ys)
        self.fig.canvas.draw_idle()

    def _set_ring_line(self, line_artist, xc, yc, r):
        xs, ys = ring_xy((xc, yc), r)
        line_artist.set_data(xs, ys)
        self.fig.canvas.draw_idle()

    def _switch_to_tab(self, idx: int):
        self.nb.select(idx)

    def _update_scatter(self):
        pts = np.array(self.outer_points, dtype=float) if self.outer_points else np.empty((0,2))
        if self.scatter is None:
            self.scatter = self.ax.scatter(pts[:,0] if len(pts) else [], pts[:,1] if len(pts) else [], c='y', s=30)
        else:
            self.scatter.set_offsets(pts if len(pts) else np.empty((0,2)))
        self.fig.canvas.draw_idle()

    def _ring_mask_from_radii(self, xc, yc, r_inner, r_outer):
        Y, X = np.ogrid[:self.h, :self.w]
        R2 = (X - xc)**2 + (Y - yc)**2
        mask = (R2 >= (r_inner**2)) & (R2 <= (r_outer**2))
        return mask

    def _update_ref_point(self):
        if self.ref_point is None:
            return
        arr = np.array([self.ref_point], dtype=float)
        if self.ref_scatter is None:
            self.ref_scatter = self.ax.scatter(arr[:,0], arr[:,1], c='r', s=40, marker='x')
        else:
            self.ref_scatter.set_offsets(arr)
        self.fig.canvas.draw_idle()

    def _gate_controls(self):
        # Enable/disable controls per step progression
        s1_done = (len(self.outer_points) == 5 and self.center is not None)
        s2_done = (self.ref_point is not None and self.ring_mask is not None)
        s3_done = (self.image_norm_full is not None)
        s4_done = (self.edges is not None)

        self.btn_confirm5.config(state="normal")
        self.btn_reset_pts.config(state="normal")
        self.btn_reset_all.config(state="normal")

        self.btn_confirm_ref.config(state=("normal" if s1_done else "disabled"))
        self.thr_input.config(state=("normal" if s1_done else "disabled"))

        self.cut_low.config(state=("normal" if s2_done else "disabled"))
        self.cut_high.config(state=("normal" if s2_done else "disabled"))
        self.btn_apply_robust.config(state=("normal" if s2_done else "disabled"))

        self.s_low.config(state=("normal" if s3_done else "disabled"))
        self.s_high.config(state=("normal" if s3_done else "disabled"))
        self.btn_apply_canny.config(state=("normal" if s3_done else "disabled"))

        self.btn_fit_show.config(state=("normal" if s4_done else "disabled"))

        # Titles
        if not s1_done:
            self.ax.set_title("Step 1: click EXACTLY 5 points on outer ring → Confirm")
        elif not s2_done:
            self.ax.set_title("Step 2: click ONE reference point, set Δdiam → Confirm reference")
        elif not s3_done:
            self.ax.set_title("Step 3: set cutoff% and Apply robust normalization")
        elif not s4_done:
            self.ax.set_title("Step 4: set Canny thresholds and Apply")
        else:
            self.ax.set_title("Step 5: Fit done — results shown")
        self.fig.canvas.draw_idle()


    def _reset_view(self):
        """Reset axes to show the full image."""
        h, w = self.gray.shape[:2]
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)  # image coords: origin at top-left by default
        self.fig.canvas.draw_idle()

    def _on_zoom(self, event):
        """Zoom at the mouse position with scroll wheel / trackpad."""
        if event.inaxes != self.ax:
            return
        # Zoom factor
        base_scale = 1.2
        scale = 1 / base_scale if event.step > 0 else base_scale

        cur_xmin, cur_xmax = self.ax.get_xlim()
        cur_ymin, cur_ymax = self.ax.get_ylim()
        xdata = event.xdata if event.xdata is not None else (cur_xmin + cur_xmax) * 0.5
        ydata = event.ydata if event.ydata is not None else (cur_ymin + cur_ymax) * 0.5

        new_xmin = xdata - (xdata - cur_xmin) * scale
        new_xmax = xdata + (cur_xmax - xdata) * scale
        new_ymin = ydata - (ydata - cur_ymin) * scale
        new_ymax = ydata + (cur_ymax - ydata) * scale

        self.ax.set_xlim(new_xmin, new_xmax)
        self.ax.set_ylim(new_ymin, new_ymax)
        self.fig.canvas.draw_idle()

    def _start_pan(self, event):
        """Start panning when SPACE is held and left button pressed."""
        if event.inaxes != self.ax or event.button != 1:
            return
        if not self._space_down:
            return
        # Enter panning mode
        self._panning = True
        self._pan_last_data = (event.xdata, event.ydata)

    def _do_pan(self, event):
        """While panning, shift the view following the mouse."""
        if not self._panning:
            return
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None or self._pan_last_data is None:
            return

        xprev, yprev = self._pan_last_data
        dx = xprev - event.xdata
        dy = yprev - event.ydata

        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        self.ax.set_xlim(xmin + dx, xmax + dx)
        self.ax.set_ylim(ymin + dy, ymax + dy)

        self._pan_last_data = (event.xdata, event.ydata)
        self.fig.canvas.draw_idle()

    def _end_pan(self, event):
        """Stop panning on mouse release."""
        if event.button != 1:
            return
        self._panning = False
        self._pan_last_data = None


    # ================== EVENTS / ACTIONS ==================
    def on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        if self._space_down and event.button == 1:
            return

        x, y = float(event.xdata), float(event.ydata)
        if self.mode == 'outer_points':
            if len(self.outer_points) >= 5:
                self.log("Already have 5 points — use Reset points to change.")
                return
            self.outer_points.append((x, y))
            self._update_scatter()

        elif self.mode == 'reference':
            self.ref_point = (x, y)
            self._update_ref_point()
            self.log(f"Reference selected: ({x:.1f}, {y:.1f})")


    # ---- Step 1 ----
    def on_next_step1(self):
        if self.center is None:
            self.on_confirm5()
            if self.center is None: return
        self.mode = 'reference'
        self._switch_to_tab(1)
        self._gate_controls()

    # ---- Step 2 ----
    def on_reset_step2(self):
        self.ref_point = None
        self.ring_mask = None
        self.ring_inner = None
        self.ring_outer = None
        self.thresh_diam = None
        if self.ref_scatter is not None:
            self.ref_scatter.set_offsets(np.empty((0, 2)))
        self.ring_inner_line.set_data([], [])
        self.ring_outer_line.set_data([], [])
        self.im.set_data(self.image_norm_full if self.image_norm_full is not None else self.gray)
        self.fig.canvas.draw_idle()
        self.mode = 'reference'
        self.log("↩︎ Step 2 reset.")
        self._gate_controls()

    def on_next_step2(self):
        if self.ring_mask is None:
            self.on_confirm_reference()
            if self.ring_mask is None: return
        self.mode = 'robust_normalize'
        self._switch_to_tab(2)
        self._gate_controls()

    # ---- Step 3 ----
    def on_reset_step3(self):
        self.image_norm_full = None
        self.im.set_data(self.gray);
        self.fig.canvas.draw_idle()
        self.mode = 'robust_normalize'
        self.log("↩︎ Step 3 reset.")
        self._gate_controls()

    def on_next_step3(self):
        if self.image_norm_full is None:
            self.on_apply_robust()
            if self.image_norm_full is None: return
        self.mode = 'edges_detect'
        self._switch_to_tab(3)
        self._gate_controls()

    # ---- Step 4 ----
    def on_reset_step4(self):
        self.edges = None
        base = self.image_norm_full if self.image_norm_full is not None else self.gray
        self.im.set_data(base);
        self.fig.canvas.draw_idle()
        self.mode = 'edges_detect'
        self.log("↩︎ Step 4 reset.")
        self._gate_controls()

    def on_next_step4(self):
        if self.edges is None:
            self.on_apply_canny()
            if self.edges is None: return
        self._switch_to_tab(4)
        self._gate_controls()

    # ---- Step 5 ----
    def on_confirm_step5(self):
        # nothing heavy to do; we just acknowledge tol param
        tol = float(self.tol_px_var.get())
        self.log(f"Step 5 params confirmed (quality tol={tol:.1f}px).")

    def _gate_controls(self):
        s1_done = (len(self.outer_points) == 5 and self.center is not None)
        s2_done = (self.ref_point is not None and self.ring_mask is not None)
        s3_done = (self.image_norm_full is not None)
        s4_done = (self.edges is not None)

        # Step 1
        self.btn_confirm5.config(state="normal")
        self.btn_reset_pts.config(state="normal")
        self.btn_next_1.config(state=("normal" if s1_done else "disabled"))

        # Step 2
        self.btn_confirm_ref.config(state=("normal" if s1_done else "disabled"))
        self.thr_input.config(state=("normal" if s1_done else "disabled"))
        self.btn_reset_step2.config(state=("normal" if s1_done else "disabled"))
        self.btn_next_2.config(state=("normal" if s2_done else "disabled"))

        # Step 3
        self.cut_low.config(state=("normal" if s2_done else "disabled"))
        self.cut_high.config(state=("normal" if s2_done else "disabled"))
        self.btn_confirm_norm.config(state=("normal" if s2_done else "disabled"))
        self.btn_reset_step3.config(state=("normal" if s2_done else "disabled"))
        self.btn_next_3.config(state=("normal" if s3_done else "disabled"))

        # Step 4
        self.s_low.config(state=("normal" if s3_done else "disabled"))
        self.s_high.config(state=("normal" if s3_done else "disabled"))
        self.btn_confirm_canny.config(state=("normal" if s3_done else "disabled"))
        self.btn_reset_step4.config(state=("normal" if s3_done else "disabled"))
        self.btn_next_4.config(state=("normal" if s4_done else "disabled"))

        # Step 5: always allow confirm + show; save/reset independent
        self.btn_confirm_step5.config(state=("normal" if s4_done else "disabled"))
        self.btn_fit_show.config(state=("normal" if s4_done else "disabled"))
        self.btn_reset_all2.config(state="normal")
        self.btn_save.config(state=("normal" if s4_done else "disabled"))

        # Titles (yours kept)
        if not s1_done:
            self.ax.set_title("Step 1: click EXACTLY 5 points on outer ring → Confirm")
        elif not s2_done:
            self.ax.set_title("Step 2: click ONE reference point, set Δdiam → Confirm reference")
        elif not s3_done:
            self.ax.set_title("Step 3: set cutoff% and Apply robust normalization")
        elif not s4_done:
            self.ax.set_title("Step 4: set Canny thresholds and Apply")
        else:
            self.ax.set_title("Step 5: Fit done — results shown")
        self.fig.canvas.draw_idle()

    def on_confirm5(self):
        if len(self.outer_points) != 5:
            self.log("❌ You must pick EXACTLY 5 points before confirming.")
            return
        try:
            xc, yc, r = fit_circle_from_points(self.outer_points)
        except Exception as e:
            self.log(f"❌ Circle fit failed: {e}")
            return
        self.center, self.radius = (xc, yc), r
        self._draw_circle(xc, yc, r)
        self.mode = 'reference'
        self.log(f"✅ Fitted circle: center=({xc:.2f}, {yc:.2f}), r={r:.2f}")
        self._gate_controls()

    def on_confirm_reference(self):
        if self.center is None or len(self.outer_points) != 5:
            self.log("❌ Finish Step 1 first.");
            return
        if self.ref_point is None:
            self.log("❌ Click a reference point in the image first.");
            return

        delta = float(self.thr_var.get())
        mask, r_in, r_out = build_annulus_mask(self.center, self.ref_point, delta, (self.h, self.w))
        self.ring_mask = mask
        self.ring_inner = r_in
        self.ring_outer = r_out
        self.thresh_diam = delta  # keep as Δdiam entered

        xc, yc = self.center
        self._set_ring_line(self.ring_inner_line, xc, yc, r_in)
        self._set_ring_line(self.ring_outer_line, xc, yc, r_out)

        self.log(f"✅ Reference confirmed. Δdiam={delta:.1f}px → r_in={r_in:.1f}, r_out={r_out:.1f}")
        self.mode = 'robust_normalize'
        self._gate_controls()

    def on_apply_robust(self):
        low = float(self.cut_low_var.get());
        high = float(self.cut_high_var.get())
        self.robust_parameter = [max(0.0, low), max(0.0, high)]
        self.image_norm_full, stretch_vals = robust_normalize(
            self.gray, self.robust_parameter[0], self.robust_parameter[1],
            robust_fn=robust_contrast_normalization
        )
        self.im.set_data(self.image_norm_full);
        self.fig.canvas.draw_idle()
        self.log(f"✅ Robust normalization on FULL image: low={low:.2f}%, high={high:.2f}% stretch={stretch_vals}")
        self.mode = 'edges_detect'
        self._gate_controls()

    def on_apply_canny(self):
        if self.ring_mask is None:
            self.log("❌ No ROI mask. Confirm reference & threshold first.");
            return
        src = self.image_norm_full if self.image_norm_full is not None else self.gray
        self.edges = canny_roi(src, self.ring_mask, self.s_low.get(), self.s_high.get())

        src_u8 = (src if src.dtype == np.uint8 else
                  cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
        overlay = cv2.cvtColor(src_u8, cv2.COLOR_GRAY2BGR)
        overlay[self.edges > 0] = (0, 255, 255)
        self.im.set_data(overlay[..., ::-1]);
        self.fig.canvas.draw_idle()

        self.log(f"✅ Canny (ROI-only) applied (low={int(self.s_low.get())}, high={int(self.s_high.get())}).")
        self._gate_controls()

    def on_show_results(self):
        if self.edges is None:
            self.log("❌ Apply Canny first.");
            return

        base = self.image_norm_full if self.image_norm_full is not None else self.gray

        # fit + overlay (unlabeled)
        self.fit_results, vis_rgb = fit_and_overlay(
            base, self.edges,
            max_regions=2, pre_close_kernel=3, pre_dilate_iter=1,
            ransac_iters=1000, sample_size=30, residual_thresh=0.03,
            min_inliers_ratio=0.50, spline_smoothing=2.0, num_spline_samples=400
        )

        # re-label inner/outer by area + containment
        from Evaluation import classify_inner_outer, evaluate_fit_quality, detection_success
        classified, stats = classify_inner_outer(self.fit_results, (self.h, self.w), min_containment=0.95)

        # evaluate quality with the new labels
        self.quality = evaluate_fit_quality(classified, self.edges, tol_px=float(self.tol_px_var.get()))

        # success decision
        ok, msg = detection_success(self.quality, stats, tol_px=float(self.tol_px_var.get()),
                                    min_inlier=0.60, max_p95=5.0)

        # redraw overlay (same visualization; labels don’t change colors)
        self.im.set_data(vis_rgb);
        self.fig.canvas.draw_idle()

        # log a friendly summary
        def fmt(lbl):
            q = self.quality.get(lbl, {})
            return (f"{lbl}: mean(M→E)={q.get('mean_md', np.nan):.2f}px, p95={q.get('p95_md', np.nan):.2f}px, "
                    f"inlier≤tol={q.get('inlier_md', 0) * 100:.1f}% | "
                    f"mean(E→M)={q.get('mean_em', np.nan):.2f}px, p95={q.get('p95_em', np.nan):.2f}px, "
                    f"inlier≤tol={q.get('inlier_em', 0) * 100:.1f}%")

        self.log("Quality:\n  " + "\n  ".join([fmt('outer'), fmt('inner')]))
        self.log(f"Containment(inner⊆outer): {stats.get('containment', 0):.2f}  "
                 f"D_eq outer={stats['outer']['D_eq']:.1f}px  inner={stats['inner']['D_eq']:.1f}px")
        if ok:
            self.log("🎉 SUCCESS: Inner and outer cages detected.")
        else:
            self.log(f"⚠️ Not good enough: {msg}")

        # store labeled results for saving
        self.fit_results = classified
        self._gate_controls()

    def on_save_json(self):
        if not hasattr(self, "quality"): self.quality = {}
        state = {
            "outer_points": [(float(x), float(y)) for x, y in self.outer_points],
            "fitted_outer_circle": {
                "center": list(map(float, self.center)) if self.center else None,
                "radius": float(self.radius) if self.radius is not None else None
            },
            "reference_point": list(map(float, self.ref_point)) if self.ref_point else None,
            "delta_diam_px": float(self.thr_var.get()),
            "robust_cutoff_percent": [float(self.cut_low_var.get()), float(self.cut_high_var.get())],
            "canny_thresholds": {"low": int(self.s_low.get()), "high": int(self.s_high.get())},
            "fit_params": {
                "ransac_iters": 1000, "sample_size": 30, "residual_thresh": 0.03,
                "min_inliers_ratio": 0.5, "spline_smoothing": 2.0, "num_spline_samples": 400
            },
            "quality_tol_px": float(self.tol_px_var.get()),
            "quality_metrics": self.quality,
            "models": [
                ({
                     "label": r.get("label", "?"), "type": "ellipse",
                     "center": [float(r['model'].center[0]), float(r['model'].center[1])],
                     "axes": [float(r['model'].axes[0]), float(r['model'].axes[1])],
                     "angle_deg": float(r['model'].angle),
                     "inlier_ratio": float(r['model'].inlier_ratio),
                     "rmseF": float(r['model'].rmse)
                 } if getattr(r['model'], "type", "") == "ellipse" else
                 {"label": r.get("label", "?"), "type": "spline",
                  "smoothing": float(r['model'].smoothing),
                  "samples": r['model'].samples.astype(float).tolist()})
                for r in (self.fit_results or [])
            ]
        }
        sess = build_session_dict(state)
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path: self.log("⚠️ Save canceled."); return
        save_session_json(path, sess)
        self.log(f"💾 Saved: {path}")

    # ================== Resets ==================
    def on_reset_points(self):
        self.outer_points = []
        self.center = None
        self.radius = None
        if self.circle_line is not None:
            self.circle_line.set_data([], [])
        self._update_scatter()
        self.mode = 'outer_points'
        self.log("↩︎ Step 1 reset.")
        self._gate_controls()

    def on_reset_all(self):
        self.mode = 'outer_points'
        self.outer_points = []
        self.center = None
        self.radius = None
        self.ref_point = None
        self.ring_mask = None
        self.ring_inner = None
        self.ring_outer = None
        self.thresh_diam = None
        self.ring_inner_line.set_data([], [])
        self.ring_outer_line.set_data([], [])
        self.image_norm_full = None
        self.edges = None
        self.fit_results = None
        if self.circle_line is not None:
            self.circle_line.set_data([], [])
        if self.scatter is not None:
            self.scatter.set_offsets(np.empty((0,2)))
        if self.ref_scatter is not None:
            self.ref_scatter.set_offsets(np.empty((0,2)))
        self.im.set_data(self.gray)
        self.fig.canvas.draw_idle()
        self._reset_view()
        self.thr_var.set(20.0)
        self.cut_low_var.set(0.5); self.cut_high_var.set(0.5)
        self.s_low.set(60); self.s_high.set(140)
        self.log("🔄 Reset ALL.")
        self._gate_controls()

    # ================== run ==================
    def run(self):
        self.root.mainloop()
