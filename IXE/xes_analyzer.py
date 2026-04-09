# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk, filedialog, simpledialog
from tkinter.colorchooser import askcolor
import numpy as np
from PIL import Image
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from skimage.filters import threshold_otsu
#from .remove_cross import delete_zero_rows_and_columns
#from .spectrum_utils import SpectrumProcessor
import matplotlib as mpl
from matplotlib import rcParams
import sys
import csv
import re
import random
from IXE.remove_cross import delete_zero_rows_and_columns
from IXE.spectrum_utils import SpectrumProcessor
from IXE import image_processing
from IXE import iad_controls
from IXE import spectrum_controls
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'IXE'))
#print(sys.path)


class RoundedButton(tk.Canvas):
    """Compact rounded button for emphasizing the primary action."""

    def __init__(
        self,
        parent,
        text,
        command,
        width=108,
        height=30,
        radius=14,
        border_width=1,
        fill="#f7faf7",
        outline="#1f5c3f",
        text_color="#1f5c3f",
        active_fill="#e4efe6",
        font=("Arial", 12, "bold"),
    ):
        style = ttk.Style()
        canvas_bg = style.lookup('TFrame', 'background') or parent.winfo_toplevel().cget('bg')
        super().__init__(
            parent,
            width=width,
            height=height,
            highlightthickness=0,
            bd=0,
            relief="flat",
            bg=canvas_bg,
            cursor="hand2",
        )
        self.command = command
        self.button_width = width
        self.button_height = height
        self.radius = radius
        self.border_width = border_width
        self.fill = fill
        self.outline = outline
        self.text_color = text_color
        self.active_fill = active_fill
        self.font = font
        self.text = text
        self._pressed = False
        self._draw(self.fill)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Leave>", self._on_leave)

    def _rounded_points(self):
        w = self.button_width - 1
        h = self.button_height - 1
        r = min(self.radius, w / 2, h / 2)
        return [
            r, 0,
            r, 0,
            w - r, 0,
            w - r, 0,
            w, 0,
            w, r,
            w, r,
            w, h - r,
            w, h - r,
            w, h,
            w - r, h,
            w - r, h,
            r, h,
            r, h,
            0, h,
            0, h - r,
            0, h - r,
            0, r,
            0, r,
            0, 0,
        ]

    def _draw(self, fill_color):
        self.delete("all")
        self.create_polygon(
            self._rounded_points(),
            smooth=True,
            splinesteps=24,
            fill=fill_color,
            outline=self.outline,
            width=self.border_width,
        )
        self.create_text(
            self.button_width / 2,
            self.button_height / 2,
            text=self.text,
            fill=self.text_color,
            font=self.font,
        )

    def _on_press(self, _event):
        self._pressed = True
        self._draw(self.active_fill)

    def _on_release(self, event):
        inside = 0 <= event.x <= self.button_width and 0 <= event.y <= self.button_height
        self._pressed = False
        self._draw(self.fill)
        if inside and callable(self.command):
            self.command()

    def _on_leave(self, _event):
        if self._pressed:
            self._pressed = False
            self._draw(self.fill)


class RangeSlider(tk.Canvas):
    """Two-handle horizontal slider for selecting a row interval."""

    def __init__(
        self,
        parent,
        left_var,
        right_var,
        min_value=0,
        max_value=1,
        width=320,
        height=52,
        command=None,
        allow_overlap=False,
    ):
        style = ttk.Style()
        canvas_bg = style.lookup('TFrame', 'background') or parent.winfo_toplevel().cget('bg')
        super().__init__(
            parent,
            width=width,
            height=height,
            highlightthickness=0,
            bd=0,
            relief="flat",
            bg=canvas_bg,
            cursor="hand2",
        )
        self.left_var = left_var
        self.right_var = right_var
        self.min_value = min_value
        self.max_value = max(max_value, min_value)
        self.slider_width = width
        self.slider_height = height
        self.command = command
        self.allow_overlap = allow_overlap
        self.track_left = 22
        self.track_right = width - 22
        self.track_y = 28
        self.handle_radius = 8
        self._active_handle = None
        self._hover_handle = None
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Motion>", self._on_motion)
        self.bind("<Configure>", self._on_configure)
        self.bind("<Leave>", self._on_leave)
        self.set_limits(self.min_value, self.max_value, self.left_var.get(), self.right_var.get(), invoke=False)

    def set_limits(self, min_value, max_value, left_value=None, right_value=None, invoke=True):
        self.min_value = min_value
        self.max_value = max(max_value, min_value)
        if left_value is None:
            left_value = self.left_var.get()
        if right_value is None:
            right_value = self.right_var.get()
        self._set_values(left_value, right_value, invoke=invoke)

    def _value_span(self):
        return max(self.max_value - self.min_value, 1)

    def _value_to_x(self, value):
        if self.max_value <= self.min_value:
            return self.track_left
        fraction = (value - self.min_value) / self._value_span()
        return self.track_left + fraction * (self.track_right - self.track_left)

    def _x_to_value(self, x_pos):
        if self.max_value <= self.min_value:
            return self.min_value
        x_pos = min(max(x_pos, self.track_left), self.track_right)
        fraction = (x_pos - self.track_left) / (self.track_right - self.track_left)
        return int(round(self.min_value + fraction * self._value_span()))

    def _clamp_pair(self, left_value, right_value):
        left_value = min(max(int(left_value), self.min_value), self.max_value)
        right_value = min(max(int(right_value), self.min_value), self.max_value)
        if self.allow_overlap:
            return left_value, right_value
        if self.max_value > self.min_value:
            if left_value >= right_value:
                if left_value >= self.max_value:
                    left_value = self.max_value - 1
                    right_value = self.max_value
                else:
                    right_value = left_value + 1
        else:
            left_value = self.min_value
            right_value = self.max_value
        return left_value, right_value

    def _set_values(self, left_value, right_value, invoke=True):
        left_value, right_value = self._clamp_pair(left_value, right_value)
        self.left_var.set(left_value)
        self.right_var.set(right_value)
        self._draw()
        if invoke and self.command:
            self.command(left_value, right_value)

    def _draw(self):
        self.delete("all")
        left_x = self._value_to_x(self.left_var.get())
        right_x = self._value_to_x(self.right_var.get())

        self.create_line(
            self.track_left,
            self.track_y,
            self.track_right,
            self.track_y,
            fill="#9da7a1",
            width=2,
        )
        self.create_line(
            left_x,
            self.track_y,
            right_x,
            self.track_y,
            fill="#1f5c3f",
            width=3,
        )
        self._draw_handle(left_x, "#1f5c3f")
        self._draw_handle(right_x, "#1f5c3f")
        visible_handle = self._active_handle or self._hover_handle
        if visible_handle:
            self._draw_value_tag(visible_handle)

    def _draw_handle(self, x_pos, fill_color):
        self.create_oval(
            x_pos - self.handle_radius,
            self.track_y - self.handle_radius,
            x_pos + self.handle_radius,
            self.track_y + self.handle_radius,
            fill=fill_color,
            outline="white",
            width=1,
        )

    def _draw_value_tag(self, handle_name):
        if handle_name == "left":
            value = self.left_var.get()
            x_pos = self._value_to_x(value)
        else:
            value = self.right_var.get()
            x_pos = self._value_to_x(value)
        text_id = self.create_text(
            x_pos,
            max(self.track_y - 18, 10),
            text=str(value),
            fill="#1f5c3f",
            font=("Arial", 10, "bold"),
        )
        x0, y0, x1, y1 = self.bbox(text_id)
        pad_x = 5
        pad_y = 2
        bg_id = self.create_rectangle(
            x0 - pad_x,
            y0 - pad_y,
            x1 + pad_x,
            y1 + pad_y,
            fill="#f8fbf8",
            outline="#1f5c3f",
            width=1,
        )
        self.tag_lower(bg_id, text_id)

    def _nearest_handle(self, x_pos):
        left_distance = abs(x_pos - self._value_to_x(self.left_var.get()))
        right_distance = abs(x_pos - self._value_to_x(self.right_var.get()))
        return "left" if left_distance <= right_distance else "right"

    def _handle_under_cursor(self, x_pos, y_pos):
        if abs(y_pos - self.track_y) > self.handle_radius + 6:
            return None
        nearest_handle = self._nearest_handle(x_pos)
        handle_x = self._value_to_x(self.left_var.get() if nearest_handle == "left" else self.right_var.get())
        if abs(x_pos - handle_x) <= self.handle_radius + 6:
            return nearest_handle
        return None

    def _on_press(self, event):
        self._active_handle = self._nearest_handle(event.x)
        self._hover_handle = self._active_handle
        self._update_active_handle(event.x)

    def _on_drag(self, event):
        if self._active_handle:
            self._update_active_handle(event.x)

    def _on_release(self, event):
        if self._active_handle:
            self._update_active_handle(event.x)
            self._active_handle = None
        self._hover_handle = self._handle_under_cursor(event.x, event.y)
        self._draw()

    def _on_motion(self, event):
        if self._active_handle:
            return
        hover_handle = self._handle_under_cursor(event.x, event.y)
        if hover_handle != self._hover_handle:
            self._hover_handle = hover_handle
            self._draw()

    def _on_configure(self, event):
        self.slider_width = max(event.width, 80)
        self.slider_height = max(event.height, 36)
        self.track_left = 22
        self.track_right = max(self.slider_width - 22, self.track_left)
        self.track_y = max(self.slider_height // 2, self.handle_radius + 6)
        self._draw()

    def _on_leave(self, _event):
        if self._active_handle:
            return
        if self._hover_handle is not None:
            self._hover_handle = None
            self._draw()

    def _update_active_handle(self, x_pos):
        value = self._x_to_value(x_pos)
        if self._active_handle == "left":
            self._set_values(value, self.right_var.get())
        elif self._active_handle == "right":
            self._set_values(self.left_var.get(), value)


class TIFFAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1200x900")
        self.root.title("XES Analyzer")
        self.setup_fonts()
        # Initialize parameters
        self.parm = {
            'tilt_cor': 0,
            'threshold': 0,
            'vmin': 0,
            'vmax': 100,
            'row_begin': 0,
            'row_end': 1,
            'column_begin': 0,
            'column_end': 1,
            'bg_row_begin': 0,
            'bg_row_end': 0,
            'bg_col_begin': 320,
            'bg_col_end': 618, 
            'n_moveavg': 5,
            'PK intersect': {'i_l': 50, 'i_r': 120}, # Default range for satellite peak intersection
            'eye_ball_cross': None,
        }
        
        # Initialize state variables
        #self.current_image = None
        self.bg_subtraction_enabled = False
        self.spect_processor = None
        
        # Main layout structure
        self.setup_main_layout()
        
        # Initialize UI components
        self.setup_controls()
        #self.setup_display()
        self.setup_display_area()

    def setup_main_layout(self):
         # Top panel for controls
        self.control_panel = ttk.Frame(self.root)
        self.control_panel.pack(fill=tk.X, padx=5, pady=5)
        
        # Bottom panel for display
        self.display_panel = ttk.Frame(self.root)
        self.display_panel.pack(fill=tk.BOTH, expand=True)
        
        # Left side for images (set fixed width)
        self.image_panel = ttk.Frame(self.display_panel, width=700)
        self.image_panel.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        self.image_panel.pack_propagate(False)
        # Right side for spectrum
        self.spectrum_panel = ttk.Frame(self.display_panel)
        self.spectrum_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)


    def setup_controls(self):
        """Set up the control buttons and entries"""
        # Notebook for tabs
        self.notebook = ttk.Notebook(self.control_panel)
        self.notebook.pack(fill=tk.X)
        
        # Image processing tab
        self.processing_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.processing_tab, text="Image Processing")
        
        # Spectrum analysis tab
        self.spectrum_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.spectrum_tab, text="Spectrum Analysis")
        # IAD Calculation tab
        self.iad_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.iad_tab, text="IAD Calculation")

        # Setup controls for each tab
        self.setup_processing_controls()
        self.setup_spectrum_controls()
        self.setup_iad_controls()

    def setup_processing_controls(self):
        """Image processing controls"""
        frame = ttk.Frame(self.processing_tab, padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Define the number of columns
        frame.grid_columnconfigure(0, weight=0)  # Makes column 0 expand
        frame.grid_columnconfigure(1, weight=0)  # Makes column 1 expand more
        frame.grid_columnconfigure(2, weight=0)  # Makes column 2 expand equally
        frame.grid_columnconfigure(3, weight=0)  # Makes column 3 expand equally (optional)
        frame.grid_columnconfigure(4, weight=0)  # Makes column 3 expand equally (optional)
        frame.grid_columnconfigure(5, weight=0)  # Makes column 3 expand equally (optional)
        frame.grid_columnconfigure(6, weight=0) 
        frame.grid_columnconfigure(7, weight=0) 
        frame.grid_columnconfigure(8, weight=0)  # Makes column 0 expand
        frame.grid_columnconfigure(9, weight=0)  # Makes column 1 expand more
        frame.grid_columnconfigure(10, weight=0)  # Makes column 2 expand equally
        frame.grid_columnconfigure(11, weight=0)  # Makes column 3 expand equally (optional)
        frame.grid_columnconfigure(12, weight=0)  # Makes column 3 expand equally (optional)
        frame.grid_columnconfigure(13, weight=0)  # Makes column 3 expand equally (optional)
        frame.grid_columnconfigure(14, weight=1) 
        #frame.grid_columnconfigure(15, weight=1) 
        # First row - File Path and Import TIFF
        # Button in the first column (to place ahead of File Path)
        ttk.Button(frame, text="Import TIFF", command=self.load_tiff).grid(row=0, column=0,columnspan=2, padx=5, pady=5)
        ttk.Label(frame, text="File Path:").grid(row=0, column=3,columnspan=2, padx=5, pady=5, sticky="w")
        self.file_path_entry = ttk.Entry(frame, width=150, font=('Arial', 12))
        self.file_path_entry.insert(0, " ")  # Set the default placeholder tex
        self.file_path_entry.grid(row=0, column=5, columnspan=10, padx=5, pady=5)
        # Second row - Threshold, vmin, vmax, Tilt controls
        ttk.Label(frame, text="Threshold:").grid(row=1, column=0,columnspan=2, padx=5, pady=5)
        self.thresh_entry = ttk.Entry(frame, width=4, font=('Arial', 12))
        self.thresh_entry.insert(0, str(self.parm['threshold']))
        self.thresh_entry.grid(row=1, column=3, padx=5, pady=5)
        ttk.Button(frame, text="Auto", command=self.auto_threshold).grid(row=1, column=5, padx=5, pady=5)
        
        ttk.Label(frame, text="vmin:").grid(row=1, column=7, padx=5, pady=5)
        self.vmin_entry = ttk.Entry(frame, width=4, font=('Arial', 12))
        self.vmin_entry.insert(0, str(self.parm['vmin']))
        self.vmin_entry.grid(row=1, column=9, padx=5, pady=5)
        
        ttk.Label(frame, text="vmax:").grid(row=1, column=11, padx=5, pady=5)
        self.vmax_entry = ttk.Entry(frame, width=4, font=('Arial', 12))
        self.vmax_entry.insert(0, str(self.parm['vmax']))
        self.vmax_entry.grid(row=1, column=13, padx=5, pady=5)
        
        # Third row - Tilt angle
        ttk.Label(frame, text="Tilt:").grid(row=2, column=0, padx=5, pady=5)
        self.tilt_entry = ttk.Entry(frame, width=4, font=('Arial', 12))
        self.tilt_entry.insert(0, str(self.parm['tilt_cor']))
        self.tilt_entry.grid(row=2, column=3, padx=5, pady=5)
        
        # Fourth row - Process buttons
        ttk.Button(frame, text="Remove Cross", command=self.remove_cross_raw).grid(row=2, column=5, columnspan=3,padx=5, pady=5)
        ttk.Button(frame, text="Save", command=self.save_processed_image).grid(row=2, column=9, columnspan=3,padx=5, pady=5)
        self.process_button = RoundedButton(
            frame,
            text="Process >",
            command=self.process_image,
            width=104,
            height=28,
            radius=13,
            border_width=1,
            fill="#f8fbf8",
            outline="#1f5c3f",
            text_color="#1f5c3f",
            active_fill="#e6f0e8",
            font=("Arial", 12, "bold"),
        )
        self.process_button.grid(row=2, column=13, padx=5, pady=5)

    def setup_spectrum_controls(self):
        """Spectrum analysis controls"""
        frame = ttk.Frame(self.spectrum_tab)
        frame.pack(fill=tk.X, padx=5, pady=5)
        frame.grid_columnconfigure(0, weight=0)  
        frame.grid_columnconfigure(1, weight=1)  
        frame.grid_columnconfigure(2, weight=0) 
        # ROI controls (still in the same row)
        ttk.Label(frame, text="ROI Rows:").grid(row=0, column=0, padx=5)
        self.row_begin = tk.IntVar(value=self.parm['row_begin'])
        self.row_end = tk.IntVar(value=self.parm['row_end'])
        self.row_slider = RangeSlider(
            frame,
            left_var=self.row_begin,
            right_var=self.row_end,
            min_value=0,
            max_value=max(self.parm['row_end'], self.parm['row_begin'] + 1),
            width=320,
            height=52,
            command=self.on_roi_rows_changed,
        )
        self.row_slider.grid(row=0, column=1, padx=(5, 8), sticky="ew")
        self.on_roi_rows_changed(self.row_begin.get(), self.row_end.get(), redraw=False)

        ttk.Label(frame, text="ROI Columns:").grid(row=1, column=0, padx=5)
        self.column_begin = tk.IntVar(value=self.parm['column_begin'])
        self.column_end = tk.IntVar(value=self.parm['column_end'])
        self.column_slider = RangeSlider(
            frame,
            left_var=self.column_begin,
            right_var=self.column_end,
            min_value=0,
            max_value=max(self.parm['column_end'], self.parm['column_begin'] + 1),
            width=320,
            height=52,
            command=self.on_roi_cols_changed,
        )
        self.column_slider.grid(row=1, column=1, padx=(5, 8), sticky="ew")
        self.on_roi_cols_changed(self.column_begin.get(), self.column_end.get(), redraw=False)
        self.plot_button = RoundedButton(
            frame,
            text="Plot",
            command=self.plot_roi_spectrum,
            width=116,
            height=34,
            radius=15,
            border_width=1,
            fill="#f8fbf8",
            outline="#1f5c3f",
            text_color="#1f5c3f",
            active_fill="#e6f0e8",
            font=("Arial", 13, "bold"),
        )
        self.plot_button.grid(row=1, column=2, padx=(4, 6), pady=5, sticky="w")
        
        ttk.Label(frame, text="BG Rows:").grid(row=2, column=0, padx=5)
        self.bg_row_begin = tk.IntVar(value=self.parm['bg_row_begin'])
        self.bg_row_end = tk.IntVar(value=self.parm['bg_row_end'])
        self.bg_row_slider = RangeSlider(
            frame,
            left_var=self.bg_row_begin,
            right_var=self.bg_row_end,
            min_value=0,
            max_value=max(self.parm['bg_row_end'], self.parm['bg_row_begin'] + 1),
            width=320,
            height=52,
            command=self.on_bg_rows_changed,
            allow_overlap=True,
        )
        self.bg_row_slider.grid(row=2, column=1, padx=(5, 8), sticky="ew")
        self.on_bg_rows_changed(self.bg_row_begin.get(), self.bg_row_end.get(), redraw=False)

        self.bg_toggle = ttk.Button(frame, text="BG Remove", command=self.toggle_background_removal)
        self.bg_toggle.grid(row=2, column=2, padx=(4, 6), pady=5, sticky="w")
        self.line_color = tk.StringVar(value="black")

        self.line_controls_row = ttk.Frame(frame)
        self.line_controls_row.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(0, 0))
        self.line_controls_row.grid_columnconfigure(0, weight=1)
        self.line_controls_row.grid_columnconfigure(1, weight=1)
        self.line_controls_row.grid_columnconfigure(2, weight=1)
        self.line_controls_row.grid_columnconfigure(3, weight=1)
        self.line_controls_row.grid_columnconfigure(4, weight=1)
        self.line_controls_row.grid_columnconfigure(5, weight=1)
        self.line_controls_row.grid_columnconfigure(6, weight=1)
        self.line_controls_row.grid_columnconfigure(7, weight=1)
        self.line_controls_row.grid_columnconfigure(8, weight=1)
        self.line_controls_row.grid_columnconfigure(9, weight=1)

        ttk.Button(self.line_controls_row, text="Pick Color", command=self.open_color_picker).grid(row=0, column=0, padx=6, pady=2)
        ttk.Label(self.line_controls_row, text="Line Style:").grid(row=0, column=1, padx=6, pady=2, sticky="e")
        self.line_style = ttk.Combobox(self.line_controls_row, values=["-", "--", "-.", ":"], width=4)
        self.line_style.set("-")  # Default line style
        self.line_style.grid(row=0, column=2, padx=6, pady=2, sticky="w")
        self.line_style.bind("<<ComboboxSelected>>", lambda event: self.update_plot)
        ttk.Label(self.line_controls_row, text="Line Width:").grid(row=0, column=3, padx=6, pady=2, sticky="e")
        self.line_width = ttk.Entry(self.line_controls_row, width=5)
        self.line_width.insert(0, "0.5")  # Default line width
        self.line_width.grid(row=0, column=4, padx=6, pady=2, sticky="w")
        self.line_width.bind("<KeyRelease>", lambda event: self.update_plot)
        ttk.Label(self.line_controls_row, text="Smooth (n):").grid(row=0, column=5, padx=6, pady=2, sticky="e")
        self.n_moveavg = ttk.Entry(self.line_controls_row, width=4)
        self.n_moveavg.insert(0, str(self.parm['n_moveavg']))
        self.n_moveavg.grid(row=0, column=6, padx=6, pady=2, sticky="w")
        ttk.Button(self.line_controls_row, text="Apply Smooth", command=self.apply_moving_average).grid(row=0, column=7, padx=6, pady=2)
        ttk.Button(self.line_controls_row, text="Save Spectrum", command=self.save_spectrum_data).grid(row=0, column=8, padx=6, pady=2)
        ttk.Button(self.line_controls_row, text="Save Image", command=self.save_spectrum_image).grid(row=0, column=9, padx=6, pady=2)

    def setup_iad_controls(self):
        """IAD Calculation controls"""
        frame = ttk.Frame(self.iad_tab, padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create toolbar for IAD controls
        self.iad_toolbar = ttk.Frame(frame)
        self.iad_toolbar.grid(row=0, column=0, sticky='w', padx=5, pady=5)  # First row for toolbar buttons

        ttk.Button(self.iad_toolbar, text="Import Ref. Data", command=self.import_ref_data).grid(row=0, column=0, padx=5)
        ttk.Button(self.iad_toolbar, text="Remove", command=self.remove_selected_line).grid(row=0, column=1, padx=5)
        ttk.Button(self.iad_toolbar, text="Color", command=self.change_selected_line_color).grid(row=0, column=2, padx=5)
        ttk.Button(self.iad_toolbar, text="Integrated Diff.", command=self.plot_integrated_diff).grid(row=0, column=3, padx=5)
        
        # Second row (Satellite Diff. and cross entries)
        second_row_frame = ttk.Frame(frame)
        second_row_frame.grid(row=1, column=0, sticky='w', padx=5, pady=5)

        ttk.Button(second_row_frame, text="Satellite Diff.", command=self.calculate_and_display_satellite_peak_iad).grid(row=0, column=0, padx=5)
        
        # Spectra Cross (with entries)
        spectra_cross_frame = ttk.Frame(second_row_frame)
        spectra_cross_frame.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(spectra_cross_frame, text="Spectra Cross:").grid(row=0, column=0, padx=5)
        self.cross_begin = ttk.Entry(spectra_cross_frame, width=4)
        self.cross_begin.insert(0, str(self.parm['PK intersect']['i_l']))
        self.cross_begin.grid(row=0, column=1, padx=5)
        ttk.Label(spectra_cross_frame, text="-").grid(row=0, column=2, padx=5)
        self.cross_end = ttk.Entry(spectra_cross_frame, width=4)
        self.cross_end.insert(0, str(self.parm['PK intersect']['i_r']))
        self.cross_end.grid(row=0, column=3, padx=5)
        
        # Eye Ball Cross (with entries)
        eyeball_cross_frame = ttk.Frame(second_row_frame)
        eyeball_cross_frame.grid(row=0, column=2, padx=5, pady=5)
        ttk.Label(eyeball_cross_frame, text="Eye Ball Cross:").grid(row=0, column=0, padx=5)
        self.eye_ball_cross = ttk.Entry(eyeball_cross_frame, width=4)
        eye_ball_val = self.parm.get('eye_ball_cross', 0)
        if eye_ball_val is None:
            self.eye_ball_cross.insert(0, "")
        else:
            self.eye_ball_cross.insert(0, str(eye_ball_val))
        self.eye_ball_cross.grid(row=0, column=1, padx=5)

        # List of plotted lines (to be placed in the first row)
        self.line_list_frame = ttk.Frame(frame)
        self.line_list_frame.grid(row=2, column=0, columnspan=4, sticky='w', padx=5, pady=5)  # Ensure it's below the second row
        self.plotted_lines = {}
        
    def setup_display_area(self):
        """Set up the display areas with larger fonts and higher resolution"""
        try:
            self.display_scale = max(1.0, float(self.root.tk.call('tk', 'scaling')))
        except tk.TclError:
            self.display_scale = 1.0

        # Keep a logical display DPI for layout, then render the Matplotlib
        # canvas at a higher pixel density for sharper ticks on Retina screens.
        self.base_display_dpi = 100
        self.fig_dpi = int(self.base_display_dpi * self.display_scale)
        
        # Single frame for images (raw or processed)
        self.image_frame = ttk.LabelFrame(self.image_panel, text="Image Display", padding=(10, 10), width=430, height=350)
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.image_frame.pack_propagate(False)
        self.run_number_label = ttk.Label(self.image_frame, text="Run: Not Loaded", font=("Arial", 16))
        self.run_number_label.pack(side=tk.TOP, padx=5)

        self.image_plot_frame = ttk.Frame(self.image_frame)
        self.image_plot_frame.pack(fill=tk.BOTH, expand=True)
        self.image_plot_frame.columnconfigure(1, weight=1)
        self.image_plot_frame.rowconfigure(0, weight=1)

        self.image_ylabel_canvas = tk.Canvas(
            self.image_plot_frame,
            width=40,
            highlightthickness=0,
            bd=0,
        )
        self.image_ylabel_canvas.grid(row=0, column=0, sticky="ns")
        self.image_ylabel_canvas.bind("<Configure>", self.update_image_ylabel)

        # Smaller figure size for image display
        self.fig_image = plt.Figure(figsize=(6, 5), dpi=self.fig_dpi)
        self.fig_image.subplots_adjust(left=0.14, bottom=0.14, right=0.88, top=0.95)
        self.ax_image = self.fig_image.add_subplot(111)
        self.cbar = None
        # Hide ticks and ticklabels for image axes initially
        self.ax_image.set_xticks([])
        self.ax_image.set_yticks([])
        self.ax_image.set_xticklabels([])
        self.ax_image.set_yticklabels([])
        for spine in self.ax_image.spines.values():
            spine.set_linewidth(0.5)
        self.canvas_image = FigureCanvasTkAgg(self.fig_image, master=self.image_plot_frame)
        self.canvas_image.get_tk_widget().grid(row=0, column=1, sticky="nsew")

        self.image_xlabel_label = ttk.Label(self.image_frame, text="Columns", font=("Arial", 14))
        self.image_xlabel_label.pack(side=tk.BOTTOM, pady=(2, 0))
        
        # Spectrum frame (right side) - larger
        spectrum_frame = ttk.LabelFrame(self.spectrum_panel, text="Spectrum", padding=(10, 10))
        spectrum_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.fig_spectrum = plt.Figure(figsize=(6, 7), dpi=self.fig_dpi)
        self.ax_spectrum = self.fig_spectrum.add_subplot(111)
        # Hide ticks and ticklabels for spectrum axes initially
        self.ax_spectrum.set_xticks([])
        self.ax_spectrum.set_yticks([])
        self.ax_spectrum.set_xticklabels([])
        self.ax_spectrum.set_yticklabels([])
        for spine in self.ax_spectrum.spines.values():
            spine.set_linewidth(0.5)
        self.canvas_spectrum = FigureCanvasTkAgg(self.fig_spectrum, master=spectrum_frame)
        self.canvas_spectrum.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # Add run number label to spectrum display
        self.run_number_label_spectrum = ttk.Label(spectrum_frame, text="Run: Not Loaded", font=("Arial", 12))
        self.run_number_label_spectrum.pack(side=tk.TOP, padx=5)
        # Larger toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas_spectrum, spectrum_frame)
        self.toolbar.update()
        self.toolbar.pack(fill=tk.X)


    def setup_fonts(self):
        """Configure Arial font with larger sizes throughout the application"""
        # Set default font for all widgets (larger size)
        if sys.platform == 'win32':
            import matplotlib.font_manager as fm
            fm._rebuild()
            
        default_font = ('Arial', 14)  # Increased from 10 to 12
        self.root.option_add('*Font', default_font)
        
        # Configure Ttk styles for larger fonts
        style = ttk.Style()
        style.configure('.', font=('Arial', 14))
        style.configure('TButton', font=('Arial', 14))
        style.configure('TLabel', font=('Arial', 14))
        style.configure('TEntry', font=('Arial', 14))
        
        # Configure matplotlib to use Arial with larger sizes
        mpl.rcParams.update({
            'font.family': 'Arial',
            'font.size': 11,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 12
        })

    def update_image_ylabel(self, event=None):
        """Render the image y-axis label outside the plot area."""
        if not hasattr(self, 'image_ylabel_canvas'):
            return
        canvas = self.image_ylabel_canvas
        canvas.delete("all")
        height = max(canvas.winfo_height(), 1)
        width = max(canvas.winfo_width(), 1)
        canvas.create_text(
            width / 2,
            height / 2,
            text="Rows",
            angle=90,
            font=("Arial", 14),
        )

    def on_roi_rows_changed(self, row_begin=None, row_end=None, redraw=True):
        """Sync the selected ROI row range with the UI and image overlay."""
        if row_begin is None:
            row_begin = self.row_begin.get()
        if row_end is None:
            row_end = self.row_end.get()
        row_begin = int(row_begin)
        row_end = int(row_end)
        self.parm['row_begin'] = row_begin
        self.parm['row_end'] = row_end
        if hasattr(self, 'row_range_label'):
            row_count = (getattr(self, 'row_slider', None).max_value + 1) if hasattr(self, 'row_slider') else (row_end + 1)
            self.row_range_label.config(text=f"{row_begin} - {row_end}  ({row_count} rows)")
        if redraw and hasattr(self, 'current_display_data'):
            self.refresh_image_overlays()

    def update_roi_row_slider_limits(self, row_count):
        """Resize the ROI row slider to the current image height."""
        if not hasattr(self, 'row_slider'):
            return
        max_row = max(int(row_count) - 1, 0)
        row_begin = min(self.row_begin.get(), max_row)
        row_end = min(self.row_end.get(), max_row)
        if max_row > 0 and row_begin >= row_end:
            if row_begin >= max_row:
                row_begin = max_row - 1
                row_end = max_row
            else:
                row_end = row_begin + 1
        elif max_row == 0:
            row_begin = 0
            row_end = 0
        self.row_slider.set_limits(0, max_row, row_begin, row_end, invoke=False)
        self.on_roi_rows_changed(self.row_begin.get(), self.row_end.get(), redraw=False)

    def get_roi_row_range(self):
        """Return the current ROI row selection from the slider controls."""
        row_begin, row_end = sorted((int(self.row_begin.get()), int(self.row_end.get())))
        self.parm['row_begin'] = row_begin
        self.parm['row_end'] = row_end
        if hasattr(self, 'row_range_label'):
            self.on_roi_rows_changed(row_begin, row_end, redraw=False)
        return row_begin, row_end

    def on_roi_cols_changed(self, column_begin=None, column_end=None, redraw=True):
        """Sync the selected ROI column range with the UI and image overlay."""
        if column_begin is None:
            column_begin = self.column_begin.get()
        if column_end is None:
            column_end = self.column_end.get()
        column_begin = int(column_begin)
        column_end = int(column_end)
        self.parm['column_begin'] = column_begin
        self.parm['column_end'] = column_end
        if hasattr(self, 'column_range_label'):
            col_count = (getattr(self, 'column_slider', None).max_value + 1) if hasattr(self, 'column_slider') else (column_end + 1)
            self.column_range_label.config(text=f"{column_begin} - {column_end}  ({col_count} cols)")
        if redraw and hasattr(self, 'current_display_data'):
            self.refresh_image_overlays()

    def update_roi_col_slider_limits(self, col_count):
        """Resize the ROI column slider to the current image width."""
        if not hasattr(self, 'column_slider'):
            return
        max_col = max(int(col_count) - 1, 0)
        column_begin = min(self.column_begin.get(), max_col)
        column_end = min(self.column_end.get(), max_col)
        if max_col > 0 and column_begin >= column_end:
            if column_begin >= max_col:
                column_begin = max_col - 1
                column_end = max_col
            else:
                column_end = column_begin + 1
        elif max_col == 0:
            column_begin = 0
            column_end = 0
        self.column_slider.set_limits(0, max_col, column_begin, column_end, invoke=False)
        self.on_roi_cols_changed(self.column_begin.get(), self.column_end.get(), redraw=False)

    def get_roi_col_range(self):
        """Return the current ROI column selection from the slider controls."""
        column_begin, column_end = sorted((int(self.column_begin.get()), int(self.column_end.get())))
        self.parm['column_begin'] = column_begin
        self.parm['column_end'] = column_end
        if hasattr(self, 'column_range_label'):
            self.on_roi_cols_changed(column_begin, column_end, redraw=False)
        return column_begin, column_end

    def on_bg_rows_changed(self, bg_row_begin=None, bg_row_end=None, redraw=True):
        """Sync the selected background row range with the UI and image overlay."""
        if bg_row_begin is None:
            bg_row_begin = self.bg_row_begin.get()
        if bg_row_end is None:
            bg_row_end = self.bg_row_end.get()
        bg_row_begin = int(bg_row_begin)
        bg_row_end = int(bg_row_end)
        self.parm['bg_row_begin'] = bg_row_begin
        self.parm['bg_row_end'] = bg_row_end
        if hasattr(self, 'bg_row_range_label'):
            row_count = (getattr(self, 'bg_row_slider', None).max_value + 1) if hasattr(self, 'bg_row_slider') else (bg_row_end + 1)
            self.bg_row_range_label.config(text=f"{bg_row_begin} - {bg_row_end}  ({row_count} rows)")
        if redraw and hasattr(self, 'current_display_data'):
            self.refresh_image_overlays()

    def update_bg_row_slider_limits(self, row_count):
        """Resize the background row slider to the current image height."""
        if not hasattr(self, 'bg_row_slider'):
            return
        max_row = max(int(row_count) - 1, 0)
        bg_row_begin = min(self.bg_row_begin.get(), max_row)
        bg_row_end = min(self.bg_row_end.get(), max_row)
        self.bg_row_slider.set_limits(0, max_row, bg_row_begin, bg_row_end, invoke=False)
        self.on_bg_rows_changed(self.bg_row_begin.get(), self.bg_row_end.get(), redraw=False)

    def get_bg_row_range(self):
        """Return the current background row selection from the slider controls."""
        bg_row_begin, bg_row_end = sorted((int(self.bg_row_begin.get()), int(self.bg_row_end.get())))
        self.parm['bg_row_begin'] = bg_row_begin
        self.parm['bg_row_end'] = bg_row_end
        if hasattr(self, 'bg_row_range_label'):
            self.on_bg_rows_changed(bg_row_begin, bg_row_end, redraw=False)
        return bg_row_begin, bg_row_end

    def refresh_image_overlays(self):
        """Redraw ROI lines and background ROI overlay on the image canvas."""
        if not hasattr(self, 'current_display_data'):
            return
        self.draw_background_roi()
        self.draw_lines()

    def reset_spectrum_state(self):
        """Clear stale spectrum state when a new TIFF is loaded."""
        self.spect_processor = None
        self.bg_subtraction_enabled = False
        if hasattr(self, 'bg_toggle'):
            self.bg_toggle.config(style='TButton', text="BG Remove")
        for attr_name in ('last_spectrum_roi', 'current_spectrum_moving_avg', 'current_spectrum_bg_removed'):
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        if hasattr(self, 'ax_spectrum'):
            self.ax_spectrum.clear()
            self.ax_spectrum.set_xticks([])
            self.ax_spectrum.set_yticks([])
            self.ax_spectrum.set_xticklabels([])
            self.ax_spectrum.set_yticklabels([])
            for spine in self.ax_spectrum.spines.values():
                spine.set_linewidth(0.5)
            self.canvas_spectrum.draw_idle()
        
    # Delegated methods only, all previous method bodies for moved functions are removed
    def load_tiff(self):
        return image_processing.load_tiff(self)
    def auto_threshold(self):
        return image_processing.auto_threshold(self)
    def remove_cross_raw(self):
        return image_processing.remove_cross_raw(self)
    def process_image(self):
        return image_processing.process_image(self)
    def display_image(self, data, ax, title):
        return image_processing.display_image(self, data, ax, title)
    def draw_lines(self):
        return image_processing.draw_lines(self)
    def draw_background_roi(self):
        return image_processing.draw_background_roi(self)
    def save_processed_image(self):
        return image_processing.save_processed_image(self)
    def import_ref_data(self):
        return iad_controls.import_ref_data(self)
    def remove_selected_line(self):
        return iad_controls.remove_selected_line(self)
    def change_selected_line_color(self):
        return iad_controls.change_selected_line_color(self)
    def get_selected_line_name(self):
        return iad_controls.get_selected_line_name(self)
    def update_line_list(self):
        return iad_controls.update_line_list(self)
    def add_line_to_list(self, line_name, line_color):
        return iad_controls.add_line_to_list(self, line_name, line_color)
    def toggle_line_visibility(self, line_name, checkbox_var):
        return iad_controls.toggle_line_visibility(self, line_name, checkbox_var)
    def plot_integrated_diff(self):
        return iad_controls.plot_integrated_diff(self)
    def calculate_and_display_satellite_peak_iad(self):
        return iad_controls.calculate_and_display_satellite_peak_iad(self)
    def pick_ref_color(self):
        return iad_controls.pick_ref_color(self)
    def toggle_background_removal(self):
        return spectrum_controls.toggle_background_removal(self)
    def plot_roi_spectrum(self):
        return spectrum_controls.plot_roi_spectrum(self)
    def apply_moving_average(self):
        return spectrum_controls.apply_moving_average(self)
    def open_color_picker(self):
        return spectrum_controls.open_color_picker(self)
    def update_plot(self):
        return spectrum_controls.update_plot(self)
    def save_spectrum_data(self):
        return spectrum_controls.save_spectrum_data(self)
    def save_spectrum_image(self):
        return spectrum_controls.save_spectrum_image(self)
    def save_smoothed_spectrum_data(self):
        return spectrum_controls.save_smoothed_spectrum_data(self)

def run_gui():
    root = tk.Tk()
    app = TIFFAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    run_gui()  
