# -*- coding: utf-8 -*-
import os
import sys
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk, filedialog, simpledialog
from tkinter.colorchooser import askcolor
import csv
import re
import random
import numpy as np
from PIL import Image
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from skimage.filters import threshold_otsu
import matplotlib as mpl
from matplotlib import rcParams

if __package__:
    from .remove_cross import delete_zero_rows_and_columns
    from .spectrum_utils import SpectrumProcessor
    from . import image_processing
    from . import iad_controls
    from . import calibration_controls
    from . import project_io
    from . import spectrum_controls
else:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(module_dir)
    for import_path in (module_dir, parent_dir):
        if import_path and import_path not in sys.path:
            sys.path.insert(0, import_path)
    try:
        from remove_cross import delete_zero_rows_and_columns
        from spectrum_utils import SpectrumProcessor
        import image_processing
        import iad_controls
        import calibration_controls
        import project_io
        import spectrum_controls
    except ModuleNotFoundError:
        from IXE.remove_cross import delete_zero_rows_and_columns
        from IXE.spectrum_utils import SpectrumProcessor
        from IXE import image_processing
        from IXE import iad_controls
        from IXE import calibration_controls
        from IXE import project_io
        from IXE import spectrum_controls


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


class DualRangeSlider(tk.Canvas):
    """Four-handle slider for selecting two row intervals on one track."""

    def __init__(
        self,
        parent,
        first_left_var,
        first_right_var,
        second_left_var,
        second_right_var,
        min_value=0,
        max_value=1,
        width=320,
        height=62,
        command=None,
        second_enabled=True,
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
        self.first_left_var = first_left_var
        self.first_right_var = first_right_var
        self.second_left_var = second_left_var
        self.second_right_var = second_right_var
        self.min_value = min_value
        self.max_value = max(max_value, min_value)
        self.slider_width = width
        self.slider_height = height
        self.command = command
        self.second_enabled = bool(second_enabled)
        self.track_left = 22
        self.track_right = width - 22
        self.track_y = 32
        self.handle_radius = 7
        self._active_handle = None
        self._hover_handle = None
        self._handle_vars = {
            "first_left": self.first_left_var,
            "first_right": self.first_right_var,
            "second_left": self.second_left_var,
            "second_right": self.second_right_var,
        }
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Motion>", self._on_motion)
        self.bind("<Configure>", self._on_configure)
        self.bind("<Leave>", self._on_leave)
        self.set_limits(
            self.min_value,
            self.max_value,
            self.first_left_var.get(),
            self.first_right_var.get(),
            self.second_left_var.get(),
            self.second_right_var.get(),
            invoke=False,
        )

    def set_second_enabled(self, enabled):
        self.second_enabled = bool(enabled)
        if self._active_handle and self._active_handle.startswith("second_") and not self.second_enabled:
            self._active_handle = None
        if self._hover_handle and self._hover_handle.startswith("second_") and not self.second_enabled:
            self._hover_handle = None
        self._draw()

    def set_limits(
        self,
        min_value,
        max_value,
        first_left=None,
        first_right=None,
        second_left=None,
        second_right=None,
        invoke=True,
    ):
        self.min_value = int(min_value)
        self.max_value = max(int(max_value), self.min_value)
        values = (
            self.first_left_var.get() if first_left is None else first_left,
            self.first_right_var.get() if first_right is None else first_right,
            self.second_left_var.get() if second_left is None else second_left,
            self.second_right_var.get() if second_right is None else second_right,
        )
        self._set_values(*values, invoke=invoke)

    def _value_span(self):
        return max(self.max_value - self.min_value, 1)

    def _value_to_x(self, value):
        if self.max_value <= self.min_value:
            return self.track_left
        fraction = (int(value) - self.min_value) / self._value_span()
        return self.track_left + fraction * (self.track_right - self.track_left)

    def _x_to_value(self, x_pos):
        if self.max_value <= self.min_value:
            return self.min_value
        x_pos = min(max(x_pos, self.track_left), self.track_right)
        fraction = (x_pos - self.track_left) / (self.track_right - self.track_left)
        return int(round(self.min_value + fraction * self._value_span()))

    def _clamp(self, value):
        return min(max(int(value), self.min_value), self.max_value)

    def _set_values(self, first_left, first_right, second_left, second_right, invoke=True):
        first_left = self._clamp(first_left)
        first_right = self._clamp(first_right)
        second_left = self._clamp(second_left)
        second_right = self._clamp(second_right)
        self.first_left_var.set(first_left)
        self.first_right_var.set(first_right)
        self.second_left_var.set(second_left)
        self.second_right_var.set(second_right)
        self._draw()
        if invoke and self.command:
            self.command(first_left, first_right, second_left, second_right)

    def _sorted_pair(self, left_var, right_var):
        left, right = sorted((int(left_var.get()), int(right_var.get())))
        return left, right

    def _draw(self):
        self.delete("all")
        self.create_line(self.track_left, self.track_y, self.track_right, self.track_y, fill="#9da7a1", width=2)

        first_left, first_right = self._sorted_pair(self.first_left_var, self.first_right_var)
        first_left_x = self._value_to_x(first_left)
        first_right_x = self._value_to_x(first_right)
        self.create_line(first_left_x, self.track_y - 4, first_right_x, self.track_y - 4, fill="#7b3294", width=4)
        if self.second_enabled:
            second_left, second_right = self._sorted_pair(self.second_left_var, self.second_right_var)
            second_left_x = self._value_to_x(second_left)
            second_right_x = self._value_to_x(second_right)
            self.create_line(second_left_x, self.track_y + 4, second_right_x, self.track_y + 4, fill="#008837", width=4)

        handles = [
            ("first_left", "#7b3294"),
            ("first_right", "#7b3294"),
        ]
        if self.second_enabled:
            handles.extend([
                ("second_left", "#008837"),
                ("second_right", "#008837"),
            ])
        for handle_name, fill_color in handles:
            self._draw_handle(self._value_to_x(self._handle_vars[handle_name].get()), fill_color)
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
        value = self._handle_vars[handle_name].get()
        x_pos = self._value_to_x(value)
        text_id = self.create_text(
            x_pos,
            max(self.track_y - 22, 10),
            text=str(value),
            fill="#1f2d1f",
            font=("Arial", 10, "bold"),
        )
        x0, y0, x1, y1 = self.bbox(text_id)
        bg_id = self.create_rectangle(
            x0 - 5,
            y0 - 2,
            x1 + 5,
            y1 + 2,
            fill="#f8fbf8",
            outline="#9da7a1",
            width=1,
        )
        self.tag_lower(bg_id, text_id)

    def _nearest_handle(self, x_pos):
        names = [name for name in self._handle_vars if self.second_enabled or not name.startswith("second_")]
        return min(
            names,
            key=lambda name: abs(x_pos - self._value_to_x(self._handle_vars[name].get())),
        )

    def _handle_under_cursor(self, x_pos, y_pos):
        if abs(y_pos - self.track_y) > self.handle_radius + 8:
            return None
        nearest_handle = self._nearest_handle(x_pos)
        if abs(x_pos - self._value_to_x(self._handle_vars[nearest_handle].get())) <= self.handle_radius + 6:
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
        self.slider_height = max(event.height, 42)
        self.track_left = 22
        self.track_right = max(self.slider_width - 22, self.track_left)
        self.track_y = max(self.slider_height // 2, self.handle_radius + 8)
        self._draw()

    def _on_leave(self, _event):
        if self._active_handle:
            return
        if self._hover_handle is not None:
            self._hover_handle = None
            self._draw()

    def _update_active_handle(self, x_pos):
        if self._active_handle is None:
            return
        if self._active_handle.startswith("second_") and not self.second_enabled:
            return
        self._handle_vars[self._active_handle].set(self._x_to_value(x_pos))
        self._set_values(
            self.first_left_var.get(),
            self.first_right_var.get(),
            self.second_left_var.get(),
            self.second_right_var.get(),
        )


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
            'cmap': 'viridis',
            'row_begin': 0,
            'row_end': 1,
            'row2_begin': 0,
            'row2_end': 1,
            'roi_second_enabled': False,
            'column_begin': 0,
            'column_end': 1,
            'bg_row_begin': 0,
            'bg_row_end': 0,
            'bg2_row_begin': 0,
            'bg2_row_end': 0,
            'bg_col_begin': 320,
            'bg_col_end': 618, 
            'n_moveavg': 5,
            'PK intersect': {'i_l': 50, 'i_r': 120}, # Default range for satellite peak intersection
            'eye_ball_cross': None,
            'peak_fit': {'n_peaks': 3, 'x_min': None, 'x_max': None, 'peak_shape': 'pseudo_voigt'},
            'ccd_gap': {
                'enabled': True,
                'max_width': 8,
                'drop_fraction': 0.65,
                'mask_max_width': 16,
                'mask_drop_fraction': 0.55,
                'mask_dilate': 1,
                'mask_center_window': 0.35,
                'edge_valid_fraction': 0.80,
            },
            'trace_extraction': {'enabled': False, 'search_margin': 60},
        }
        
        # Initialize state variables
        #self.current_image = None
        self.bg_subtraction_enabled = False
        self.gap_correction_enabled = self.parm['ccd_gap']['enabled']
        self.trace_extraction_enabled = self.parm['trace_extraction']['enabled']
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
        self.display_panel.grid_rowconfigure(0, weight=1)
        self.display_panel.grid_columnconfigure(0, weight=0, minsize=700)
        self.display_panel.grid_columnconfigure(1, weight=1)
        
        # Left side for images: fixed CCD panel, right side stretches like a plot scene.
        self.image_panel = ttk.Frame(self.display_panel, width=700, style="White.TFrame")
        self.image_panel.grid(row=0, column=0, sticky="ns", padx=0, pady=0)
        self.image_panel.pack_propagate(False)
        self.image_panel.grid_propagate(False)
        # Right side for spectrum
        self.spectrum_panel = ttk.Frame(self.display_panel)
        self.spectrum_panel.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)


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
        # Calibration tab
        self.calibration_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.calibration_tab, text="Calibration")

        # Setup controls for each tab
        self.setup_processing_controls()
        self.setup_spectrum_controls()
        self.setup_iad_controls()
        self.setup_calibration_controls()

    def setup_processing_controls(self):
        """Image processing controls"""
        frame = ttk.Frame(self.processing_tab, padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=1)

        file_row = ttk.Frame(frame)
        file_row.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        file_row.grid_columnconfigure(3, weight=1)
        ttk.Button(file_row, text="Import TIFF", width=13, command=self.load_tiff).grid(row=0, column=0, padx=(0, 8), pady=3)
        ttk.Button(file_row, text="Import TIFF Stack", width=17, command=self.load_tiff_stack).grid(row=0, column=1, padx=(0, 8), pady=3)
        ttk.Label(file_row, text="File Path:").grid(row=0, column=2, padx=(0, 6), pady=3, sticky="w")
        self.file_path_entry = ttk.Entry(frame, width=150, font=('Arial', 12))
        self.file_path_entry.insert(0, " ")  # Set the default placeholder tex
        self.file_path_entry.grid(in_=file_row, row=0, column=3, padx=(0, 0), pady=3, sticky="ew")

        display_row = ttk.Frame(frame)
        display_row.grid(row=1, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Label(display_row, text="vmin:").grid(row=0, column=0, padx=(0, 6), pady=4)
        self.vmin_entry = ttk.Entry(frame, width=4, font=('Arial', 12))
        self.vmin_entry.insert(0, str(self.parm['vmin']))
        self.vmin_entry.grid(in_=display_row, row=0, column=1, padx=(0, 14), pady=4)
        
        ttk.Label(display_row, text="vmax:").grid(row=0, column=2, padx=(0, 6), pady=4)
        self.vmax_entry = ttk.Entry(frame, width=4, font=('Arial', 12))
        self.vmax_entry.insert(0, str(self.parm['vmax']))
        self.vmax_entry.grid(in_=display_row, row=0, column=3, padx=(0, 14), pady=4)

        ttk.Label(display_row, text="Cmap:").grid(row=0, column=4, padx=(0, 6), pady=4)
        self.image_cmap = ttk.Combobox(
            frame,
            values=["viridis", "plasma", "inferno", "magma", "cividis", "gray", "Greys", "turbo"],
            width=10,
            state="readonly",
        )
        self.image_cmap.set(self.parm.get('cmap', 'viridis'))
        self.image_cmap.grid(in_=display_row, row=0, column=5, padx=(0, 16), pady=4)
        self.image_cmap.bind("<<ComboboxSelected>>", lambda event: self.update_image_cmap())

        ttk.Label(display_row, text="Auto Tilt:").grid(row=0, column=6, padx=(0, 6), pady=4)
        self.tilt_value = tk.StringVar(value=f"{self.parm['tilt_cor']:.2f}")
        self.tilt_entry = ttk.Entry(frame, width=7, font=('Arial', 12), textvariable=self.tilt_value, state='readonly')
        self.tilt_entry.grid(in_=display_row, row=0, column=7, padx=(0, 0), pady=4)
        
        button_row = ttk.Frame(frame)
        button_row.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.process_button = RoundedButton(
            button_row,
            text="Process>",
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
        self.process_button.grid(row=0, column=0, padx=(0, 8), pady=3)
        ttk.Button(button_row, text="Gap Mask", width=10, command=self.show_gap_mask).grid(row=0, column=1, padx=(0, 8), pady=3)
        ttk.Button(button_row, text="Save Image", width=11, command=self.save_processed_image).grid(row=0, column=2, padx=(0, 8), pady=3)
        ttk.Button(button_row, text="Open Project", width=13, command=self.open_project).grid(row=0, column=3, padx=(0, 8), pady=3)
        ttk.Button(button_row, text="Save Project", width=13, command=self.save_project).grid(row=0, column=4, padx=(0, 0), pady=3)

    def setup_spectrum_controls(self):
        """Spectrum analysis controls"""
        frame = ttk.Frame(self.spectrum_tab)
        frame.pack(fill=tk.X, padx=5, pady=5)
        frame.grid_columnconfigure(0, weight=1)

        self.spectrum_control_notebook = ttk.Notebook(frame)
        self.spectrum_control_notebook.grid(row=0, column=0, sticky="ew", padx=0, pady=0)

        plotting_panel = ttk.Frame(self.spectrum_control_notebook, padding=(8, 6))
        self.spectrum_control_notebook.add(plotting_panel, text="Plotting")
        plotting_panel.grid_columnconfigure(0, weight=0)
        plotting_panel.grid_columnconfigure(1, weight=1)
        plotting_panel.grid_columnconfigure(2, weight=0)

        fitting_panel = ttk.Frame(self.spectrum_control_notebook, padding=(8, 6))
        self.spectrum_control_notebook.add(fitting_panel, text="Fitting")
        fitting_panel.grid_columnconfigure(0, weight=1)

        label_grid = {'padx': (0, 6), 'pady': 1, 'sticky': 'e'}
        slider_grid = {'padx': (2, 6), 'pady': 1, 'sticky': 'ew'}
        action_grid = {'padx': (2, 4), 'pady': 1, 'sticky': 'w'}
        ttk.Label(plotting_panel, text="ROI Rows:", width=12, anchor="e").grid(row=0, column=0, **label_grid)
        self.row_begin = tk.IntVar(value=self.parm['row_begin'])
        self.row_end = tk.IntVar(value=self.parm['row_end'])
        self.row2_begin = tk.IntVar(value=self.parm.get('row2_begin', self.parm['row_begin']))
        self.row2_end = tk.IntVar(value=self.parm.get('row2_end', self.parm['row_end']))
        self.roi_second_enabled = bool(self.parm.get('roi_second_enabled', False))
        self.row_slider = DualRangeSlider(
            plotting_panel,
            first_left_var=self.row_begin,
            first_right_var=self.row_end,
            second_left_var=self.row2_begin,
            second_right_var=self.row2_end,
            min_value=0,
            max_value=max(
                self.parm['row_end'],
                self.parm['row_begin'] + 1,
                self.parm.get('row2_end', 0),
                self.parm.get('row2_begin', 0) + 1,
            ),
            width=300,
            height=50,
            command=self.on_roi_rows_changed,
            second_enabled=self.roi_second_enabled,
        )
        self.row_slider.grid(row=0, column=1, **slider_grid)
        self.on_roi_rows_changed(
            self.row_begin.get(),
            self.row_end.get(),
            self.row2_begin.get(),
            self.row2_end.get(),
            redraw=False,
        )
        self.roi_button_frame = ttk.Frame(plotting_panel)
        self.roi_button_frame.grid(row=0, column=2, **action_grid)
        self.roi_add_button = ttk.Button(self.roi_button_frame, text="+", width=2, command=self.add_roi_row_range)
        self.roi_add_button.grid(row=0, column=0, padx=(0, 2), pady=0)
        self.roi_remove_button = ttk.Button(self.roi_button_frame, text="-", width=2, command=self.remove_roi_row_range)
        self.roi_remove_button.grid(row=0, column=1, padx=(0, 4), pady=0)

        ttk.Label(plotting_panel, text="ROI Columns:", width=12, anchor="e").grid(row=1, column=0, **label_grid)
        self.column_begin = tk.IntVar(value=self.parm['column_begin'])
        self.column_end = tk.IntVar(value=self.parm['column_end'])
        self.column_slider = RangeSlider(
            plotting_panel,
            left_var=self.column_begin,
            right_var=self.column_end,
            min_value=0,
            max_value=max(self.parm['column_end'], self.parm['column_begin'] + 1),
            width=300,
            height=44,
            command=self.on_roi_cols_changed,
        )
        self.column_slider.grid(row=1, column=1, **slider_grid)
        self.on_roi_cols_changed(self.column_begin.get(), self.column_end.get(), redraw=False)
        self.plot_button = ttk.Button(
            self.roi_button_frame,
            text="Plot",
            width=10,
            style='Active.TButton',
            command=self.plot_roi_spectrum,
        )
        self.plot_button.grid(row=0, column=2, padx=(4, 0), pady=0, sticky="w")
        self._sync_roi_row_buttons()

        self.gap_toggle = ttk.Button(plotting_panel, text="Gap Mask (ON)", command=self.toggle_gap_correction)
        self.gap_toggle.grid(row=1, column=2, **action_grid)
        self.gap_toggle.config(style='Active.TButton' if self.gap_correction_enabled else 'TButton')
        
        ttk.Label(plotting_panel, text="BG Rows:", width=12, anchor="e").grid(row=2, column=0, **label_grid)
        self.bg_row_begin = tk.IntVar(value=self.parm['bg_row_begin'])
        self.bg_row_end = tk.IntVar(value=self.parm['bg_row_end'])
        self.bg2_row_begin = tk.IntVar(value=self.parm.get('bg2_row_begin', self.parm['bg_row_begin']))
        self.bg2_row_end = tk.IntVar(value=self.parm.get('bg2_row_end', self.parm['bg_row_end']))
        self.bg_row_slider = DualRangeSlider(
            plotting_panel,
            first_left_var=self.bg_row_begin,
            first_right_var=self.bg_row_end,
            second_left_var=self.bg2_row_begin,
            second_right_var=self.bg2_row_end,
            min_value=0,
            max_value=max(
                self.parm['bg_row_end'],
                self.parm['bg_row_begin'] + 1,
                self.parm.get('bg2_row_end', 0),
                self.parm.get('bg2_row_begin', 0) + 1,
            ),
            width=300,
            height=50,
            command=self.on_bg_rows_changed,
        )
        self.bg_row_slider.grid(row=2, column=1, **slider_grid)
        self.on_bg_rows_changed(
            self.bg_row_begin.get(),
            self.bg_row_end.get(),
            self.bg2_row_begin.get(),
            self.bg2_row_end.get(),
            redraw=False,
        )

        self.bg_button_frame = ttk.Frame(plotting_panel)
        self.bg_button_frame.grid(row=2, column=2, **action_grid)
        self.auto_bg_button = ttk.Button(self.bg_button_frame, text="Auto BG", command=self.auto_select_background_rows)
        self.auto_bg_button.grid(row=0, column=0, padx=(0, 4), pady=0, sticky="w")
        self.bg_toggle = ttk.Button(self.bg_button_frame, text="BG Remove", command=self.toggle_background_removal)
        self.bg_toggle.grid(row=0, column=1, padx=0, pady=0, sticky="w")
        self.line_color = tk.StringVar(value="black")

        self.line_controls_row = ttk.Frame(plotting_panel)
        self.line_controls_row.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(0, 0))
        for column in range(7):
            self.line_controls_row.grid_columnconfigure(column, weight=1)

        ttk.Button(self.line_controls_row, text="Pick Color", command=self.open_color_picker).grid(row=0, column=0, padx=6, pady=2)
        ttk.Label(self.line_controls_row, text="Line Style:").grid(row=0, column=1, padx=6, pady=2, sticky="e")
        self.line_style = ttk.Combobox(self.line_controls_row, values=["-", "--", "-.", ":"], width=4)
        self.line_style.set("-")  # Default line style
        self.line_style.grid(row=0, column=2, padx=6, pady=2, sticky="w")
        self.line_style.bind("<<ComboboxSelected>>", lambda event: self.update_plot())
        ttk.Label(self.line_controls_row, text="Line Width:").grid(row=0, column=3, padx=6, pady=2, sticky="e")
        self.line_width = ttk.Entry(self.line_controls_row, width=5)
        self.line_width.insert(0, "0.5")  # Default line width
        self.line_width.grid(row=0, column=4, padx=6, pady=2, sticky="w")
        self.line_width.bind("<KeyRelease>", lambda event: self.update_plot())
        ttk.Button(self.line_controls_row, text="Save Spectrum", command=self.save_spectrum_data).grid(row=0, column=5, padx=6, pady=2)
        ttk.Button(self.line_controls_row, text="Save Image", command=self.save_spectrum_image).grid(row=0, column=6, padx=6, pady=2)

        peak_defaults = self.parm.get('peak_fit', {})
        self.peak_fit_model_var = tk.StringVar(value=peak_defaults.get('peak_shape', 'pseudo_voigt'))
        self.fitting_controls_row = ttk.Frame(fitting_panel)
        self.fitting_controls_row.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        for column in range(10):
            self.fitting_controls_row.grid_columnconfigure(column, weight=0)
        self.fitting_controls_row.grid_columnconfigure(9, weight=1)

        ttk.Label(self.fitting_controls_row, text="Fit Model:").grid(row=0, column=0, padx=6, pady=2, sticky="e")
        ttk.Radiobutton(
            self.fitting_controls_row,
            text="Pseudo-Voigt",
            value="pseudo_voigt",
            variable=self.peak_fit_model_var,
            command=lambda: self.parm.setdefault('peak_fit', {}).__setitem__('peak_shape', self.peak_fit_model_var.get()),
        ).grid(row=0, column=1, padx=6, pady=2, sticky="w")
        ttk.Radiobutton(
            self.fitting_controls_row,
            text="Lorentzian",
            value="lorentzian",
            variable=self.peak_fit_model_var,
            command=lambda: self.parm.setdefault('peak_fit', {}).__setitem__('peak_shape', self.peak_fit_model_var.get()),
        ).grid(row=0, column=2, padx=6, pady=2, sticky="w")

        ttk.Label(self.fitting_controls_row, text="Peaks:").grid(row=0, column=3, padx=(16, 4), pady=2, sticky="e")
        self.peak_fit_count = ttk.Entry(self.fitting_controls_row, width=4)
        self.peak_fit_count.insert(0, str(peak_defaults.get('n_peaks', 3)))
        self.peak_fit_count.grid(row=0, column=4, padx=(0, 6), pady=2, sticky="w")

        ttk.Label(self.fitting_controls_row, text="Fit Range:").grid(row=0, column=5, padx=(12, 4), pady=2, sticky="e")
        self.peak_fit_start = ttk.Entry(self.fitting_controls_row, width=8)
        if peak_defaults.get('x_min') is not None:
            self.peak_fit_start.insert(0, str(peak_defaults.get('x_min')))
        self.peak_fit_start.grid(row=0, column=6, padx=(0, 2), pady=2, sticky="w")
        ttk.Label(self.fitting_controls_row, text="-").grid(row=0, column=7, padx=2, pady=2)
        self.peak_fit_end = ttk.Entry(self.fitting_controls_row, width=8)
        if peak_defaults.get('x_max') is not None:
            self.peak_fit_end.insert(0, str(peak_defaults.get('x_max')))
        self.peak_fit_end.grid(row=0, column=8, padx=(2, 6), pady=2, sticky="w")

        self.fitting_actions_row = ttk.Frame(fitting_panel)
        self.fitting_actions_row.grid(row=1, column=0, sticky="w", pady=(0, 2))
        ttk.Button(self.fitting_actions_row, text="Peak Fit", command=self.show_peak_fit_profile).grid(row=0, column=0, padx=6, pady=2)
        ttk.Button(self.fitting_actions_row, text="Save pkfit", command=self.save_peak_fit_data).grid(row=0, column=1, padx=6, pady=2)
        ttk.Button(
            self.fitting_actions_row,
            text="Save Params",
            command=self.save_peak_fit_parameters,
        ).grid(row=0, column=2, padx=6, pady=2)

        self.peak_fit_results_frame = ttk.Frame(fitting_panel)
        self.peak_fit_results_frame.grid(row=2, column=0, sticky="ew", pady=(2, 0))
        self.peak_fit_results_frame.grid_columnconfigure(0, weight=1)
        self.peak_fit_summary_var = tk.StringVar(value="Peak fit parameters: not fitted")
        ttk.Label(self.peak_fit_results_frame, textvariable=self.peak_fit_summary_var).grid(
            row=0,
            column=0,
            sticky="w",
            padx=5,
            pady=(0, 2),
        )
        self.peak_fit_table = ttk.Treeview(
            self.peak_fit_results_frame,
            columns=("peak", "model", "lfrac", "position", "width", "area"),
            show="headings",
            height=3,
        )
        headings = (
            ("peak", "Peak", 115),
            ("model", "Model", 120),
            ("lfrac", "L frac", 75),
            ("position", "Position", 90),
            ("width", "Width", 90),
            ("area", "Area", 90),
        )
        for column_id, heading, width in headings:
            self.peak_fit_table.heading(column_id, text=heading)
            self.peak_fit_table.column(column_id, width=width, anchor="center", stretch=True)
        self.peak_fit_table.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 2))

    def setup_calibration_controls(self):
        """Calibration controls."""
        frame = ttk.Frame(self.calibration_tab, padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)
        for column in range(10):
            frame.grid_columnconfigure(column, weight=0)
        frame.grid_columnconfigure(9, weight=1)

        ttk.Button(frame, text="Import Cal.", command=self.import_calibration).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Button(frame, text="Fit Cal.", command=self.fit_calibration_spectrum).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(frame, text="Calc Map", command=self.calculate_energy_calibration).grid(row=0, column=2, padx=5, pady=5, sticky="w")
        ttk.Button(frame, text="Apply Cal.", command=self.apply_energy_calibration).grid(row=0, column=3, padx=5, pady=5, sticky="w")
        ttk.Button(frame, text="Show Cal. Fit", command=self.show_calibration_fit).grid(row=0, column=4, padx=5, pady=5, sticky="w")
        ttk.Button(frame, text="Compare Cal.", command=self.compare_calibration_overlay).grid(row=0, column=5, padx=5, pady=5, sticky="w")
        ttk.Button(frame, text="Save Cal.", command=self.save_calibrated_spectrum_data).grid(row=0, column=6, padx=5, pady=5, sticky="w")
        self.calibration_file_label = ttk.Label(frame, text="No calibration file loaded")
        self.calibration_file_label.grid(row=0, column=7, columnspan=3, padx=8, pady=5, sticky="w")

        fit_range = ttk.Frame(frame)
        fit_range.grid(row=1, column=0, columnspan=8, sticky="w", padx=5, pady=(2, 5))
        ttk.Label(fit_range, text="Cal. fit range:").grid(row=0, column=0, padx=(0, 5), pady=2)
        self.cal_fit_start = ttk.Entry(fit_range, width=8)
        self.cal_fit_start.grid(row=0, column=1, padx=(0, 4), pady=2)
        ttk.Label(fit_range, text="-").grid(row=0, column=2, padx=(0, 4), pady=2)
        self.cal_fit_end = ttk.Entry(fit_range, width=8)
        self.cal_fit_end.grid(row=0, column=3, padx=(0, 12), pady=2)
        ttk.Label(fit_range, text="Use Kβ' + Kβ1,3 anchors; Kβres is residual check").grid(row=0, column=4, padx=(0, 0), pady=2)

        self.cal_sat_pixel_var = tk.StringVar()
        self.cal_res_pixel_var = tk.StringVar()
        self.cal_main_pixel_var = tk.StringVar()
        self.cal_sat_energy_var = tk.StringVar()
        self.cal_res_energy_var = tk.StringVar()
        self.cal_main_energy_var = tk.StringVar()
        self.cal_slope_var = tk.StringVar()
        self.cal_intercept_var = tk.StringVar()
        self.cal_residual_var = tk.StringVar()
        self.cal_status_var = tk.StringVar(value="Import a calibrant spectrum to begin.")

        output = ttk.Frame(frame)
        output.grid(row=2, column=0, columnspan=8, sticky="ew", padx=5, pady=(0, 4))
        labels = [
            ("Current px Kβ'", self.cal_sat_pixel_var),
            ("Current px Kβres", self.cal_res_pixel_var),
            ("Current px Kβ1,3", self.cal_main_pixel_var),
            ("Cal E Kβ'", self.cal_sat_energy_var),
            ("Cal E Kβres", self.cal_res_energy_var),
            ("Cal E Kβ1,3", self.cal_main_energy_var),
        ]
        for index, (label, var) in enumerate(labels):
            ttk.Label(output, text=label).grid(row=index // 3, column=(index % 3) * 2, padx=(0, 5), pady=2, sticky="e")
            ttk.Entry(output, width=10, textvariable=var, state="readonly").grid(row=index // 3, column=(index % 3) * 2 + 1, padx=(0, 12), pady=2, sticky="w")

        equation = ttk.Frame(frame)
        equation.grid(row=3, column=0, columnspan=8, sticky="w", padx=5, pady=(0, 2))
        ttk.Label(equation, text="E =").grid(row=0, column=0, padx=(0, 4), pady=2)
        ttk.Entry(equation, width=12, textvariable=self.cal_slope_var, state="readonly").grid(row=0, column=1, padx=(0, 4), pady=2)
        ttk.Label(equation, text="* pixel +").grid(row=0, column=2, padx=(0, 4), pady=2)
        ttk.Entry(equation, width=12, textvariable=self.cal_intercept_var, state="readonly").grid(row=0, column=3, padx=(0, 12), pady=2)
        ttk.Label(equation, text="Kβres residual:").grid(row=0, column=4, padx=(0, 4), pady=2)
        ttk.Entry(equation, width=10, textvariable=self.cal_residual_var, state="readonly").grid(row=0, column=5, padx=(0, 4), pady=2)
        ttk.Label(equation, text="energy").grid(row=0, column=6, padx=(0, 4), pady=2)

        ttk.Label(frame, textvariable=self.cal_status_var).grid(row=4, column=0, columnspan=8, padx=5, pady=(2, 0), sticky="w")

    def setup_iad_controls(self):
        """IAD Calculation controls"""
        frame = ttk.Frame(self.iad_tab, padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create toolbar for IAD controls
        self.iad_toolbar = ttk.Frame(frame)
        self.iad_toolbar.grid(row=0, column=0, sticky='w', padx=5, pady=5)  # First row for toolbar buttons

        ttk.Button(self.iad_toolbar, text="Import Ref.", command=self.import_ref_data).grid(row=0, column=0, padx=5)
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
        self.image_fig_dpi = 300
        self.image_point_scale = self.base_display_dpi / self.image_fig_dpi
        self.spectrum_initial_width_px = 600
        self.spectrum_initial_height_px = 400
        self.spectrum_base_dpi = self.base_display_dpi
        self.spectrum_fig_dpi = 300
        self.spectrum_point_scale = self.spectrum_base_dpi / self.spectrum_fig_dpi
        self.spectrum_subplot_margins_px = {'left': 45, 'bottom': 24, 'right': 10, 'top': 10}
        
        # Single frame for images (raw or processed)
        self.image_frame = ttk.LabelFrame(
            self.image_panel,
            text="Image Display",
            padding=(10, 10),
            width=430,
            height=350,
            style="White.TLabelframe",
        )
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.image_frame.pack_propagate(False)
        self.run_number_label = ttk.Label(self.image_frame, text="Run: Not Loaded", font=("Arial", 16))
        self.image_title_label = ttk.Label(
            self.image_frame,
            text="No image loaded",
            font=("Arial", 13),
            style="White.TLabel",
        )
        self.image_title_label.pack(side=tk.TOP, padx=5, pady=(0, 3))

        self.image_plot_frame = ttk.Frame(self.image_frame, style="White.TFrame")
        self.image_plot_frame.pack(fill=tk.BOTH, expand=True)
        self.image_plot_frame.columnconfigure(1, weight=1)
        self.image_plot_frame.rowconfigure(0, weight=1)

        self.image_ylabel_canvas = tk.Canvas(
            self.image_plot_frame,
            width=32,
            bg='white',
            highlightthickness=0,
            bd=0,
        )
        self.image_ylabel_canvas.grid(row=0, column=0, sticky="ns")
        self.image_ylabel_canvas.bind("<Configure>", self.update_image_ylabel)

        # Smaller figure size for image display
        self.fig_image = plt.Figure(figsize=(420 / self.image_fig_dpi, 320 / self.image_fig_dpi), dpi=self.image_fig_dpi)
        self.fig_image.patch.set_facecolor('white')
        self.fig_image.subplots_adjust(left=0.10, bottom=0.10, right=0.93, top=0.97)
        self.ax_image = self.fig_image.add_subplot(111)
        self.ax_image.set_facecolor('white')
        self.cbar = None
        # Hide ticks and ticklabels for image axes initially
        self.ax_image.set_xticks([])
        self.ax_image.set_yticks([])
        self.ax_image.set_xticklabels([])
        self.ax_image.set_yticklabels([])
        for spine in self.ax_image.spines.values():
            spine.set_linewidth(0.5)
        self.canvas_image = FigureCanvasTkAgg(self.fig_image, master=self.image_plot_frame)
        self.canvas_image.get_tk_widget().configure(bg='white', highlightthickness=0, bd=0)
        self.canvas_image.get_tk_widget().grid(row=0, column=1, sticky="nsew")

        self.image_xlabel_label = ttk.Label(
            self.image_frame,
            text="Columns",
            font=("Arial", 13),
            style="White.TLabel",
        )
        self.image_xlabel_label.pack(side=tk.BOTTOM, pady=(0, 0))
        
        # Spectrum frame (right side) - larger
        spectrum_frame = ttk.LabelFrame(self.spectrum_panel, text="Spectrum", padding=(10, 10))
        spectrum_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        spectrum_frame.grid_columnconfigure(0, weight=1)
        spectrum_frame.grid_rowconfigure(1, weight=1, minsize=120)
        
        spectrum_figsize = (
            self.spectrum_initial_width_px / self.spectrum_fig_dpi,
            self.spectrum_initial_height_px / self.spectrum_fig_dpi,
        )
        self.fig_spectrum = plt.Figure(figsize=spectrum_figsize, dpi=self.spectrum_fig_dpi)
        self.fig_spectrum.patch.set_facecolor('white')
        self.fig_spectrum.subplots_adjust(left=0.075, bottom=0.08, right=0.985, top=0.975)
        self.ax_spectrum = self.fig_spectrum.add_subplot(111)
        self.ax_spectrum.set_facecolor('white')
        # Hide ticks and ticklabels for spectrum axes initially
        self.ax_spectrum.set_xticks([])
        self.ax_spectrum.set_yticks([])
        self.ax_spectrum.set_xticklabels([])
        self.ax_spectrum.set_yticklabels([])
        for spine in self.ax_spectrum.spines.values():
            spine.set_linewidth(0.5)
        # Add run number and compact spectrum-state labels to the display.
        self.spectrum_header = ttk.Frame(spectrum_frame)
        self.spectrum_header.grid(row=0, column=0, sticky="ew", padx=5, pady=(0, 2))
        self.spectrum_header.grid_columnconfigure(0, weight=0)
        self.spectrum_header.grid_columnconfigure(1, weight=1)
        self.run_number_label_spectrum = ttk.Label(self.spectrum_header, text="Run: Not Loaded", font=("Arial", 12))
        self.run_number_label_spectrum.grid(row=0, column=0, sticky="w")
        self.spectrum_status_label = ttk.Label(self.spectrum_header, text="No spectrum plotted", font=("Arial", 11))
        self.spectrum_status_label.grid(row=0, column=1, sticky="e", padx=(12, 0))

        self.spectrum_plot_body = tk.Frame(spectrum_frame, bg='white', highlightthickness=0, bd=0)
        self.spectrum_plot_body.grid(row=1, column=0, sticky="nsew")
        self.spectrum_plot_body.grid_columnconfigure(0, weight=0)
        self.spectrum_plot_body.grid_columnconfigure(1, weight=1)
        self.spectrum_plot_body.grid_rowconfigure(0, weight=1)
        self.spectrum_plot_body.grid_rowconfigure(1, weight=0)

        self.spectrum_ylabel_text = "Normalized intensity"
        self.spectrum_ylabel_canvas = tk.Canvas(
            self.spectrum_plot_body,
            width=34,
            bg='white',
            highlightthickness=0,
            bd=0,
        )
        self.spectrum_ylabel_canvas.grid(row=0, column=0, sticky="ns")
        self.spectrum_ylabel_canvas.bind("<Configure>", self.update_spectrum_ylabel)

        self.canvas_spectrum = FigureCanvasTkAgg(self.fig_spectrum, master=self.spectrum_plot_body)
        self.canvas_spectrum.get_tk_widget().configure(bg='white', highlightthickness=0, bd=0)
        self.canvas_spectrum.get_tk_widget().grid(row=0, column=1, sticky="nsew")
        self.canvas_spectrum.get_tk_widget().bind("<Configure>", self.on_spectrum_canvas_configure, add="+")

        self.spectrum_xlabel_label = tk.Label(
            self.spectrum_plot_body,
            text="Column Index",
            font=("Arial", 13),
            bg='white',
            anchor='center',
        )
        self.spectrum_xlabel_label.grid(row=1, column=1, sticky="ew", pady=(0, 0))

        self.spectrum_view_bar = ttk.Frame(spectrum_frame)
        self.spectrum_view_bar.grid(row=2, column=0, sticky="e", padx=0, pady=(2, 0))
        self.spectrum_view_buttons = {}
        view_specs = (
            ("original", "Original", 7, self.show_original_spectrum),
            ("corrected", "Gap C.", 6, self.show_gap_corrected_spectrum),
            ("fit", "Peak fit", 7, self.show_peak_fit_view),
            ("calibration", "Calibration", 9, self.show_calibrated_spectrum_view),
        )
        for column, (mode, text, width, command) in enumerate(view_specs):
            button = ttk.Button(self.spectrum_view_bar, text=text, width=width, style="Compact.TButton", command=command)
            button.grid(row=0, column=column, padx=1, pady=0)
            self.spectrum_view_buttons[mode] = button

        # Larger toolbar
        self.spectrum_toolbar_frame = ttk.Frame(spectrum_frame)
        self.spectrum_toolbar_frame.grid(row=3, column=0, sticky="ew", pady=(2, 0))
        self.toolbar = NavigationToolbar2Tk(self.canvas_spectrum, self.spectrum_toolbar_frame, pack_toolbar=False)
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
        style.configure('Compact.TButton', font=('Arial', 12), padding=(2, 1))
        style.configure('CompactActive.TButton', font=('Arial', 12, 'bold'), padding=(2, 1), foreground='#0b5d3b')
        style.configure(
            'Active.TButton',
            font=('Arial', 14, 'bold'),
            foreground='#0b5d3b',
            background='#d9f0df',
            padding=(8, 4),
        )
        style.map(
            'Active.TButton',
            background=[('active', '#c7e8cf'), ('pressed', '#b6ddc0')],
            foreground=[('disabled', '#5e7766'), ('!disabled', '#0b5d3b')],
        )
        style.configure('TLabel', font=('Arial', 14))
        style.configure('TEntry', font=('Arial', 14))
        style.configure('White.TFrame', background='white')
        style.configure('White.TLabel', background='white')
        style.configure('White.TLabelframe', background='white')
        style.configure('White.TLabelframe.Label', background='white')
        
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
            font=("Arial", 13),
        )

    def update_spectrum_ylabel(self, event=None):
        """Render the spectrum y-axis label outside the Matplotlib figure."""
        if not hasattr(self, 'spectrum_ylabel_canvas'):
            return
        canvas = self.spectrum_ylabel_canvas
        canvas.delete("all")
        height = max(canvas.winfo_height(), 1)
        width = max(canvas.winfo_width(), 1)
        canvas.create_text(
            width / 2,
            height / 2,
            text=getattr(self, 'spectrum_ylabel_text', "Normalized intensity"),
            angle=90,
            font=("Arial", 13),
        )

    def on_roi_rows_changed(self, row_begin=None, row_end=None, row2_begin=None, row2_end=None, redraw=True):
        """Sync the selected ROI row ranges with the UI and image overlay."""
        if row_begin is None:
            row_begin = self.row_begin.get()
        if row_end is None:
            row_end = self.row_end.get()
        if row2_begin is None:
            row2_begin = self.row2_begin.get() if hasattr(self, 'row2_begin') else row_begin
        if row2_end is None:
            row2_end = self.row2_end.get() if hasattr(self, 'row2_end') else row_end
        row_begin = int(row_begin)
        row_end = int(row_end)
        row2_begin = int(row2_begin)
        row2_end = int(row2_end)
        row_begin, row_end = sorted((row_begin, row_end))
        row2_begin, row2_end = sorted((row2_begin, row2_end))
        self.parm['row_begin'] = row_begin
        self.parm['row_end'] = row_end
        self.parm['row2_begin'] = row2_begin
        self.parm['row2_end'] = row2_end
        self.parm['roi_second_enabled'] = bool(getattr(self, 'roi_second_enabled', False))
        if hasattr(self, 'row_range_label'):
            row_count = (getattr(self, 'row_slider', None).max_value + 1) if hasattr(self, 'row_slider') else (row_end + 1)
            if getattr(self, 'roi_second_enabled', False):
                self.row_range_label.config(text=f"{row_begin}-{row_end}; {row2_begin}-{row2_end}  ({row_count} rows)")
            else:
                self.row_range_label.config(text=f"{row_begin} - {row_end}  ({row_count} rows)")
        if redraw and hasattr(self, 'current_display_data'):
            self.refresh_image_overlays()

    def _sync_roi_row_buttons(self):
        enabled = bool(getattr(self, 'roi_second_enabled', False))
        if hasattr(self, 'row_slider') and hasattr(self.row_slider, 'set_second_enabled'):
            self.row_slider.set_second_enabled(enabled)
        if hasattr(self, 'roi_add_button'):
            self.roi_add_button.config(state='disabled' if enabled else 'normal')
        if hasattr(self, 'roi_remove_button'):
            self.roi_remove_button.config(state='normal' if enabled else 'disabled')

    def add_roi_row_range(self):
        if getattr(self, 'roi_second_enabled', False):
            return
        row_begin, row_end = self.get_roi_row_range()
        max_row = getattr(self.row_slider, 'max_value', row_end)
        height = max(row_end - row_begin, 1)
        gap = max(2, int(round(height * 0.35)))
        second_begin = min(row_end + gap + 1, max(max_row - height, 0))
        second_end = min(second_begin + height, max_row)
        if second_end <= second_begin:
            second_end = min(max_row, second_begin + 1)
        if second_end <= second_begin:
            second_begin, second_end = row_begin, row_end
        self.roi_second_enabled = True
        self.parm['roi_second_enabled'] = True
        self.row_slider.set_limits(0, max_row, row_begin, row_end, second_begin, second_end, invoke=False)
        self._sync_roi_row_buttons()
        self.on_roi_rows_changed(row_begin, row_end, second_begin, second_end, redraw=True)
        if hasattr(self, 'last_spectrum_roi'):
            self.plot_roi_spectrum()

    def remove_roi_row_range(self):
        if not getattr(self, 'roi_second_enabled', False):
            return
        self.roi_second_enabled = False
        self.parm['roi_second_enabled'] = False
        self._sync_roi_row_buttons()
        self.on_roi_rows_changed(
            self.row_begin.get(),
            self.row_end.get(),
            self.row2_begin.get() if hasattr(self, 'row2_begin') else self.row_begin.get(),
            self.row2_end.get() if hasattr(self, 'row2_end') else self.row_end.get(),
            redraw=True,
        )
        if hasattr(self, 'last_spectrum_roi'):
            self.plot_roi_spectrum()

    def update_roi_row_slider_limits(self, row_count):
        """Resize the ROI row slider to the current image height."""
        if not hasattr(self, 'row_slider'):
            return
        max_row = max(int(row_count) - 1, 0)
        row_begin = min(self.row_begin.get(), max_row)
        row_end = min(self.row_end.get(), max_row)
        row2_begin = min(self.row2_begin.get(), max_row) if hasattr(self, 'row2_begin') else row_begin
        row2_end = min(self.row2_end.get(), max_row) if hasattr(self, 'row2_end') else row_end
        if max_row > 0 and row_begin >= row_end:
            if row_begin >= max_row:
                row_begin = max_row - 1
                row_end = max_row
            else:
                row_end = row_begin + 1
        elif max_row == 0:
            row_begin = 0
            row_end = 0
        self.row_slider.set_limits(0, max_row, row_begin, row_end, row2_begin, row2_end, invoke=False)
        self._sync_roi_row_buttons()
        self.on_roi_rows_changed(
            self.row_begin.get(),
            self.row_end.get(),
            self.row2_begin.get() if hasattr(self, 'row2_begin') else self.row_begin.get(),
            self.row2_end.get() if hasattr(self, 'row2_end') else self.row_end.get(),
            redraw=False,
        )

    def get_roi_row_range(self):
        """Return the first ROI row selection from the slider controls."""
        row_begin, row_end = sorted((int(self.row_begin.get()), int(self.row_end.get())))
        self.parm['row_begin'] = row_begin
        self.parm['row_end'] = row_end
        if hasattr(self, 'row_range_label'):
            self.on_roi_rows_changed(
                row_begin,
                row_end,
                self.row2_begin.get() if hasattr(self, 'row2_begin') else row_begin,
                self.row2_end.get() if hasattr(self, 'row2_end') else row_end,
                redraw=False,
            )
        return row_begin, row_end

    def get_roi_row_ranges(self):
        """Return active ROI row selections as inclusive row ranges."""
        ranges = []
        row_begin, row_end = sorted((int(self.row_begin.get()), int(self.row_end.get())))
        self.parm['row_begin'] = row_begin
        self.parm['row_end'] = row_end
        if row_end >= row_begin:
            ranges.append((row_begin, row_end))
        if getattr(self, 'roi_second_enabled', False) and hasattr(self, 'row2_begin') and hasattr(self, 'row2_end'):
            row2_begin, row2_end = sorted((int(self.row2_begin.get()), int(self.row2_end.get())))
            self.parm['row2_begin'] = row2_begin
            self.parm['row2_end'] = row2_end
            if row2_end >= row2_begin:
                ranges.append((row2_begin, row2_end))
        self.parm['roi_second_enabled'] = bool(getattr(self, 'roi_second_enabled', False))
        return ranges

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

    def on_bg_rows_changed(self, bg_row_begin=None, bg_row_end=None, bg2_row_begin=None, bg2_row_end=None, redraw=True):
        """Sync the selected background row ranges with the UI and image overlay."""
        if bg_row_begin is None:
            bg_row_begin = self.bg_row_begin.get()
        if bg_row_end is None:
            bg_row_end = self.bg_row_end.get()
        if bg2_row_begin is None:
            bg2_row_begin = self.bg2_row_begin.get() if hasattr(self, 'bg2_row_begin') else bg_row_begin
        if bg2_row_end is None:
            bg2_row_end = self.bg2_row_end.get() if hasattr(self, 'bg2_row_end') else bg_row_end
        bg_row_begin = int(bg_row_begin)
        bg_row_end = int(bg_row_end)
        bg2_row_begin = int(bg2_row_begin)
        bg2_row_end = int(bg2_row_end)
        self.parm['bg_row_begin'] = bg_row_begin
        self.parm['bg_row_end'] = bg_row_end
        self.parm['bg2_row_begin'] = bg2_row_begin
        self.parm['bg2_row_end'] = bg2_row_end
        if hasattr(self, 'bg_row_range_label'):
            row_count = (getattr(self, 'bg_row_slider', None).max_value + 1) if hasattr(self, 'bg_row_slider') else (bg_row_end + 1)
            self.bg_row_range_label.config(text=f"{bg_row_begin}-{bg_row_end}; {bg2_row_begin}-{bg2_row_end}  ({row_count} rows)")
        if redraw and hasattr(self, 'current_display_data'):
            self.refresh_image_overlays()

    def update_bg_row_slider_limits(self, row_count):
        """Resize the background row slider to the current image height."""
        if not hasattr(self, 'bg_row_slider'):
            return
        max_row = max(int(row_count) - 1, 0)
        bg_row_begin = min(self.bg_row_begin.get(), max_row)
        bg_row_end = min(self.bg_row_end.get(), max_row)
        bg2_row_begin = min(self.bg2_row_begin.get(), max_row) if hasattr(self, 'bg2_row_begin') else bg_row_begin
        bg2_row_end = min(self.bg2_row_end.get(), max_row) if hasattr(self, 'bg2_row_end') else bg_row_end
        self.bg_row_slider.set_limits(0, max_row, bg_row_begin, bg_row_end, bg2_row_begin, bg2_row_end, invoke=False)
        self.on_bg_rows_changed(
            self.bg_row_begin.get(),
            self.bg_row_end.get(),
            self.bg2_row_begin.get(),
            self.bg2_row_end.get(),
            redraw=False,
        )

    def get_bg_row_range(self):
        """Return the first background row selection for backward compatibility."""
        bg_row_begin, bg_row_end = sorted((int(self.bg_row_begin.get()), int(self.bg_row_end.get())))
        self.parm['bg_row_begin'] = bg_row_begin
        self.parm['bg_row_end'] = bg_row_end
        if hasattr(self, 'bg_row_range_label'):
            self.on_bg_rows_changed(
                bg_row_begin,
                bg_row_end,
                self.bg2_row_begin.get() if hasattr(self, 'bg2_row_begin') else bg_row_begin,
                self.bg2_row_end.get() if hasattr(self, 'bg2_row_end') else bg_row_end,
                redraw=False,
            )
        return bg_row_begin, bg_row_end

    def get_bg_row_ranges(self):
        """Return valid background row selections as inclusive row ranges."""
        ranges = []
        for left_var, right_var, begin_key, end_key in (
            (self.bg_row_begin, self.bg_row_end, 'bg_row_begin', 'bg_row_end'),
            (
                getattr(self, 'bg2_row_begin', self.bg_row_begin),
                getattr(self, 'bg2_row_end', self.bg_row_end),
                'bg2_row_begin',
                'bg2_row_end',
            ),
        ):
            begin, end = sorted((int(left_var.get()), int(right_var.get())))
            self.parm[begin_key] = begin
            self.parm[end_key] = end
            if end > begin:
                ranges.append((begin, end))
        if hasattr(self, 'bg_row_range_label'):
            self.on_bg_rows_changed(
                self.bg_row_begin.get(),
                self.bg_row_end.get(),
                self.bg2_row_begin.get() if hasattr(self, 'bg2_row_begin') else self.bg_row_begin.get(),
                self.bg2_row_end.get() if hasattr(self, 'bg2_row_end') else self.bg_row_end.get(),
                redraw=False,
            )
        return ranges

    def _score_background_band(self, data, invalid, start, height, col_begin, col_end, roi_center):
        stop = min(start + height, data.shape[0])
        if stop <= start:
            return None
        band = np.asarray(data[start:stop, col_begin:col_end + 1], dtype=float)
        if invalid is not None and invalid.shape == data.shape:
            band = np.where(invalid[start:stop, col_begin:col_end + 1], np.nan, band)
        finite = band[np.isfinite(band)]
        if finite.size < max(12, band.size * 0.25):
            return None
        median = float(np.nanmedian(finite))
        p90 = float(np.nanpercentile(finite, 90))
        spread = float(np.nanpercentile(finite, 75) - np.nanpercentile(finite, 25))
        distance = abs((start + stop - 1) / 2.0 - roi_center)
        return median + 0.35 * p90 + 0.75 * spread + 0.002 * distance

    def _best_background_band(self, data, invalid, candidates, height, col_begin, col_end, roi_center):
        best = None
        for start in candidates:
            score = self._score_background_band(data, invalid, int(start), height, col_begin, col_end, roi_center)
            if score is None:
                continue
            if best is None or score < best[0]:
                best = (score, int(start))
        if best is None:
            return None
        start = best[1]
        return start, min(start + height - 1, data.shape[0] - 1)

    def auto_select_background_rows(self):
        """Suggest two local background strips around the current signal ROI."""
        if not hasattr(self, 'immm'):
            messagebox.showwarning("Auto BG", "Process an image before selecting background rows.")
            return
        raw_roi_ranges = self.get_roi_row_ranges() if hasattr(self, 'get_roi_row_ranges') else [self.get_roi_row_range()]
        roi_ranges = []
        for begin, end in raw_roi_ranges:
            begin, end = sorted((int(begin), int(end)))
            if end >= begin:
                roi_ranges.append((begin, end))
        if not roi_ranges:
            messagebox.showwarning("Auto BG", "Select a valid ROI before selecting background rows.")
            return
        row_begin = min(begin for begin, _end in roi_ranges)
        row_end = max(end for _begin, end in roi_ranges)
        col_begin, col_end = self.get_roi_col_range()
        data = np.asarray(self.immm, dtype=float)
        row_count = data.shape[0]
        height = max(max(end - begin + 1 for begin, end in roi_ranges), 1)
        separation = max(3, int(round(height * 0.35)))
        search_margin = max(60, height * 4)
        roi_center = (row_begin + row_end) / 2.0
        invalid = getattr(self, 'processed_gap_mask', None)

        above_min = max(0, row_begin - search_margin - height)
        above_max = max(-1, row_begin - separation - height)
        below_min = min(row_count - height, row_end + separation + 1)
        below_max = min(row_count - height, row_end + separation + search_margin)

        above_candidates = range(above_min, above_max + 1) if above_max >= above_min else []
        below_candidates = range(below_min, below_max + 1) if below_max >= below_min else []
        above = self._best_background_band(data, invalid, above_candidates, height, col_begin, col_end, roi_center)
        below = self._best_background_band(data, invalid, below_candidates, height, col_begin, col_end, roi_center)

        if above is None and below is None:
            messagebox.showwarning("Auto BG", "Could not find a suitable local background strip.")
            return
        if above is None:
            above = below
        if below is None:
            below = above

        self.bg_row_slider.set_limits(0, row_count - 1, above[0], above[1], below[0], below[1], invoke=False)
        self.on_bg_rows_changed(above[0], above[1], below[0], below[1], redraw=True)
        if getattr(self, 'bg_subtraction_enabled', False) and hasattr(self, 'last_spectrum_roi'):
            self.plot_roi_spectrum()

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
            self.bg_toggle.config(style='TButton', text="BG Remove (OFF)")
        for attr_name in (
            'last_spectrum_roi',
            'last_spectrum_roi_uncorrected',
            'current_spectrum_moving_avg',
            'current_spectrum_bg_removed',
            'current_spectrum_gap_corrected',
            'current_spectrum_gap_columns',
            'current_spectrum_gap_indices',
            'current_spectrum_x_axis',
            'current_spectrum_view_label',
            'current_spectrum_trace_extracted',
            '_spectrum_gap_marker_y',
            '_spectrum_gap_marker_pad',
            'last_peak_fit_profile',
        ):
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        if hasattr(self, 'ax_spectrum'):
            self.ax_spectrum.clear()
            if hasattr(self, 'fig_spectrum') and hasattr(self.fig_spectrum, 'legends'):
                self.fig_spectrum.legends.clear()
            self.ax_spectrum.set_xticks([])
            self.ax_spectrum.set_yticks([])
            self.ax_spectrum.set_xticklabels([])
            self.ax_spectrum.set_yticklabels([])
            for spine in self.ax_spectrum.spines.values():
                spine.set_linewidth(0.5)
            if hasattr(self, 'spectrum_status_label'):
                self.spectrum_status_label.config(text="No spectrum plotted")
            if hasattr(spectrum_controls, '_set_spectrum_view_mode'):
                spectrum_controls._set_spectrum_view_mode(self, None)
            if hasattr(spectrum_controls, '_clear_peak_fit_results_box'):
                spectrum_controls._clear_peak_fit_results_box(self)
            if hasattr(self, 'canvas_spectrum'):
                self.canvas_spectrum.draw()

    def open_calibration_panel(self):
        """Switch to the calibration controls."""
        if hasattr(self, 'calibration_tab'):
            self.notebook.select(self.calibration_tab)

    def show_calibrated_spectrum_view(self):
        """Show the calibrated spectrum view if an energy calibration exists."""
        if getattr(self, 'energy_calibration', None) is None:
            return
        return calibration_controls.apply_energy_calibration(self)

    def import_calibration(self):
        return calibration_controls.import_calibration(self)
    def fit_calibration_spectrum(self):
        return calibration_controls.fit_calibration_spectrum(self)
    def calculate_energy_calibration(self):
        return calibration_controls.calculate_energy_calibration(self)
    def apply_energy_calibration(self):
        return calibration_controls.apply_energy_calibration(self)
    def show_calibration_fit(self):
        return calibration_controls.show_calibration_fit(self)
    def compare_calibration_overlay(self):
        return calibration_controls.compare_calibration_overlay(self)
    def save_calibrated_spectrum_data(self):
        return calibration_controls.save_calibrated_spectrum_data(self)
        
    # Delegated methods only, all previous method bodies for moved functions are removed
    def load_tiff(self):
        return image_processing.load_tiff(self)
    def load_tiff_stack(self):
        return image_processing.load_tiff_stack(self)
    def auto_threshold(self):
        return image_processing.auto_threshold(self)
    def remove_cross_raw(self):
        return image_processing.remove_cross_raw(self)
    def process_image(self):
        return image_processing.process_image(self)
    def show_gap_mask(self):
        return image_processing.show_gap_mask(self)
    def display_image(self, data, ax, title):
        return image_processing.display_image(self, data, ax, title)
    def draw_lines(self):
        return image_processing.draw_lines(self)
    def draw_background_roi(self):
        return image_processing.draw_background_roi(self)
    def save_processed_image(self):
        return image_processing.save_processed_image(self)
    def update_image_cmap(self):
        return image_processing.update_image_cmap(self)
    def save_project(self):
        return project_io.save_project(self)
    def open_project(self):
        return project_io.open_project(self)
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
    def plot_fitted_integrated_diff(self):
        return iad_controls.plot_fitted_integrated_diff(self)
    def calculate_and_display_satellite_peak_iad(self):
        return iad_controls.calculate_and_display_satellite_peak_iad(self)
    def calculate_and_display_fitted_satellite_peak_iad(self):
        return iad_controls.calculate_and_display_fitted_satellite_peak_iad(self)
    def pick_ref_color(self):
        return iad_controls.pick_ref_color(self)
    def toggle_background_removal(self):
        return spectrum_controls.toggle_background_removal(self)
    def plot_roi_spectrum(self):
        return spectrum_controls.plot_roi_spectrum(self)
    def apply_moving_average(self):
        return spectrum_controls.apply_moving_average(self)
    def show_peak_fit_profile(self):
        return spectrum_controls.show_peak_fit_profile(self)
    def show_original_spectrum(self):
        return spectrum_controls.show_original_spectrum(self)
    def show_gap_corrected_spectrum(self):
        return spectrum_controls.show_gap_corrected_spectrum(self)
    def show_peak_fit_view(self):
        return spectrum_controls.show_peak_fit_view(self)
    def toggle_gap_correction(self):
        return spectrum_controls.toggle_gap_correction(self)
    def toggle_trace_extraction(self):
        return spectrum_controls.toggle_trace_extraction(self)
    def on_spectrum_canvas_configure(self, event=None):
        return spectrum_controls.on_spectrum_canvas_configure(self, event)
    def open_color_picker(self):
        return spectrum_controls.open_color_picker(self)
    def update_plot(self):
        return spectrum_controls.update_plot(self)
    def save_spectrum_data(self):
        return spectrum_controls.save_spectrum_data(self)
    def save_peak_fit_data(self):
        return spectrum_controls.save_peak_fit_data(self)
    def save_peak_fit_parameters(self):
        return spectrum_controls.save_peak_fit_parameters(self)
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
