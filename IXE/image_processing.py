# -*- coding: utf-8 -*-
import numpy as np
import re
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
#from IXE.remove_cross import delete_zero_rows_and_columns

def load_tiff(self):
    """Load a TIFF file and display it in the image panel"""
    filepath = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tiff *.tif")])
    if filepath:
        self.original_filepath = filepath  # Store the file path
        if hasattr(self, 'reset_spectrum_state'):
            self.reset_spectrum_state()
        if hasattr(self, 'immm'):
            delattr(self, 'immm')
        if hasattr(self, 'line_begin') and self.line_begin:
            self.line_begin.remove()
            self.line_begin = None
        if hasattr(self, 'line_end') and self.line_end:
            self.line_end.remove()
            self.line_end = None
        if hasattr(self, 'bg_rect') and self.bg_rect:
            self.bg_rect.remove()
            self.bg_rect = None
        #self.file_path_label.config(text=f"File Path: {filepath}")  # Update the label with the file path
        self.file_path_entry.delete(0, tk.END)  # Clear the current value
        self.file_path_entry.insert(0, filepath)
        run_number_match = re.search(r"Run_(\d+)", filepath)
        if run_number_match:
            run_number = run_number_match.group(1)
            self.run_number_label.config(text=f"Run: {run_number}")
            self.run_number_label_spectrum.config(text=f"Run: {run_number}")
        else:
            print("Error: Run number not found in filename.")
        self.im = Image.open(filepath)
        self.imm = np.array(self.im, dtype=np.float32)
        if hasattr(self, 'update_roi_row_slider_limits'):
            self.update_roi_row_slider_limits(self.imm.shape[0])
        if hasattr(self, 'update_bg_row_slider_limits'):
            self.update_bg_row_slider_limits(self.imm.shape[0])
        if hasattr(self, 'update_roi_col_slider_limits'):
            self.update_roi_col_slider_limits(self.imm.shape[1])
        self.display_image(self.imm, self.ax_image, "Raw Image")
        self.canvas_image.draw()

def auto_threshold(self):
    """Calculate threshold using Otsu's method."""
    from skimage.filters import threshold_otsu
    if hasattr(self, 'imm'):
        thresh = threshold_otsu(self.imm)
        self.thresh_entry.delete(0, tk.END)
        self.thresh_entry.insert(0, str(int(thresh)))
        self.parm['threshold'] = thresh

def remove_cross_raw(self):
    """Remove the zero-intensity cross from the raw image."""
    from IXE.remove_cross import delete_zero_rows_and_columns
    if not hasattr(self, 'imm'):
        print("Error: No image loaded. Import a TIFF first.")
        return
    try:
        self.imm = delete_zero_rows_and_columns(self.imm)
        if hasattr(self, 'reset_spectrum_state'):
            self.reset_spectrum_state()
        if hasattr(self, 'update_roi_row_slider_limits'):
            self.update_roi_row_slider_limits(self.imm.shape[0])
        if hasattr(self, 'update_bg_row_slider_limits'):
            self.update_bg_row_slider_limits(self.imm.shape[0])
        if hasattr(self, 'update_roi_col_slider_limits'):
            self.update_roi_col_slider_limits(self.imm.shape[1])
        self.display_image(self.imm, self.ax_image, "Raw Image (Cross Removed)")#
        self.canvas_image.draw()
    except Exception as e:
        print(f"Error removing cross: {e}")

def process_image(self):
    """Process the image (rotate and threshold)"""
    from IXE.spectrum_utils import SpectrumProcessor
    if not hasattr(self, 'imm'):
        return
    try:
        tilt = float(self.tilt_entry.get())
        threshold = float(self.thresh_entry.get())
        self.immm = scipy.ndimage.rotate(self.imm, angle=tilt)
        self.immm[self.immm < threshold] = 0
        self.spect_processor = SpectrumProcessor(self.immm, self.parm['n_moveavg'])
        if hasattr(self, 'update_roi_row_slider_limits'):
            self.update_roi_row_slider_limits(self.immm.shape[0])
        if hasattr(self, 'update_bg_row_slider_limits'):
            self.update_bg_row_slider_limits(self.immm.shape[0])
        if hasattr(self, 'update_roi_col_slider_limits'):
            self.update_roi_col_slider_limits(self.immm.shape[1])
        self.display_image(self.immm, self.ax_image, "Processed Image")
        self.canvas_image.draw()
    except Exception as e:
        print(f"Error processing image: {e}")

def display_image(self, data, ax, title):
    """Display an image in the specified axes"""
    self.root.update_idletasks()

    ax.clear()
    self.current_display_data = data
    self.line_begin = None
    self.line_end = None
    self.bg_rect = None

    canvas_widget = self.canvas_image.get_tk_widget()
    canvas_width = max(canvas_widget.winfo_width(), 420)
    canvas_height = max(canvas_widget.winfo_height(), 320)
    self.fig_image.set_size_inches(
        canvas_width / self.base_display_dpi,
        canvas_height / self.base_display_dpi,
        forward=True,
    )
    self.fig_image.set_dpi(self.fig_dpi)
    self.fig_image.subplots_adjust(left=0.14, bottom=0.14, right=0.88, top=0.95)

    img = ax.imshow(
        data,
        interpolation='antialiased',
        vmin=float(self.vmin_entry.get()),
        vmax=float(self.vmax_entry.get()),
        cmap='viridis',
    )
    if hasattr(self, 'image_title_label'):
        self.image_title_label.config(text=title)
    if hasattr(self, 'image_xlabel_label'):
        self.image_xlabel_label.config(text='Columns')
    if hasattr(self, 'update_image_ylabel'):
        self.update_image_ylabel()

    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='both', which='major', labelsize=14, width=0.9, pad=4)
    if not hasattr(self, 'cbar') or self.cbar is None:
        self.cbar = self.fig_image.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        self.cbar.ax.tick_params(labelsize=13, width=0.9)
    else:
        self.cbar.update_normal(img)
        self.cbar.ax.tick_params(labelsize=13, width=0.9)
    ax.set_aspect('auto', adjustable='box')
    ax.set_xlim(0, data.shape[1])
    ax.set_ylim(data.shape[0], 0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    if hasattr(self, 'refresh_image_overlays'):
        self.refresh_image_overlays()
    else:
        self.canvas_image.draw_idle()

def draw_lines(self):
    """Draw lines at row_begin and row_end on the processed image."""
    if not hasattr(self, 'current_display_data'):
        return
    try:
        data = self.current_display_data
        row_begin, row_end = self.get_roi_row_range()
        col_begin, col_end = self.get_roi_col_range()
        if row_begin < 0 or row_end < 0 or row_begin >= data.shape[0] or row_end >= data.shape[0]:
            print("Invalid row indices.")
            return
        if col_begin < 0 or col_end < 0 or col_begin >= data.shape[1] or col_end >= data.shape[1]:
            print("Invalid column indices.")
            return
        if hasattr(self, 'line_begin') and self.line_begin:
            self.line_begin.remove()
        if hasattr(self, 'line_end') and self.line_end:
            self.line_end.remove()
        self.line_begin = self.ax_image.plot([col_begin, col_end], [row_begin, row_begin], color='w', linestyle='--', linewidth=0.5)[0]
        self.line_end = self.ax_image.plot([col_begin, col_end], [row_end, row_end], color='w', linestyle='--', linewidth=0.5)[0]
        self.canvas_image.draw()
    except Exception as e:
        print(f"Error drawing lines: {e}")

def draw_background_roi(self):
    """Draw the translucent background ROI rectangle on the current image."""
    if not hasattr(self, 'current_display_data'):
        return
    try:
        data = self.current_display_data
        bg_row_begin, bg_row_end = self.get_bg_row_range()
        col_begin, col_end = self.get_roi_col_range()
        if bg_row_begin < 0 or bg_row_end < 0 or bg_row_begin >= data.shape[0] or bg_row_end >= data.shape[0]:
            return
        if col_begin < 0 or col_end < 0 or col_begin >= data.shape[1] or col_end >= data.shape[1]:
            return
        if hasattr(self, 'bg_rect') and self.bg_rect:
            self.bg_rect.remove()
        self.bg_rect = Rectangle(
            (col_begin, bg_row_begin),
            max(col_end - col_begin + 1, 1),
            max(bg_row_end - bg_row_begin + 1, 1),
            linewidth=0.4,
            edgecolor='white',
            facecolor='white',
            alpha=0.3,
        )
        self.ax_image.add_patch(self.bg_rect)
        self.canvas_image.draw_idle()
    except Exception as e:
        print(f"Error drawing background ROI: {e}")

def save_processed_image(self):
    """Save the processed image with high quality"""
    import matplotlib.pyplot as plt
    if not hasattr(self, 'immm'):
        messagebox.showwarning("Reminder", "Error: No processed image to save")
        return
    try:
        filetypes = [
            ('TIFF files', '*.tif;*.tiff'),
            ('PNG files', '*.png'),
            ('JPEG files', '*.jpg;*.jpeg'),
            ('All files', '*.*')
        ]
        save_path = filedialog.asksaveasfilename(defaultextension='.tiff', filetypes=filetypes)
        if not save_path:
            return
        save_fig = plt.figure(figsize=(8, 6), dpi=300)
        save_ax = save_fig.add_subplot(111)
        save_ax.imshow(self.immm, interpolation='antialiased', vmin=float(self.vmin_entry.get()), vmax=float(self.vmax_entry.get()), cmap='viridis')
        save_ax.set_title('Processed Image', fontsize=14, fontname='Arial')
        save_fig.savefig(save_path, dpi=300, bbox_inches='tight', quality=95)
        plt.close(save_fig)
        print(f"Image saved to {save_path}")
    except Exception as e:
        print(f"Error saving image: {e}")
# image_processing.py
# Image loading, cross removal, thresholding, and display functions for TIFFAnalyzer.

# All functions will be moved here unchanged in the next step.
