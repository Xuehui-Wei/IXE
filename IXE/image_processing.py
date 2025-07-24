# -*- coding: utf-8 -*-
import numpy as np
import re
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
import scipy.ndimage
import matplotlib.pyplot as plt
#from IXE.remove_cross import delete_zero_rows_and_columns

def load_tiff(self):
    """Load a TIFF file and display it in the image panel"""
    filepath = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tiff *.tif")])
    if filepath:
        self.original_filepath = filepath  # Store the file path
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
        self.spect_processor = None
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
        self.display_image(self.immm, self.ax_image, "Processed Image")
        self.canvas_image.draw()
    except Exception as e:
        print(f"Error processing image: {e}")

def display_image(self, data, ax, title):
    """Display an image in the specified axes"""
    import matplotlib as mpl
    ax.clear()
    img_aspect = data.shape[1] / data.shape[0]  # width / height aspect ratio
    fig_width = 6  # you can define a static width or calculate based on your need
    fig_height = fig_width / img_aspect  # height based on the aspect ratio

    # Create the figure with dynamic size and dpi
    self.fig_image = plt.Figure(figsize=(fig_width, fig_height), dpi=self.fig_dpi)
    img = ax.imshow(data, interpolation='antialiased', vmin=float(self.vmin_entry.get()), vmax=float(self.vmax_entry.get()), cmap='viridis')
    ax.set_title(title, fontsize=5)
    ax.set_xlabel('Columns', fontsize=4)
    ax.set_ylabel('Rows', fontsize=4)
    ax.tick_params(axis='both', which='major', labelsize=4)
    ax.tick_params(axis='both', which='major', width=0.5)
    if not hasattr(self, 'cbar') or self.cbar is None:
        self.cbar = self.fig_image.colorbar(img, ax=ax, fraction=0.03, pad=0.04)
        self.cbar.ax.tick_params(labelsize=4)
        self.cbar.ax.tick_params(width=0.5)
    else:
        self.cbar.update_normal(img)
    ax.set_aspect('auto', adjustable='box')
    ax.set_xlim(0, data.shape[1])
    ax.set_ylim(data.shape[0], 0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    self.fig_image.tight_layout(pad=3.0, h_pad=0.5, w_pad=0.5)
    self.canvas_image.draw()

def draw_lines(self):
    """Draw lines at row_begin and row_end on the processed image."""
    if not hasattr(self, 'immm'):
        messagebox.showwarning("Reminder", "Error: No processed image to draw lines on.")
        return
    try:
        row_begin = int(self.row_begin.get())
        row_end = int(self.row_end.get())
        if row_begin < 0 or row_end < 0 or row_begin >= self.immm.shape[0] or row_end >= self.immm.shape[0]:
            print("Invalid row indices.")
            return
        if hasattr(self, 'line_begin') and self.line_begin:
            self.line_begin.remove()
        if hasattr(self, 'line_end') and self.line_end:
            self.line_end.remove()
        self.line_begin = self.ax_image.plot([0, self.immm.shape[1]-1], [row_begin, row_begin], color='w', linestyle='--', linewidth=0.5)[0]
        self.line_end = self.ax_image.plot([0, self.immm.shape[1]-1], [row_end, row_end], color='w', linestyle='--', linewidth=0.5)[0]
        self.canvas_image.draw()
    except Exception as e:
        print(f"Error drawing lines: {e}")

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
