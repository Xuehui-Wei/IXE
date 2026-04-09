# -*- coding: utf-8 -*-
# spectrum_controls.py
# Spectrum plotting, moving average, and spectrum controls for TIFFAnalyzer.


import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk, filedialog, simpledialog
from tkinter.colorchooser import askcolor
import csv
import re


def plot_roi_spectrum(self, line_color=None, line_style=None, line_width=None):
    """Plot the spectrum for the selected ROI with custom line style, color, and width."""
    if not hasattr(self, 'spect_processor'):
        print("Error: Process the image first")
        return

    try:
        row_begin, row_end = self.get_roi_row_range()
        column_begin, column_end = self.get_roi_col_range()

        # Clear previous spectrum plot
        self.ax_spectrum.clear()

        # Get spectrum data
        spectrum = self.spect_processor.get_spectrum(
            row_begin,
            row_end + 1,
            column_begin,
            column_end + 1
        )
        # Normalize the spectrum before plotting
        norm_spectrum = self.spect_processor.get_norm_spectrum(spectrum)

        # Store the last displayed ROI spectrum (normalized, raw)
        self.last_spectrum_roi = norm_spectrum.copy()
        self.current_spectrum_moving_avg = 0
        self.current_spectrum_bg_removed = self.bg_subtraction_enabled

        # Get user-selected values for line color, style, and width
        line_color = line_color or self.line_color.get()
        line_style = line_style or self.line_style.get()
        try:
            line_width = float(line_width) if line_width else float(self.line_width.get())
        except ValueError:
            print("Invalid line width value. Using default.")
            line_width = 0.5  # Default value if invalid input

        # Plot the spectrum with the specified color, style, and width
        self.ax_spectrum.plot(norm_spectrum, color=line_color, linestyle=line_style, linewidth=line_width,
                              label=f"Run {self.run_number_label.cget('text').split(': ')[1]}")
        self.ax_spectrum.legend()

        # Update the title and labels
        self.ax_spectrum.set_title("")
        self.ax_spectrum.set_xlabel("Column Index")
        self.ax_spectrum.set_ylabel("Intensity")
        self.ax_spectrum.tick_params(axis='both', which='major', width=0.5)
        self.ax_spectrum.set_yticks([])  # Removes y-axis ticks
        self.ax_spectrum.set_yticklabels([])  # Removes y-axis labels
        for spine in self.ax_spectrum.spines.values():
          spine.set_linewidth(0.5)
        # Adjust the layout and update the canvas
        self.fig_spectrum.tight_layout()
        self.canvas_spectrum.draw()
    except Exception as e:
        print(f"Error plotting spectrum: {e}")

def update_plot(self, event=None):
    """Update the plot with the new line style and line width."""
    # Get the user-selected values for line style, color, and width
    line_color = self.line_color.get()  # Assuming line_color is set somewhere else
    line_style = self.line_style.get()  # Get the selected line style
    try:
        line_width = float(self.line_width.get())  # Get the line width from the entry box
    except ValueError:
        print("Invalid line width value. Using default.")
        line_width = 0.5  # Default value if invalid input

    # Now call the function to replot the spectrum with the new style and width
    self.plot_roi_spectrum(line_color=line_color, line_style=line_style, line_width=line_width)


        
def open_color_picker(self):
    """Open a color picker dialog and change the color of the selected line."""
    color = askcolor()[1]  # Open the color picker and get the selected color
    
    if color:
        self.line_color.set(color)  # Update the line color variable
        for line in self.ax_spectrum.lines:
            # Assuming the label corresponds to the line you want to update
            if line.get_label() == "Run {}".format(self.run_number_label.cget('text').split(': ')[1]):
                line.set_color(color)  # Update the color of the line
                break
        
        # Redraw the canvas with the updated line color
        self.fig_spectrum.tight_layout()
        self.canvas_spectrum.draw()
        
def apply_moving_average(self):
    """Apply moving average to the spectrum data and update the plot."""
    if not hasattr(self, 'spect_processor'):
        print("Error: Process the image first")
        return

    try:
        # Get the spectrum data from the selected ROI
        n_moveavg = int(self.n_moveavg.get())  # Get the user-input value from the UI
        row_begin, row_end = self.get_roi_row_range()
        column_begin, column_end = self.get_roi_col_range()
        
        # Update the n_moveavg in the self.parm dictionary
        self.parm['n_moveavg'] = n_moveavg

        # --- FIX START ---
        # OLD INCORRECT LINE: This wiped the background settings by making a new object
        # self.spect_processor = self.spect_processor.__class__(self.immm, self.parm['n_moveavg'])
        
        # NEW CORRECT LINE: Update the existing processor's setting directly.
        # This preserves the background spectrum and the 'bg_subtracted' toggle state.
        self.spect_processor.n_moveavg = self.parm['n_moveavg']
        # --- FIX END ---

        # Get the spectrum data from the selected ROI
        # Because we kept the existing spect_processor, get_spectrum() now checks 
        # the internal self.bg_subtracted flag and applies subtraction if it was enabled.
        spectrum = self.spect_processor.get_spectrum(
            row_begin,
            row_end + 1,
            column_begin,
            column_end + 1
        )

        # Apply the moving average to the spectrum data
        smoothed_spectrum = self.spect_processor.moving_average(spectrum)
        norm_smoothed_spectrum = self.spect_processor.get_norm_spectrum(smoothed_spectrum)

        # Store the last displayed ROI spectrum (normalized, smoothed)
        self.last_spectrum_roi = norm_smoothed_spectrum.copy()
        self.current_spectrum_moving_avg = self.parm['n_moveavg']
        self.current_spectrum_bg_removed = self.bg_subtraction_enabled

        # Plot the smoothed spectrum in the spectrum axes
        self.ax_spectrum.clear()  # Clear the previous plot
        
        # Update title to reflect if BG is removed
        self.ax_spectrum.set_title("")
        
        self.ax_spectrum.plot(norm_smoothed_spectrum, color=self.line_color.get(), linestyle=self.line_style.get(), linewidth=float(self.line_width.get()), 
                              label=f"Run {self.run_number_label.cget('text').split(': ')[1]}")
        self.ax_spectrum.legend()
        self.ax_spectrum.set_xlabel("Column Index")
        self.ax_spectrum.set_ylabel("Intensity")
        self.ax_spectrum.tick_params(axis='both', which='major', width=0.5)
        self.ax_spectrum.set_yticks([])  # Removes y-axis ticks
        self.ax_spectrum.set_yticklabels([])  # Removes y-axis labels
        for spine in self.ax_spectrum.spines.values():
          spine.set_linewidth(0.5)
        self.fig_spectrum.tight_layout()
        self.canvas_spectrum.draw()  # Update the plot with the smoothed spectrum

    except Exception as e:
        print(f"Error applying moving average: {e}")

def toggle_background_removal(self):
    """Toggle background subtraction on/off."""
    self.bg_subtraction_enabled = not self.bg_subtraction_enabled
    from tkinter import ttk
    style = ttk.Style()
    if self.bg_subtraction_enabled:
        style.configure('Toggle.TButton', background='light green')
        self.bg_toggle.config(style='Toggle.TButton', text="BG Remove (ON)")
    else:
        self.bg_toggle.config(style='TButton', text="BG Remove (OFF)")
    if not hasattr(self, 'spect_processor'):
        print("Error: Process the image first")
        self.bg_subtraction_enabled = False
        self.bg_toggle.config(style='TButton', text="BG Remove (OFF)")
        return
    try:
        bg_row_start, bg_row_end = self.get_bg_row_range()
        bg_col_start, bg_col_end = self.get_roi_col_range()
        if (bg_row_start >= bg_row_end or bg_col_start >= bg_col_end or bg_row_start < 0 or bg_col_start < 0):
            raise ValueError("Invalid background ROI coordinates")
        self.spect_processor.set_background_roi(bg_row_start, bg_row_end + 1, bg_col_start, bg_col_end + 1)
        self.spect_processor.toggle_background_subtraction(self.bg_subtraction_enabled)
        if hasattr(self, 'last_spectrum_roi'):
            self.plot_roi_spectrum()
    except ValueError as ve:
        print(f"Invalid background ROI: {ve}")
        self.bg_subtraction_enabled = False
        self.bg_toggle.config(style='TButton', text="BG Remove (OFF)")
    except Exception as e:
        print(f"Error toggling background removal: {e}")
        self.bg_subtraction_enabled = False
        self.bg_toggle.config(style='TButton', text="BG Remove (OFF)")

def save_spectrum_data(self):
    """Save the spectrum currently displayed in the GUI."""
    if not hasattr(self, 'last_spectrum_roi'):
        print("Error: Plot a spectrum first")
        return
    
    try:
        y_data = self.last_spectrum_roi.copy()
        x_data = np.arange(len(y_data))
        moving_avg = getattr(self, 'current_spectrum_moving_avg', 0)
        bg_removed = getattr(self, 'current_spectrum_bg_removed', self.bg_subtraction_enabled)
        row_begin, row_end = self.get_roi_row_range()
        column_begin, column_end = self.get_roi_col_range()
        bg_row_begin, bg_row_end = self.get_bg_row_range()
        
        # Get the input filename and extract the run number
        filename = self.original_filepath.split("/")[-1]  # Extract filename from path
        run_number_match = re.search(r"Run_(\d+)", filename)  # Regex to find "Run_XXX"
        
        if run_number_match:
            run_number = run_number_match.group(1)  # Extract the run number
        else:
            print("Error: Run number not found in filename.")
            return

        save_filename = f"Run_{run_number}_spectrum.txt"

        # Ask the user for a file path to save the data
        save_path = filedialog.asksaveasfilename(
            initialfile=save_filename,
            defaultextension='.txt',
            filetypes=[('Text files', '*.txt'), ('All files', '*.*')]
        )
        
        if not save_path:
            return  # User cancelled

        with open(save_path, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter='	')
            writer.writerow([f"Spectrum (Moving Avg: {moving_avg})"])
            writer.writerow([f"Background removed: {bg_removed}"])
            writer.writerow([f"ROI Rows: {row_begin}-{row_end}"])
            writer.writerow([f"ROI Columns: {column_begin}-{column_end}"])
            writer.writerow([f"BG Rows: {bg_row_begin}-{bg_row_end}"])
            writer.writerow(['X', 'Y'])
            for i in range(len(x_data)):
                writer.writerow([x_data[i], y_data[i]])
                
            print(f"Spectrum saved to {save_path}")
        
    except Exception as e:
        print(f"Error saving spectrum data: {e}")
    
def save_smoothed_spectrum_data(self):
    """Backward-compatible alias for saving the displayed spectrum."""
    return save_spectrum_data(self)

def save_spectrum_image(self):
    """Save the current spectrum plot as an image file."""
    if not hasattr(self, 'last_spectrum_roi'):
        print("Error: Plot a spectrum first")
        return

    try:
        filename = self.original_filepath.split("/")[-1]
        run_number_match = re.search(r"Run_(\d+)", filename)

        if run_number_match:
            run_number = run_number_match.group(1)
        else:
            print("Error: Run number not found in filename.")
            return

        save_filename = f"Run_{run_number}_spectrum.png"
        save_path = filedialog.asksaveasfilename(
            initialfile=save_filename,
            defaultextension='.png',
            filetypes=[
                ('PNG files', '*.png'),
                ('PDF files', '*.pdf'),
                ('SVG files', '*.svg'),
                ('JPEG files', '*.jpg;*.jpeg'),
                ('All files', '*.*'),
            ]
        )

        if not save_path:
            return

        self.fig_spectrum.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Spectrum image saved to {save_path}")

    except Exception as e:
        print(f"Error saving spectrum image: {e}")
