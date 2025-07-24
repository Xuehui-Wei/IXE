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
        # Clear previous spectrum plot
        self.ax_spectrum.clear()

        # Get spectrum data
        spectrum = self.spect_processor.get_spectrum(
            int(self.row_begin.get()),
            int(self.row_end.get()),
            int(self.column_begin.get()),
            int(self.column_end.get())
        )
        # Normalize the spectrum before plotting
        norm_spectrum = self.spect_processor.get_norm_spectrum(spectrum)

        # Store the last displayed ROI spectrum (normalized, raw)
        self.last_spectrum_roi = norm_spectrum.copy()

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
        title = "ROI Spectrum" + (" (BG Subtracted)" if self.bg_subtraction_enabled else "")
        self.ax_spectrum.set_title(title)
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
        # Update the n_moveavg in the self.parm dictionary
        self.parm['n_moveavg'] = n_moveavg

        # Re-initialize the SpectrumProcessor with the updated n_moveavg value
        self.spect_processor = self.spect_processor.__class__(self.immm, self.parm['n_moveavg'])

        # Get the spectrum data from the selected ROI
        spectrum = self.spect_processor.get_spectrum(
            int(self.row_begin.get()),
            int(self.row_end.get()),
            int(self.column_begin.get()),
            int(self.column_end.get())
        )

        # Apply the moving average to the spectrum data
        smoothed_spectrum = self.spect_processor.moving_average(spectrum)
        norm_smoothed_spectrum = self.spect_processor.get_norm_spectrum(smoothed_spectrum)

        # Store the last displayed ROI spectrum (normalized, smoothed)
        self.last_spectrum_roi = norm_smoothed_spectrum.copy()

        # Plot the smoothed spectrum in the spectrum axes
        self.ax_spectrum.clear()  # Clear the previous plot
        self.ax_spectrum.plot(norm_smoothed_spectrum, color=self.line_color.get(), linestyle=self.line_style.get(), linewidth=float(self.line_width.get()), 
                              label=f"Run {self.run_number_label.cget('text').split(': ')[1]}")
        self.ax_spectrum.legend()
        self.ax_spectrum.set_title("Smoothed ROI Spectrum")
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
        bg_row_start = int(self.bg_row_begin.get())
        bg_row_end = int(self.bg_row_end.get())
        bg_col_start = int(self.column_begin.get())
        bg_col_end = int(self.column_end.get())
        if (bg_row_start >= bg_row_end or bg_col_start >= bg_col_end or bg_row_start < 0 or bg_col_start < 0):
            raise ValueError("Invalid background ROI coordinates")
        self.spect_processor.set_background_roi(bg_row_start, bg_row_end, bg_col_start, bg_col_end)
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
    """Save the displayed spectrum data (x, y) to a CSV file with customized name."""
    if not hasattr(self, 'spect_processor'):
        print("Error: Process the image first")
        return
    
    try:
        # Get the spectrum data from the selected ROI
        spectrum = self.spect_processor.get_spectrum(
            int(self.row_begin.get()),
            int(self.row_end.get()),
            int(self.column_begin.get()),
            int(self.column_end.get())
        )
        # Prepare the x (column indices) and y (spectrum intensities) data
        x_data = np.arange(len(spectrum))  # Column indices (x axis)
        y_data = spectrum  # Spectrum intensities (y axis)
        
        # Get the input filename and extract the run number
        filename = self.original_filepath.split("/")[-1]  # Extract filename from path
        run_number_match = re.search(r"Run_(\d+)", filename)  # Regex to find "Run_XXX"
        
        if run_number_match:
            run_number = run_number_match.group(1)  # Extract the run number
        else:
            print("Error: Run number not found in filename.")
            return

        # Start building the new filename for saving
        save_filename = f"Run_{run_number}_spectrum"
        
        # Append 'rmbg' if background removal has been applied
        if self.bg_subtraction_enabled:
            save_filename += "_rmbg"
        
        # Append moving average information if applied
        #if hasattr(self, 'spect_processor') and self.spect_processor.n_moveavg != 5:  # Check if n_moveavg is updated
            #save_filename += f"_movavg_{self.spect_processor.n_moveavg}n"
        
        save_filename += ".txt"  # Add .txt or .chi extension

        # Ask the user for a file path to save the data
        save_path = filedialog.asksaveasfilename(
            initialfile=save_filename,  # Use the generated filename
            defaultextension='.txt',
            filetypes=[('Text files', '*.txt'), ('CHI files', '*.chi'), ('All files', '*.*')]
        )
        
        if not save_path:
            return  # User cancelled

        # Save the data as a CSV file
        with open(save_path, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter='	')  # Tab-separated values
            writer.writerow([f"Spectrum (Moving Avg: 0)"])
            writer.writerow(['X', 'Y'])  # Write header for data columns
            for i in range(len(x_data)):
                writer.writerow([x_data[i], y_data[i]])  # Write each (x, y) pair of the smoothed data
                
            print(f"Data saved to {save_path}")
        
    except Exception as e:
        print(f"Error saving spectrum data: {e}")
    
def save_smoothed_spectrum_data(self):
    """Save the smoothed spectrum data (x, y) to a .txt or .chi file."""
    if not hasattr(self, 'spect_processor'):
        print("Error: Process the image first")
        return
    
    try:
        # Get the smoothed spectrum data (after moving average is applied)
        spectrum = self.spect_processor.get_spectrum(
            int(self.row_begin.get()),
            int(self.row_end.get()),
            int(self.column_begin.get()),
            int(self.column_end.get())
        )
        
        # Apply moving average to the spectrum data
        smoothed_spectrum = self.spect_processor.moving_average(spectrum)
        
        # Prepare the x (column indices) and y (smoothed spectrum intensities) data
        x_data = np.arange(len(smoothed_spectrum))  # Column indices (x axis)
        y_data = smoothed_spectrum  # Smoothed spectrum intensities (y axis)

        # Get the input filename and extract the run number
        filename = self.original_filepath.split("/")[-1]  # Extract filename from path
        run_number_match = re.search(r"Run_(\d+)", filename)  # Regex to find "Run_XXX"
        
        if run_number_match:
            run_number = run_number_match.group(1)  # Extract the run number
        else:
            print("Error: Run number not found in filename.")
            return

        # Start building the new filename for saving
        save_filename = f"Run_{run_number}_spectrum_movavg_{self.spect_processor.n_moveavg}n"
        save_filename += ".txt"  # Add .txt or .chi extension

        # Ask the user for a file path to save the data
        save_path = filedialog.asksaveasfilename(
            initialfile=save_filename,  # Use the generated filename
            defaultextension='.txt',
            filetypes=[('Text files', '*.txt'), ('CHI files', '*.chi'), ('All files', '*.*')]
        )
        
        if not save_path:
            return  # User cancelled

        # Save the smoothed data (moving average applied) as a text file
        with open(save_path, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter='	')  # Tab-separated values
            writer.writerow([f"Smoothed Spectrum (Moving Avg: {self.spect_processor.n_moveavg})"])
            writer.writerow(['X', 'Y'])  # Write header for data columns
            for i in range(len(x_data)):
                writer.writerow([x_data[i], y_data[i]])  # Write each (x, y) pair of the smoothed data
                
            print(f"Smoothed Data saved to {save_path}")

    except Exception as e:
        print(f"Error saving smoothed spectrum data: {e}")
