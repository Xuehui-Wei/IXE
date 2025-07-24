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
            'row_begin': 556,
            'row_end': 575,
            'column_begin': 320,
            'column_end': 618,
            'bg_row_begin': 533,
            'bg_row_end': 543,
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
        ttk.Button(frame, text="Process", command=self.process_image).grid(row=2, column=9, columnspan=3,padx=5, pady=5)
        ttk.Button(frame, text="Save", command=self.save_processed_image).grid(row=2, column=13, padx=5, pady=5)

    def setup_spectrum_controls(self):
        """Spectrum analysis controls"""
        frame = ttk.Frame(self.spectrum_tab)
        frame.pack(fill=tk.X, padx=5, pady=5)
        frame.grid_columnconfigure(0, weight=0)  
        frame.grid_columnconfigure(1, weight=0)  
        frame.grid_columnconfigure(2, weight=0) 
        frame.grid_columnconfigure(3, weight=0) 
        frame.grid_columnconfigure(4, weight=0) 
        frame.grid_columnconfigure(5, weight=0)  
        frame.grid_columnconfigure(6, weight=0) 
        frame.grid_columnconfigure(7, weight=0) 
        frame.grid_columnconfigure(8, weight=0)
        frame.grid_columnconfigure(9, weight=0)
        frame.grid_columnconfigure(10, weight=0)
        # ROI controls (still in the same row)
        ttk.Label(frame, text="ROI Rows:").grid(row=0, column=0, padx=5)
        self.row_begin = ttk.Entry(frame, width=4)
        self.row_begin.insert(0, str(self.parm['row_begin']))
        self.row_begin.grid(row=0, column=1, padx=5)
        ttk.Label(frame, text="--").grid(row=0, column=2)
        self.row_end = ttk.Entry(frame, width=4)
        self.row_end.insert(0, str(self.parm['row_end']))
        self.row_end.grid(row=0, column=3, padx=5)
        
        ttk.Label(frame, text="Cols:").grid(row=0, column=4, padx=5)
        self.column_begin = ttk.Entry(frame, width=4)
        self.column_begin.insert(0, str(self.parm['column_begin']))
        self.column_begin.grid(row=0, column=5, padx=5)
        ttk.Label(frame, text="--").grid(row=0, column=6)
        self.column_end = ttk.Entry(frame, width=4)
        self.column_end.insert(0, str(self.parm['column_end']))
        self.column_end.grid(row=0, column=7, padx=5)
        ttk.Button(frame, text="Draw Lines", command=self.draw_lines).grid(row=0, column=8, padx=5, pady=5)
        
        ##second row - Background rows
        ttk.Label(frame, text="BG Rows:").grid(row=1, column=0, padx=5)
        self.bg_row_begin = ttk.Entry(frame, width=4)
        self.bg_row_begin.insert(0, str(self.parm['bg_row_begin']))
        self.bg_row_begin.grid(row=1, column=1, padx=5)
        ttk.Label(frame, text="--").grid(row=1, column=2)
        self.bg_row_end = ttk.Entry(frame, width=4)
        self.bg_row_end.insert(0, str(self.parm['bg_row_end']))
        self.bg_row_end.grid(row=1, column = 3, padx=5)
        # Spectrum buttons - First row of buttons (processing, plot, etc.)
        self.bg_toggle = ttk.Button(frame, text="BG Remove", command=self.toggle_background_removal)
        self.bg_toggle.grid(row=1, column=4,columnspan=2, padx=5, pady=5)
        # Line color picker button (Second row of buttons)
        ttk.Label(frame, text="Line Color:").grid(row=2, column=0, padx=5)
        self.line_color = ttk.Combobox(frame, values=["blue", "red", "green", "black", "orange"], width=5)
        self.line_color.set("black")  # Default color
        self.line_color.grid(row=2, column=1, padx=5)

        # Color Picker Button
        ttk.Button(frame, text="Pick Color", command=self.open_color_picker).grid(row=2, column=8, padx=5)

        # Line style control
        ttk.Label(frame, text="Line Style:").grid(row=2, column=3, padx=5)
        self.line_style = ttk.Combobox(frame, values=["-", "--", "-.", ":"], width=2)
        self.line_style.set("-")  # Default line style
        self.line_style.grid(row=2, column=4, padx=5)
        self.line_style.bind("<<ComboboxSelected>>", lambda event: self.update_plot)
        # Line width control
        ttk.Label(frame, text="Line Width:").grid(row=2, column=5,columnspan=2, padx=5)
        self.line_width = ttk.Entry(frame, width=4)
        self.line_width.insert(0, "0.5")  # Default line width
        self.line_width.grid(row=2, column=7, padx=5)
        self.line_width.bind("<KeyRelease>", lambda event: self.update_plot)
        ttk.Button(frame, text="Plot📈", command=self.plot_roi_spectrum).grid(row=2, column=9, padx=5, pady=5)
        # Add a control for moving average, 4th row
        ttk.Label(frame, text="Smooth (n):").grid(row=3, column=0, padx=5)
        self.n_moveavg = ttk.Entry(frame, width=3)
        self.n_moveavg.insert(0, str(self.parm['n_moveavg']))  # Default window size for moving average
        self.n_moveavg.grid(row=3, column=1, padx=5)

        # Add a button to apply the moving average
        ttk.Button(frame, text="Apply Smooth", command=self.apply_moving_average).grid(row=3, column=3,columnspan=3, padx=5)
        ttk.Button(frame, text="Save Line", command=self.save_spectrum_data).grid(row=3, column=6, columnspan=2,padx=5, pady=5)
        ttk.Button(frame, text="Save Smoothed Line", command=self.save_smoothed_spectrum_data).grid(row=3, column=8,columnspan=2, padx=5, pady=5)

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
        # Higher DPI for better quality
        self.fig_dpi = 300
        
        # Single frame for images (raw or processed)
        self.image_frame = ttk.LabelFrame(self.image_panel, text="Image Display", padding=(10, 10), width=430, height=350)
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.image_frame.pack_propagate(False)
        self.run_number_label = ttk.Label(self.image_frame, text="Run: Not Loaded", font=("Arial", 16))
        self.run_number_label.pack(side=tk.TOP, padx=5)
        # Smaller figure size for image display
        self.fig_image = plt.Figure(figsize=(6, 5), dpi=self.fig_dpi)
        self.ax_image = self.fig_image.add_subplot(111)
        # Hide ticks and ticklabels for image axes initially
        self.ax_image.set_xticks([])
        self.ax_image.set_yticks([])
        self.ax_image.set_xticklabels([])
        self.ax_image.set_yticklabels([])
        for spine in self.ax_image.spines.values():
            spine.set_linewidth(0.5)
        self.canvas_image = FigureCanvasTkAgg(self.fig_image, master=self.image_frame)
        self.canvas_image.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
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
            'font.size': 7,            # Base size increased
            'axes.titlesize': 4.5,       # Larger title
            'axes.labelsize': 4,       # Larger axis labels
            'xtick.labelsize': 4,      # Larger x-ticks
            'ytick.labelsize': 4,      # Larger y-ticks
            'legend.fontsize': 4,      # Larger legend
            'figure.titlesize': 4      # Largest figure title
        })
        
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
    def save_smoothed_spectrum_data(self):
        return spectrum_controls.save_smoothed_spectrum_data(self)

def run_gui():
    root = tk.Tk()
    app = TIFFAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    run_gui()  
