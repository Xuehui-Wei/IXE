# -*- coding: utf-8 -*-
import numpy as np
import re
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.colorchooser import askcolor

def import_ref_data(self):
    """Import the reference spectrum data from a .txt or .chi file and plot it in the spectrum plot."""
    ref_file = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("CHI files", "*.chi"), ("All files", "*.*")])
    if ref_file:
        file_extension = ref_file.split('.')[-1].lower()
        x_data = []
        y_data = []
        try:
            if file_extension == 'txt':
                with open(ref_file, mode='r') as file:
                    next(file)
                    for line in file:
                        values = line.strip().split('	')
                        if len(values) == 2:
                            try:
                                x_data.append(float(values[0]))
                                y_data.append(float(values[1]))
                            except ValueError:
                                print(f"Skipping invalid data: {line}")
            elif file_extension == 'chi':
                with open(ref_file, mode='r') as file:
                    for line in file:
                        if not line.startswith('#'):
                            values = line.strip().split()
                            if len(values) == 2:
                                try:
                                    x_data.append(float(values[0]))
                                    y_data.append(float(values[1]))
                                except ValueError:
                                    print(f"Skipping invalid data: {line}")
            y_data_norm = self.spect_processor.get_norm_spectrum(np.array(y_data))
            line_color = self.spect_processor.random_color()
            ref_filename = ref_file.split("/")[-1]
            ref_run_number_match = re.search(r"Run_(\d+)", ref_filename)
            if ref_run_number_match:
                ref_run_number = ref_run_number_match.group(1)
            else:
                ref_run_number = "?"
            with open(ref_file, mode='r') as file:
                header = file.readline().strip()
                if 'Moving Avg' in header:
                    n_moveavg = re.search(r"Moving Avg: (\d+)", header)
                    if n_moveavg:
                        n_moveavg_value = n_moveavg.group(1)
                        label = f"Run {ref_run_number}, n_movavg = {n_moveavg_value}"
                        line_name = f"Run {ref_run_number}, n_movavg = {n_moveavg_value}"
                    else:
                        label = f"Run {ref_run_number}"
                        line_name = f"Run {ref_run_number}"
                else:
                    label = f"Run {ref_run_number}"
                    line_name = f"Run {ref_run_number}"
            self.ax_spectrum.plot(x_data, y_data_norm, label=label, color=line_color, linestyle='-', linewidth=0.5)
            self.ax_spectrum.set_yticks([])  # Removes y-axis ticks
            self.ax_spectrum.set_yticklabels([])  # Removes y-axis labels
            self.plotted_lines[line_name] = {
                'x_data': x_data,
                'y_data': y_data_norm,
                'color': line_color,
                'visible': True,
                'checkbox': None,
                'color_circle': None,
                'is_ref': True,
            }
            self.add_line_to_list(line_name, line_color)
            self.ax_spectrum.legend(fontsize = 4)
            self.fig_spectrum.tight_layout()
            self.canvas_spectrum.draw()
        except Exception as e:
            print(f"Error importing reference data: {e}")

def remove_selected_line(self):
    """Remove the selected reference line and its UI row (checkbox, color, IAD label/entry) from the plot and list."""
    selected_line = self.get_selected_line_name()
    if selected_line:
        if self.plotted_lines[selected_line].get('is_ref', False):
            widgets = ['checkbox', 'color_circle', 'iad_label', 'iad_entry', 'st_iad_label', 'st_iad_entry',]
            for w in widgets:
                widget = self.plotted_lines[selected_line].get(w)
                if widget is not None:
                    widget.destroy()
            del self.plotted_lines[selected_line]
            self.ax_spectrum.clear()
            for line_name, line_data in self.plotted_lines.items():
                if line_data['visible']:
                    self.ax_spectrum.plot(line_data['x_data'], line_data['y_data'], label=line_name, color=line_data['color'], linestyle='-', linewidth=0.5)
            self.ax_spectrum.set_yticks([])  # Removes y-axis ticks
            self.ax_spectrum.set_yticklabels([])  # Removes y-axis labels
            self.ax_spectrum.legend(fontsize = 4)
            self.fig_spectrum.tight_layout()
            self.canvas_spectrum.draw()
        else:
            print("Cannot remove the spectrum under processing!")

def change_selected_line_color(self):
    """Change the color of the selected line(s)"""
    selected_line = self.get_selected_line_name()
    if selected_line:
        new_color = askcolor()[1]
        if new_color:
            self.plotted_lines[selected_line]['color'] = new_color
            color_circle = self.plotted_lines[selected_line]['color_circle']
            color_circle.config(bg=new_color)
            if self.plotted_lines[selected_line]['visible']:
                for line in self.ax_spectrum.lines:
                    if line.get_label() == selected_line:
                        line.remove()
                self.ax_spectrum.plot(self.plotted_lines[selected_line]['x_data'], self.plotted_lines[selected_line]['y_data'], label=selected_line, color=new_color, linestyle='-', linewidth=0.5)
            self.ax_spectrum.set_yticks([])  # Removes y-axis ticks
            self.ax_spectrum.set_yticklabels([])  # Removes y-axis labels
            self.fig_spectrum.tight_layout()
            self.canvas_spectrum.draw()

def get_selected_line_name(self):
    """Retrieve the name of the selected line from the checkbox"""
    for line_name, line_data in self.plotted_lines.items():
        if line_data['checkbox_var'].get():
            return line_name
    return None

def update_line_list(self):
    """Update the line list UI (checkboxes, color circles, labels)."""
    for widget in self.line_list_frame.winfo_children():
        widget.destroy()
    for line_name in self.plotted_lines:
        line_data = self.plotted_lines[line_name]
        label = line_name
        color = line_data['color']
        row = tk.Frame(self.line_list_frame)
        row.pack(fill=tk.X, padx=5, pady=5)
        checkbox_var = tk.BooleanVar(value=line_data['visible'])
        checkbox = tk.Checkbutton(row, text=label, variable=checkbox_var, command=lambda line_name=line_name, checkbox_var=checkbox_var: self.toggle_line_visibility(line_name, checkbox_var))
        checkbox.pack(side=tk.LEFT)
        color_circle = tk.Label(row, text="  ", width=3, background=color)
        color_circle.pack(side=tk.LEFT, padx=5)
        if hasattr(self, 'selected_line') and line_name == self.selected_line:
            checkbox.config(font=('Arial', 10, 'bold'))
        self.plotted_lines[line_name]['checkbox'] = checkbox
        self.plotted_lines[line_name]['checkbox_var'] = checkbox_var
        self.plotted_lines[line_name]['color_circle'] = color_circle

def add_line_to_list(self, line_name, line_color):
    row_idx = self.line_list_frame.grid_size()[1]  # Get the current row index
    
    # Create the checkbox for the line and align it left
    checkbox_var = tk.BooleanVar(value=True)
    checkbox = tk.Checkbutton(self.line_list_frame, text=line_name, variable=checkbox_var,
                              command=lambda: self.toggle_line_visibility(line_name, checkbox_var))
    checkbox.grid(row=row_idx, column=0, sticky='w', padx=5)  # Place it in the first column
    
    # Create the color circle next to the line name
    color_circle = tk.Label(self.line_list_frame, width=1, height=0, bg=line_color)
    color_circle.grid(row=row_idx, column=1, padx=5)  # Place it in the second column
    
    # Add the IAD label and entry for the first IAD
    iad_label = tk.Label(self.line_list_frame, text="IAD =", font=("Arial", 14))
    iad_label.grid(row=row_idx, column=2, padx=2, sticky='w')  # Align the label to the left
    iad_entry = tk.Entry(self.line_list_frame, width=12, font=("Arial", 14))
    iad_entry.grid(row=row_idx, column=3, padx=5)  # Place the entry next to the IAD label
    iad_entry.insert(0, "")
    
    # Add the second IAD label and entry for Satellite IAD
    st_iad_label = tk.Label(self.line_list_frame, text="Satellite IAD =", font=("Arial", 14))
    st_iad_label.grid(row=row_idx, column=4, padx=2, sticky='w')  # Align the label to the left
    st_iad_entry = tk.Entry(self.line_list_frame, width=12, font=("Arial", 14))
    st_iad_entry.grid(row=row_idx, column=5, padx=5)  # Place the entry next to the Satellite IAD label
    st_iad_entry.insert(0, "")

    # Store all the elements in the plotted_lines dictionary
    self.plotted_lines[line_name]['checkbox'] = checkbox
    self.plotted_lines[line_name]['checkbox_var'] = checkbox_var
    self.plotted_lines[line_name]['color_circle'] = color_circle
    self.plotted_lines[line_name]['iad_entry'] = iad_entry
    self.plotted_lines[line_name]['iad_label'] = iad_label
    self.plotted_lines[line_name]['st_iad_entry'] = st_iad_entry
    self.plotted_lines[line_name]['st_iad_label'] = st_iad_label


def toggle_line_visibility(self, line_name, checkbox_var):
    if checkbox_var.get():
        if not self.plotted_lines[line_name]['visible']:
            self.plotted_lines[line_name]['visible'] = True
            self.ax_spectrum.plot(self.plotted_lines[line_name]['x_data'], self.plotted_lines[line_name]['y_data'], label=line_name, color=self.plotted_lines[line_name]['color'], linestyle='-', linewidth=0.5)
    else:
        self.plotted_lines[line_name]['visible'] = False
        for line in list(self.ax_spectrum.lines):
            if line.get_label() == line_name:
                line.remove()
        if self.plotted_lines[line_name].get('is_ref', False):
            self.ax_spectrum.clear()
            if hasattr(self, 'last_spectrum_roi'):
                roi_y = np.array(self.last_spectrum_roi)
                roi_color = self.line_color.get() if hasattr(self, 'line_color') else 'black'
                try:
                    line_width = float(self.line_width.get())
                except Exception:
                    line_width = 0.5
                self.ax_spectrum.plot(roi_y, color=roi_color, linestyle=self.line_style.get(), linewidth=line_width, label="ROI")
            self.ax_spectrum.set_title("ROI Spectrum")
            self.ax_spectrum.set_xlabel("Column Index")
            self.ax_spectrum.set_ylabel("Intensity")
            self.ax_spectrum.set_yticks([])  # Removes y-axis ticks
            self.ax_spectrum.set_yticklabels([])  # Removes y-axis labels
            self.ax_spectrum.legend(fontsize=4)
            self.fig_spectrum.tight_layout()
            self.canvas_spectrum.draw()
            return
    self.ax_spectrum.legend(fontsize=4)
    self.fig_spectrum.tight_layout()
    self.canvas_spectrum.draw()

def plot_integrated_diff(self):
    ref_line = None
    for name, data in self.plotted_lines.items():
        if data.get('is_ref', False) and data['checkbox_var'].get():
            ref_line = data
            break
    if ref_line is None:
        messagebox.showwarning("Reminder", "No reference spectrum selected.")
        return
    if not hasattr(self, 'last_spectrum_roi'):
        print("No ROI spectrum to compare. Please plot ROI or apply moving average first.")
        return
    roi_y = np.array(self.last_spectrum_roi)
    ref_y = np.array(ref_line['y_data'])
    ref_x = np.array(ref_line['x_data'])
    if len(roi_y) != len(ref_y) or len(ref_y) != len(ref_x):
        messagebox.showwarning("Error", "ROI spectrum and reference spectrum lengths do not match. Cannot compare.")
        print("Error: ROI spectrum and reference spectrum lengths do not match.")
        return
    lw = 1
    roi_color = self.line_color.get() if hasattr(self, 'line_color') else 'black'
    self.ax_spectrum.clear()
    self.ax_spectrum.plot(ref_x, ref_y, color=ref_line['color'], linewidth=0.5, label="REF")
    self.ax_spectrum.plot(ref_x, roi_y, color=roi_color, linewidth=0.5, label="ROI")
    self.ax_spectrum.fill_between(ref_x, 0, roi_y - ref_y, color=ref_line['color'], alpha=0.3, label="Integrated Diff.")
    self.ax_spectrum.legend(fontsize=4)
    self.ax_spectrum.set_title("Integrated Difference (ROI - Reference)")
    self.ax_spectrum.set_xlabel("Column Index")
    self.ax_spectrum.set_ylabel("Intensity")
    self.ax_spectrum.set_yticks([])  # Removes y-axis ticks
    self.ax_spectrum.set_yticklabels([])  # Removes y-axis labels
    self.fig_spectrum.tight_layout()
    self.canvas_spectrum.draw()
    iad_value = np.sum(np.abs(roi_y - ref_y))
    for name, data in self.plotted_lines.items():
        if data.get('is_ref', False) and data['checkbox_var'].get():
            if 'iad_entry' in data:
                data['iad_entry'].delete(0, 'end')
                data['iad_entry'].insert(0, f"{iad_value:.4f}")
    if hasattr(self, 'iad_value_label'):
        self.iad_value_label.config(text=f"IAD = {iad_value:.4f}")

def calculate_and_display_satellite_peak_iad(self):
    ref_line = None
    for name, data in self.plotted_lines.items():
        if data.get('is_ref', False) and data['checkbox_var'].get():
            ref_line = data
            break
    if ref_line is None:
        messagebox.showwarning("Reminder", "No reference spectrum selected.")
        return
    if not hasattr(self, 'last_spectrum_roi'):
        print("No ROI spectrum to compare. Please plot ROI or apply moving average first.")
        return
    roi_y = np.array(self.last_spectrum_roi)
    ref_y = np.array(ref_line['y_data'])
    ref_x = np.array(ref_line['x_data'])
    if len(roi_y) != len(ref_y) or len(ref_y) != len(ref_x):
        messagebox.showwarning("Error", "ROI spectrum and reference spectrum lengths do not match. Cannot compare.")
        print("Error: ROI spectrum and reference spectrum lengths do not match.")
        return
    cross_begin = int(self.cross_begin.get())
    cross_end = int(self.cross_end.get())
    eyeball_str = self.eye_ball_cross.get().strip()
    try:
        eyeball_point = int(eyeball_str) if eyeball_str else None
    except ValueError:
        eyeball_point = None
    iad_satellite, transition_point, spec_align, spec_r_align = self.spect_processor.calculate_satellite_peak_iad(roi_y, ref_y, cross_begin, cross_end, eyeball_point)
    self.ax_spectrum.clear()
    roi_color = self.line_color.get() if hasattr(self, 'line_color') else 'black'
    self.ax_spectrum.clear()
    self.ax_spectrum.plot(spec_r_align, color=ref_line['color'], linewidth=0.5, label="REF")
    self.ax_spectrum.plot(spec_align, color=roi_color, linewidth=0.5, label="ROI")
    self.ax_spectrum.fill_between(np.arange(len(spec_align))[:transition_point], 0, spec_align[:transition_point] -spec_r_align[:transition_point], color=ref_line['color'], alpha=0.3, label="Satellite Diff.")
    self.ax_spectrum.legend(fontsize=4)
    self.ax_spectrum.set_title("Integrated Difference (ROI - Reference)")
    self.ax_spectrum.set_xlabel("Column Index")
    self.ax_spectrum.set_ylabel("Intensity")
    self.ax_spectrum.set_yticks([])  # Removes y-axis ticks
    self.ax_spectrum.set_yticklabels([])  # Removes y-axis labels
    self.fig_spectrum.tight_layout()
    self.canvas_spectrum.draw()
    for name, data in self.plotted_lines.items():
        if data.get('is_ref', False) and data['checkbox_var'].get():
            if 'st_iad_entry' in data:
                data['st_iad_entry'].delete(0, 'end')
                data['st_iad_entry'].insert(0, f"{iad_satellite:.4f}")
    if hasattr(self, 'sat_iad_value_label'):
        self.sat_iad_value_label.config(text=f"Satellite IAD = {iad_satellite:.4f}")

def pick_ref_color(self):
    color = askcolor()[1]
    if color:
        self.ref_spectrum_color = color
# iad_controls.py
# IAD and satellite peak calculation, reference data import, and line management for TIFFAnalyzer.

# All functions will be moved here unchanged in the next step.
