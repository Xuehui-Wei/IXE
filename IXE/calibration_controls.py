# -*- coding: utf-8 -*-
"""Energy calibration controls for IXE spectra."""

import csv
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, messagebox

try:
    from .peak_fitting import PeakFitConfig, PeakFitError, fit_spectrum
    from . import spectrum_controls as _spectrum_view
except ImportError:
    try:
        from peak_fitting import PeakFitConfig, PeakFitError, fit_spectrum
        import spectrum_controls as _spectrum_view
    except ImportError:
        from IXE.peak_fitting import PeakFitConfig, PeakFitError, fit_spectrum
        from IXE import spectrum_controls as _spectrum_view


def _split_spectrum_row(line):
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return []
    if "," in stripped:
        return [value.strip() for value in next(csv.reader([stripped]))]
    if "\t" in stripped:
        return [value.strip() for value in stripped.split("\t")]
    return stripped.split()


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _header_name(value):
    return re.sub(r"\s+", " ", value.strip().lower())


def _preferred_y_column(headers):
    normalized = [_header_name(value) for value in headers]
    for name in ("peak fit", "y", "intensity", "raw y"):
        if name in normalized:
            return normalized.index(name)
    return 1 if len(headers) > 1 else None


def _is_spectrum_table_header(row):
    normalized = [_header_name(value) for value in row]
    if not normalized or normalized[0] not in ("x", "energy"):
        return False
    return any(name in normalized for name in ("peak fit", "y", "intensity", "raw y"))


def _parse_calibration_spectrum(filepath):
    with open(filepath, mode="r") as file:
        rows = [_split_spectrum_row(line) for line in file]

    x_data = []
    y_data = []
    source_column = "Y"
    for row_index, row in enumerate(rows):
        if not row or not _is_spectrum_table_header(row):
            continue
        x_column = 0
        y_column = _preferred_y_column(row)
        if y_column is None or y_column >= len(row) or y_column == x_column:
            continue
        source_column = row[y_column]
        for data_row in rows[row_index + 1:]:
            if len(data_row) <= max(x_column, y_column):
                continue
            x_value = _as_float(data_row[x_column])
            y_value = _as_float(data_row[y_column])
            if x_value is not None and y_value is not None:
                x_data.append(x_value)
                y_data.append(y_value)
        if y_data:
            break

    if not y_data:
        for row in rows:
            if len(row) < 2:
                continue
            x_value = _as_float(row[0])
            y_value = _as_float(row[1])
            if x_value is not None and y_value is not None:
                x_data.append(x_value)
                y_data.append(y_value)

    if len(y_data) < 8:
        raise ValueError("No usable two-column calibration spectrum was found.")

    x = np.asarray(x_data, dtype=float)
    y = np.asarray(y_data, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 8:
        raise ValueError("Calibration spectrum has too few finite points.")
    order = np.argsort(x)
    return x[order], y[order], source_column


def _entry_float(entry_widget):
    if entry_widget is None:
        return None
    text = entry_widget.get().strip()
    if not text:
        return None
    return float(text)


def _set_var(var, value, precision=6):
    if var is None:
        return
    if value is None or not np.isfinite(value):
        var.set("")
    else:
        var.set(f"{float(value):.{precision}g}")


def _component_scale(fit_result):
    best_fit = np.asarray(fit_result.best_fit, dtype=float)
    profile = np.asarray(fit_result.normalized_fit, dtype=float)
    raw_total = np.nansum(best_fit - np.nanmin(best_fit))
    profile_total = np.nansum(profile - np.nanmin(profile))
    if (
        not np.isfinite(raw_total)
        or not np.isfinite(profile_total)
        or abs(raw_total) <= np.finfo(float).eps
        or abs(profile_total) <= np.finfo(float).eps
    ):
        return None
    return raw_total / profile_total


def _fit_config_from_panel(self):
    try:
        x_min = _entry_float(getattr(self, "cal_fit_start", None))
        x_max = _entry_float(getattr(self, "cal_fit_end", None))
    except ValueError:
        messagebox.showwarning("Calibration", "Calibration fit range must be numeric.")
        return None
    if x_min is not None and x_max is not None and x_min > x_max:
        x_min, x_max = x_max, x_min
    try:
        return PeakFitConfig(n_peaks=3, x_min=x_min, x_max=x_max)
    except PeakFitError as exc:
        messagebox.showwarning("Calibration", str(exc))
        return None


def _centers3(fit_result):
    centers = np.asarray(getattr(fit_result, "centers", []), dtype=float)
    if centers.size < 3 or not np.all(np.isfinite(centers[:3])):
        raise ValueError("The 3-peak fit did not return valid satellite/res/main centers.")
    return float(centers[0]), float(centers[1]), float(centers[2])


def _update_fit_output(self, prefix, centers):
    sat, res, main = centers
    if prefix == "cal":
        _set_var(getattr(self, "cal_sat_energy_var", None), sat)
        _set_var(getattr(self, "cal_res_energy_var", None), res)
        _set_var(getattr(self, "cal_main_energy_var", None), main)
    else:
        _set_var(getattr(self, "cal_sat_pixel_var", None), sat)
        _set_var(getattr(self, "cal_res_pixel_var", None), res)
        _set_var(getattr(self, "cal_main_pixel_var", None), main)


def import_calibration(self):
    """Import a two-column energy/intensity calibrant spectrum."""
    filepath = filedialog.askopenfilename()
    if not filepath:
        return
    try:
        x_data, y_data, source_column = _parse_calibration_spectrum(filepath)
    except Exception as exc:
        messagebox.showwarning("Import Cal.", str(exc))
        print(f"Error importing calibration spectrum: {exc}")
        return

    self.calibration_filepath = filepath
    self.calibration_x = x_data
    self.calibration_y = y_data
    self.calibration_source_column = source_column
    if hasattr(self, "calibration_file_label"):
        self.calibration_file_label.config(text=os.path.basename(filepath))
    if hasattr(self, "cal_status_var"):
        self.cal_status_var.set(f"Imported {x_data.size} points ({source_column})")
    for attr_name in ("calibration_fit", "energy_calibration"):
        if hasattr(self, attr_name):
            delattr(self, attr_name)
    _update_fit_output(self, "cal", (np.nan, np.nan, np.nan))
    _set_var(getattr(self, "cal_slope_var", None), np.nan)
    _set_var(getattr(self, "cal_intercept_var", None), np.nan)
    _set_var(getattr(self, "cal_residual_var", None), np.nan)
    if hasattr(self, "ax_spectrum"):
        self.ax_spectrum.clear()
        self.ax_spectrum.plot(x_data, y_data, color="#8db7ff", linewidth=1.0, label=f"Cal, {os.path.basename(filepath)}")
        if hasattr(_spectrum_view, "_finish_spectrum_axes"):
            _spectrum_view._finish_spectrum_axes(self, xlabel="Energy", ylabel="Intensity", view_mode="calibration", view_label="Calibration")
        elif hasattr(self, "canvas_spectrum"):
            self.canvas_spectrum.draw_idle()


def fit_calibration_spectrum(self):
    """Fit the imported calibrant spectrum with the K-beta 3-peak model."""
    if not hasattr(self, "calibration_x") or not hasattr(self, "calibration_y"):
        messagebox.showwarning("Calibration", "Import a calibration spectrum first.")
        return None
    config = _fit_config_from_panel(self)
    if config is None:
        return None
    try:
        fit_result = fit_spectrum(self.calibration_x, self.calibration_y, config)
    except PeakFitError as exc:
        messagebox.showwarning("Calibration", str(exc))
        print(f"Calibration fit failed: {exc}")
        return None

    self.calibration_fit = fit_result
    try:
        centers = _centers3(fit_result)
    except ValueError as exc:
        messagebox.showwarning("Calibration", str(exc))
        return None
    _update_fit_output(self, "cal", centers)
    if hasattr(self, "cal_status_var"):
        self.cal_status_var.set("Calibration spectrum fitted")
    return fit_result


def calculate_energy_calibration(self):
    """Calculate E = slope * pixel + intercept from satellite/main peak centers."""
    if not hasattr(self, "last_peak_fit_profile"):
        messagebox.showwarning("Calibration", "Run Peak Fit on the current spectrum first.")
        return None
    cal_fit = getattr(self, "calibration_fit", None)
    if cal_fit is None:
        cal_fit = fit_calibration_spectrum(self)
    if cal_fit is None:
        return None

    try:
        pix_sat, pix_res, pix_main = _centers3(self.last_peak_fit_profile)
        ene_sat, ene_res, ene_main = _centers3(cal_fit)
    except ValueError as exc:
        messagebox.showwarning("Calibration", str(exc))
        return None

    denom = pix_main - pix_sat
    if abs(denom) <= np.finfo(float).eps:
        messagebox.showwarning("Calibration", "Current satellite and main peak pixel centers are identical.")
        return None

    slope = (ene_main - ene_sat) / denom
    intercept = ene_main - slope * pix_main
    predicted_res = slope * pix_res + intercept
    residual = predicted_res - ene_res

    self.energy_calibration = {
        "slope": float(slope),
        "intercept": float(intercept),
        "pixel_satellite": pix_sat,
        "pixel_res": pix_res,
        "pixel_main": pix_main,
        "energy_satellite": ene_sat,
        "energy_res": ene_res,
        "energy_main": ene_main,
        "predicted_res_energy": float(predicted_res),
        "residual_res_energy": float(residual),
    }
    _update_fit_output(self, "current", (pix_sat, pix_res, pix_main))
    _update_fit_output(self, "cal", (ene_sat, ene_res, ene_main))
    _set_var(getattr(self, "cal_slope_var", None), slope, precision=8)
    _set_var(getattr(self, "cal_intercept_var", None), intercept, precision=8)
    _set_var(getattr(self, "cal_residual_var", None), residual, precision=5)
    if hasattr(self, "cal_status_var"):
        self.cal_status_var.set("Energy calibration calculated")
    return self.energy_calibration


def _energy_axis(calibration, x_values):
    x_values = np.asarray(x_values, dtype=float)
    return calibration["slope"] * x_values + calibration["intercept"]


def _current_run_number(self):
    label = self.run_number_label.cget("text") if hasattr(self, "run_number_label") else ""
    if ": " in label:
        return label.split(": ", 1)[1]
    return "calibrated"


def _normalize_overlay(y_values):
    y_values = np.asarray(y_values, dtype=float)
    if y_values.size == 0:
        return y_values
    finite = np.isfinite(y_values)
    if not np.any(finite):
        return np.zeros_like(y_values)
    y = y_values.copy()
    fill = float(np.nanmedian(y[finite]))
    y = np.nan_to_num(y, nan=fill, posinf=fill, neginf=fill)
    y = y - float(np.nanmin(y))
    max_value = float(np.nanmax(y))
    if max_value <= np.finfo(float).eps:
        return np.zeros_like(y)
    return y / max_value


def show_calibration_fit(self):
    """Plot the imported calibrant data and fitted peak model."""
    fit_result = getattr(self, "calibration_fit", None)
    if fit_result is None:
        fit_result = fit_calibration_spectrum(self)
    if fit_result is None:
        return

    self.ax_spectrum.clear()
    self.ax_spectrum.plot(
        self.calibration_x,
        self.calibration_y,
        color="0.65",
        linewidth=0.8,
        label="Calibrant",
    )
    self.ax_spectrum.plot(
        fit_result.x,
        fit_result.normalized_fit,
        color="black",
        linewidth=1.4,
        label="Cal. fit",
    )
    scale = _component_scale(fit_result)
    if scale is not None:
        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        for index, peak_curve in enumerate(fit_result.peak_curves, start=1):
            label = fit_result.peak_labels[index - 1] if index - 1 < len(fit_result.peak_labels) else f"Peak {index}"
            color = colors[index % len(colors)] if colors else None
            self.ax_spectrum.plot(
                fit_result.x,
                np.asarray(peak_curve, dtype=float) / scale,
                linestyle="--",
                linewidth=0.9,
                color=color,
                alpha=0.9,
                label=label,
            )
    _spectrum_view._finish_spectrum_axes(
        self,
        xlabel="Energy",
        ylabel="Intensity",
        view_mode="calibration",
        view_label="Calibration fit",
    )


def apply_energy_calibration(self):
    """Plot the current spectrum on the calibrated energy axis."""
    calibration = getattr(self, "energy_calibration", None)
    if calibration is None:
        calibration = calculate_energy_calibration(self)
    if calibration is None:
        return
    if not hasattr(self, "last_spectrum_roi"):
        messagebox.showwarning("Calibration", "Plot the current spectrum first.")
        return

    y_data = np.asarray(self.last_spectrum_roi, dtype=float)
    x_pixels = _spectrum_view._current_spectrum_x_axis(self, y_data.size)
    energy = _energy_axis(calibration, x_pixels)

    self.calibration_energy_axis = energy
    self.calibrated_spectrum_y = y_data.copy()
    self.ax_spectrum.clear()
    self.ax_spectrum.plot(energy, y_data, color="0.45", linewidth=0.8, label="Current")

    fit_result = getattr(self, "last_peak_fit_profile", None)
    if fit_result is not None:
        fit_energy = _energy_axis(calibration, fit_result.x)
        self.calibrated_fit_energy = fit_energy
        self.calibrated_fit_y = np.asarray(fit_result.normalized_fit, dtype=float).copy()
        self.ax_spectrum.plot(
            fit_energy,
            fit_result.normalized_fit,
            color="black",
            linewidth=1.5,
            label="Peak fit",
        )
        scale = _component_scale(fit_result)
        if scale is not None:
            colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
            calibrated_components = []
            for index, peak_curve in enumerate(fit_result.peak_curves, start=1):
                label = fit_result.peak_labels[index - 1] if index - 1 < len(fit_result.peak_labels) else f"Peak {index}"
                color = colors[index % len(colors)] if colors else None
                component = np.asarray(peak_curve, dtype=float) / scale
                calibrated_components.append(component)
                self.ax_spectrum.plot(
                    fit_energy,
                    component,
                    linestyle="--",
                    linewidth=0.9,
                    color=color,
                    alpha=0.9,
                    label=label,
                )
            self.calibrated_component_labels = list(getattr(fit_result, "peak_labels", []))
            self.calibrated_components = calibrated_components

    _spectrum_view._finish_spectrum_axes(
        self,
        xlabel="Energy",
        ylabel="Normalized intensity",
        view_mode="calibration",
        view_label="Calibrated spectrum",
    )
    if hasattr(self, "cal_status_var"):
        self.cal_status_var.set("Current spectrum plotted on calibrated energy axis")


def save_calibrated_spectrum_data(self):
    """Save the data used by Apply Cal. as a tab-delimited text table."""
    calibration = getattr(self, "energy_calibration", None)
    if calibration is None:
        calibration = calculate_energy_calibration(self)
    if calibration is None:
        return
    if not hasattr(self, "last_spectrum_roi"):
        messagebox.showwarning("Export Cal.", "Plot the current spectrum first.")
        return

    y_data = np.asarray(self.last_spectrum_roi, dtype=float)
    energy = _energy_axis(calibration, _spectrum_view._current_spectrum_x_axis(self, y_data.size))
    fit_result = getattr(self, "last_peak_fit_profile", None)

    run_number = _current_run_number(self).replace(" ", "_").replace(":", "")
    initialfile = f"Run_{run_number}_calibrated_spectrum.txt"
    save_path = filedialog.asksaveasfilename(
        initialfile=initialfile,
        defaultextension=".txt",
    )
    if not save_path:
        return

    try:
        with open(save_path, mode="w", newline="") as file:
            writer = csv.writer(file, delimiter="\t")
            writer.writerow(["Calibrated spectrum"])
            writer.writerow([f"slope", calibration["slope"]])
            writer.writerow([f"intercept", calibration["intercept"]])
            writer.writerow([f"Kbeta_res residual", calibration.get("residual_res_energy", "")])
            writer.writerow([])
            if fit_result is None:
                writer.writerow(["Energy", "Current intensity"])
                for row_index in range(energy.size):
                    writer.writerow([energy[row_index], y_data[row_index]])
            else:
                fit_energy = _energy_axis(calibration, fit_result.x)
                fit_y = np.asarray(fit_result.normalized_fit, dtype=float)
                current_interp = np.interp(fit_energy, energy, y_data)
                scale = _component_scale(fit_result)
                component_labels = list(getattr(fit_result, "peak_labels", []))
                components = []
                if scale is not None:
                    for peak_curve in fit_result.peak_curves:
                        components.append(np.asarray(peak_curve, dtype=float) / scale)
                writer.writerow(["Energy", "Current intensity", "Peak fit", *component_labels[:len(components)]])
                for row_index in range(fit_energy.size):
                    writer.writerow(
                        [
                            fit_energy[row_index],
                            current_interp[row_index],
                            fit_y[row_index],
                            *[component[row_index] for component in components],
                        ]
                    )
        if hasattr(self, "cal_status_var"):
            self.cal_status_var.set(f"Calibrated spectrum saved: {os.path.basename(save_path)}")
        print(f"Calibrated spectrum saved to {save_path}")
    except Exception as exc:
        messagebox.showwarning("Export Cal.", f"Could not save calibrated spectrum:\n{exc}")
        print(f"Error saving calibrated spectrum: {exc}")


def compare_calibration_overlay(self):
    """Overlay calibrant fit and current peak fit on the calibrated energy axis."""
    if not hasattr(self, "last_peak_fit_profile"):
        messagebox.showwarning("Calibration", "Run Peak Fit on the current spectrum first.")
        return
    cal_fit = getattr(self, "calibration_fit", None)
    if cal_fit is None:
        cal_fit = fit_calibration_spectrum(self)
    if cal_fit is None:
        return
    calibration = getattr(self, "energy_calibration", None)
    if calibration is None:
        calibration = calculate_energy_calibration(self)
    if calibration is None:
        return

    current_fit = self.last_peak_fit_profile
    current_energy = _energy_axis(calibration, current_fit.x)
    current_y = _normalize_overlay(current_fit.normalized_fit)
    cal_y = _normalize_overlay(cal_fit.normalized_fit)

    self.ax_spectrum.clear()
    self.ax_spectrum.plot(
        cal_fit.x,
        cal_y,
        color="#d95f02",
        linewidth=1.3,
        label="Calibrant fit",
    )
    self.ax_spectrum.plot(
        current_energy,
        current_y,
        color="black",
        linewidth=1.6,
        label="Current fit",
    )
    _spectrum_view._finish_spectrum_axes(
        self,
        xlabel="Energy",
        ylabel="Normalized intensity",
        view_mode="calibration",
        view_label="Calibration comparison",
    )
    if hasattr(self, "cal_status_var"):
        self.cal_status_var.set("Calibration comparison plotted")
