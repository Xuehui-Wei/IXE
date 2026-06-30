# -*- coding: utf-8 -*-
import numpy as np
import re
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.colorchooser import askcolor

try:
    from .peak_fitting import PeakFitConfig, PeakFitError, calculate_fitted_iad, fit_spectrum
    from . import spectrum_controls as _spectrum_view
except ImportError:
    try:
        from peak_fitting import PeakFitConfig, PeakFitError, calculate_fitted_iad, fit_spectrum
        import spectrum_controls as _spectrum_view
    except ImportError:
        from IXE.peak_fitting import PeakFitConfig, PeakFitError, calculate_fitted_iad, fit_spectrum
        from IXE import spectrum_controls as _spectrum_view


def _get_selected_reference_line(self):
    for name, data in self.plotted_lines.items():
        checkbox_var = data.get('checkbox_var')
        is_checked = checkbox_var.get() if checkbox_var is not None else data.get('visible', False)
        if data.get('is_ref', False) and is_checked:
            return name, data
    return None, None


def _get_iad_input_data(self):
    ref_name, ref_line = _get_selected_reference_line(self)
    if ref_line is None:
        messagebox.showwarning("Reminder", "No reference spectrum selected.")
        return None
    if not hasattr(self, 'last_spectrum_roi'):
        print("No ROI spectrum to compare. Please plot ROI or apply moving average first.")
        return None

    ref_y = np.array(ref_line['y_data'], dtype=float)
    ref_x = np.array(ref_line['x_data'], dtype=float)
    roi_curve = _get_roi_curve_for_reference(self, ref_line)
    if roi_curve is None:
        return None
    roi_x, roi_y, _roi_label = roi_curve
    roi_y = _align_roi_to_reference(ref_x, roi_x, roi_y)
    if len(roi_y) != len(ref_y) or len(ref_y) != len(ref_x):
        messagebox.showwarning("Error", "ROI spectrum and reference spectrum lengths do not match. Cannot compare.")
        print("Error: ROI spectrum and reference spectrum lengths do not match.")
        return None
    return ref_name, ref_line, ref_x, roi_y, ref_y


def _entry_text(entry_widget):
    if entry_widget is None:
        return ""
    return entry_widget.get().strip()


def _get_peak_fit_config(self):
    peak_defaults = self.parm.setdefault('peak_fit', {}) if hasattr(self, 'parm') else {}
    peak_count_text = _entry_text(getattr(self, 'peak_fit_count', None))
    if not peak_count_text:
        peak_count_text = str(peak_defaults.get('n_peaks', 3))
    try:
        n_peaks = int(peak_count_text)
    except ValueError:
        messagebox.showwarning("Peak Fit", "Peak count must be a positive integer.")
        return None

    x_min = None
    x_max = None
    x_min_text = _entry_text(getattr(self, 'peak_fit_start', None))
    x_max_text = _entry_text(getattr(self, 'peak_fit_end', None))
    try:
        if x_min_text:
            x_min = float(x_min_text)
        if x_max_text:
            x_max = float(x_max_text)
    except ValueError:
        messagebox.showwarning("Peak Fit", "Fit range values must be numeric.")
        return None
    if x_min is not None and x_max is not None and x_min > x_max:
        x_min, x_max = x_max, x_min
    peak_shape = peak_defaults.get('peak_shape', 'pseudo_voigt')
    if hasattr(self, 'peak_fit_model_var'):
        peak_shape = self.peak_fit_model_var.get()
    peak_defaults['n_peaks'] = n_peaks
    peak_defaults['x_min'] = x_min
    peak_defaults['x_max'] = x_max
    peak_defaults['peak_shape'] = peak_shape

    try:
        return PeakFitConfig(n_peaks=n_peaks, x_min=x_min, x_max=x_max, peak_shape=peak_shape)
    except PeakFitError as exc:
        messagebox.showwarning("Peak Fit", str(exc))
        return None


def _update_iad_entry(self, ref_name, entry_name, value):
    data = self.plotted_lines.get(ref_name)
    if data is not None and entry_name in data:
        data[entry_name].delete(0, 'end')
        data[entry_name].insert(0, f"{value:.4f}")


def _run_number_from_text(text):
    text = str(text or "")
    run_match = re.search(r"Run[_\s:]*(\d+)", text, re.IGNORECASE)
    if run_match:
        return run_match.group(1)
    number_match = re.search(r"\b(\d+)\b", text)
    return number_match.group(1) if number_match else "?"


def _current_roi_run_number(self):
    if hasattr(self, 'run_number_label'):
        run_number = _run_number_from_text(self.run_number_label.cget("text"))
        if run_number != "?":
            return run_number
    return _run_number_from_text(getattr(self, 'original_filepath', ""))


def _reference_run_number(ref_name, ref_line):
    run_number = _run_number_from_text(ref_line.get('run_number'))
    if run_number != "?":
        return run_number
    return _run_number_from_text(ref_name)


def _iad_spectrum_labels(self, ref_name, ref_line):
    return (
        f"Ref, run {_reference_run_number(ref_name, ref_line)}",
        f"ROI, run {_current_roi_run_number(self)}",
    )


def _get_satellite_controls(self, spectrum_len):
    try:
        cross_begin = int(self.cross_begin.get())
        cross_end = int(self.cross_end.get())
    except ValueError:
        messagebox.showwarning("Satellite IAD", "Spectra Crossing Range values must be integers.")
        return None

    max_index = max(int(spectrum_len) - 1, 0)
    cross_begin = max(0, min(cross_begin, max_index))
    cross_end = max(0, min(cross_end, max_index))
    if cross_end <= cross_begin:
        messagebox.showwarning("Satellite IAD", "Spectra Crossing Range end must be greater than the start.")
        return None

    eyeball_str = self.eye_ball_cross.get().strip()
    try:
        eyeball_point = int(eyeball_str) if eyeball_str else None
    except ValueError:
        eyeball_point = None
    if eyeball_point is not None:
        eyeball_point = max(0, min(eyeball_point, max_index))
    return cross_begin, cross_end, eyeball_point


def _split_spectrum_row(line):
    stripped = line.strip()
    if not stripped or stripped.startswith('#'):
        return []
    if '\t' in stripped:
        return [value.strip() for value in stripped.split('\t')]
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
    for name in ("peak fit", "y", "intensity"):
        if name in normalized:
            return normalized.index(name)
    return 1 if len(headers) > 1 else None


def _is_spectrum_table_header(row):
    normalized = [_header_name(value) for value in row]
    if not normalized or normalized[0] != "x":
        return False
    return any(name in normalized for name in ("peak fit", "y", "intensity", "raw y"))


def _parse_txt_spectrum(ref_file):
    with open(ref_file, mode='r') as file:
        rows = [_split_spectrum_row(line) for line in file]

    x_data = []
    y_data = []
    source_column = "Y"

    for row_index, row in enumerate(rows):
        if not row:
            continue
        if not _is_spectrum_table_header(row):
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
            return x_data, y_data, source_column

    for row in rows:
        if len(row) < 2:
            continue
        x_value = _as_float(row[0])
        y_value = _as_float(row[1])
        if x_value is not None and y_value is not None:
            x_data.append(x_value)
            y_data.append(y_value)

    if not y_data:
        raise ValueError("No numeric spectrum data found. For pkfit files, the importer expects an X column and a Peak fit column.")
    return x_data, y_data, source_column


def _parse_chi_spectrum(ref_file):
    x_data = []
    y_data = []
    with open(ref_file, mode='r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            values = line.strip().split()
            if len(values) != 2:
                continue
            x_value = _as_float(values[0])
            y_value = _as_float(values[1])
            if x_value is not None and y_value is not None:
                x_data.append(x_value)
                y_data.append(y_value)
    if not y_data:
        raise ValueError("No numeric spectrum data found in the CHI file.")
    return x_data, y_data, "Y"


def _is_peak_fit_column(source_column):
    return _header_name(source_column) == "peak fit"


def _repair_x_axis_if_needed(x_values):
    x_values = np.asarray(x_values, dtype=float)
    if x_values.size < 2:
        return np.arange(x_values.size, dtype=float), True
    finite_x = x_values[np.isfinite(x_values)]
    if finite_x.size < 2:
        return np.arange(x_values.size, dtype=float), True
    x_span = float(np.nanmax(finite_x) - np.nanmin(finite_x))
    unique_count = np.unique(np.round(finite_x, decimals=9)).size
    if not np.isfinite(x_span) or x_span <= np.finfo(float).eps or unique_count < 2:
        return np.arange(x_values.size, dtype=float), True
    return x_values, False


def _valid_plot_x(x_values, expected_len):
    x_values = np.asarray(x_values, dtype=float)
    if x_values.size != int(expected_len) or x_values.size < 2:
        return False
    finite_x = x_values[np.isfinite(x_values)]
    if finite_x.size < 2:
        return False
    return (float(np.nanmax(finite_x) - np.nanmin(finite_x)) > np.finfo(float).eps)


def _current_peak_fit_x_axis(self, expected_len):
    if hasattr(self, 'ax_spectrum'):
        preferred_labels = ("Peak fit", "ROI fit")
        for preferred_label in preferred_labels:
            for line in self.ax_spectrum.lines:
                if line.get_label() == preferred_label:
                    x_values = np.asarray(line.get_xdata(), dtype=float)
                    if _valid_plot_x(x_values, expected_len):
                        return x_values.copy()
        for line in self.ax_spectrum.lines:
            label = line.get_label()
            if label.startswith("_") or label.startswith("Run "):
                continue
            x_values = np.asarray(line.get_xdata(), dtype=float)
            if _valid_plot_x(x_values, expected_len):
                return x_values.copy()
    if hasattr(self, 'last_peak_fit_profile'):
        x_values = np.asarray(self.last_peak_fit_profile.x, dtype=float)
        if _valid_plot_x(x_values, expected_len):
            return x_values.copy()
    return None


def _print_plot_line_axes(self):
    if not hasattr(self, 'ax_spectrum'):
        return
    for line in self.ax_spectrum.lines:
        label = line.get_label()
        if label.startswith("_") or label.startswith("Run "):
            continue
        x_values = np.asarray(line.get_xdata(), dtype=float)
        y_values = np.asarray(line.get_ydata(), dtype=float)
        if x_values.size == 0 or y_values.size == 0:
            continue
        print(
            "Plot line '{0}': x n={1}, range={2:.3g}-{3:.3g}; y n={4}, range={5:.3g}-{6:.3g}".format(
                label,
                x_values.size,
                float(np.nanmin(x_values)),
                float(np.nanmax(x_values)),
                y_values.size,
                float(np.nanmin(y_values)),
                float(np.nanmax(y_values)),
            )
        )


def _ensure_roi_peak_fit(self):
    if hasattr(self, 'last_peak_fit_profile'):
        return True
    if not hasattr(self, 'last_spectrum_roi'):
        messagebox.showwarning("Peak Fit", "Plot an ROI spectrum before importing a pkfit reference.")
        return False

    fit_config = _get_peak_fit_config(self)
    if fit_config is None:
        return False

    y_data = np.asarray(self.last_spectrum_roi, dtype=float)
    x_data = _spectrum_view._current_spectrum_x_axis(self, len(y_data))
    try:
        self.last_peak_fit_profile = fit_spectrum(x_data, y_data, fit_config)
    except PeakFitError as exc:
        messagebox.showwarning("Peak Fit", f"Could not fit current ROI spectrum: {exc}")
        print(f"Peak fitting failed during pkfit import: {exc}")
        return False
    return True


def _get_current_peak_fit_line(self):
    if hasattr(self, 'ax_spectrum'):
        for line in self.ax_spectrum.lines:
            if line.get_label() == "Peak fit":
                return (
                    np.asarray(line.get_xdata(), dtype=float).copy(),
                    np.asarray(line.get_ydata(), dtype=float).copy(),
                )
    if hasattr(self, 'last_peak_fit_profile'):
        return (
            np.asarray(self.last_peak_fit_profile.x, dtype=float),
            np.asarray(self.last_peak_fit_profile.normalized_fit, dtype=float),
        )
    return None


def _target_peak_fit_on_x(self, x_values):
    peak_fit_line = _get_current_peak_fit_line(self)
    if peak_fit_line is None:
        return None
    target_x, target_y = peak_fit_line
    x_values = np.asarray(x_values, dtype=float)
    target_x = np.asarray(target_x, dtype=float)
    target_y = np.asarray(target_y, dtype=float)
    if x_values.shape == target_x.shape and np.allclose(x_values, target_x):
        return target_y.copy()
    if target_x.size < 2:
        return None
    order = np.argsort(target_x)
    target_x = target_x[order]
    target_y = target_y[order]
    if np.nanmin(x_values) < np.nanmin(target_x) or np.nanmax(x_values) > np.nanmax(target_x):
        return None
    return np.interp(x_values, target_x, target_y)


def _tail_area_match_to_target(y_values, target_values, edge_fraction=0.08):
    y_values = np.asarray(y_values, dtype=float)
    target_values = np.asarray(target_values, dtype=float)
    if y_values.shape != target_values.shape or y_values.size < 8:
        return y_values.copy(), False
    finite = np.isfinite(y_values) & np.isfinite(target_values)
    if finite.sum() < 8:
        return y_values.copy(), False

    edge_count = max(3, int(round(y_values.size * float(edge_fraction))))
    edge_count = min(edge_count, max(3, y_values.size // 3))
    x_scaled = np.linspace(0.0, 1.0, y_values.size)

    ref_left = float(np.nanmedian(y_values[:edge_count]))
    ref_right = float(np.nanmedian(y_values[-edge_count:]))
    target_left = float(np.nanmedian(target_values[:edge_count]))
    target_right = float(np.nanmedian(target_values[-edge_count:]))
    if not all(np.isfinite(value) for value in (ref_left, ref_right, target_left, target_right)):
        return y_values.copy(), False

    ref_tail = np.interp(x_scaled, [0.0, 1.0], [ref_left, ref_right])
    target_tail = np.interp(x_scaled, [0.0, 1.0], [target_left, target_right])
    ref_signal = y_values - ref_tail
    target_signal = target_values - target_tail

    ref_area = float(np.nansum(np.clip(ref_signal, 0.0, None)))
    target_area = float(np.nansum(np.clip(target_signal, 0.0, None)))
    scale = 1.0
    if ref_area > np.finfo(float).eps and target_area > np.finfo(float).eps:
        scale = target_area / ref_area

    matched = target_tail + scale * ref_signal
    matched_left = float(np.nanmedian(matched[:edge_count]))
    matched_right = float(np.nanmedian(matched[-edge_count:]))
    if np.isfinite(matched_left) and np.isfinite(matched_right):
        matched = matched + np.interp(
            x_scaled,
            [0.0, 1.0],
            [target_left - matched_left, target_right - matched_right],
        )

    return matched, True


def _align_spectra_by_main_peak(roi_y, ref_y):
    roi_y = np.asarray(roi_y, dtype=float)
    ref_y = np.asarray(ref_y, dtype=float)
    if roi_y.size == 0 or ref_y.size == 0:
        raise ValueError("Cannot align empty spectra.")

    i_max = int(np.nanargmax(roi_y))
    i_ref_max = int(np.nanargmax(ref_y))
    shift = i_ref_max - i_max
    if shift < 0:
        roi_aligned = roi_y[-shift:]
        ref_aligned = ref_y[:shift]
    elif shift > 0:
        roi_aligned = roi_y[:-shift]
        ref_aligned = ref_y[shift:]
    else:
        roi_aligned = roi_y.copy()
        ref_aligned = ref_y.copy()

    common_len = min(roi_aligned.size, ref_aligned.size)
    if common_len == 0:
        raise ValueError("Main-peak alignment produced empty spectra.")
    return roi_aligned[:common_len], ref_aligned[:common_len]


def _find_satellite_transition(roi_y, ref_y, cross_begin, cross_end, eyeball_point=None):
    """Find the satellite/main transition without assuming which spectrum is above."""
    max_index = max(min(len(roi_y), len(ref_y)) - 1, 0)
    if max_index <= 0:
        return None
    if eyeball_point is not None:
        return max(0, min(int(eyeball_point), max_index))

    start = max(0, min(int(cross_begin), max_index - 1))
    end = max(start + 1, min(int(cross_end), max_index))
    diff = np.asarray(roi_y, dtype=float) - np.asarray(ref_y, dtype=float)
    for index in range(start, end):
        left = diff[index]
        right = diff[index + 1]
        if not np.isfinite(left) or not np.isfinite(right):
            continue
        if left == 0:
            return index
        if right == 0:
            return index + 1
        if np.sign(left) != np.sign(right):
            return index + 1

    window = np.abs(diff[start:end + 1])
    finite = np.isfinite(window)
    if np.any(finite):
        finite_indices = np.flatnonzero(finite)
        return start + int(finite_indices[int(np.nanargmin(window[finite]))])
    return None


def _calculate_peak_fit_satellite_iad(roi_y, ref_y, cross_begin, cross_end, eyeball_point):
    roi_aligned, ref_aligned = _align_spectra_by_main_peak(roi_y, ref_y)
    ref_aligned, tail_matched = _tail_area_match_to_target(ref_aligned, roi_aligned)
    transition_point = _find_satellite_transition(
        roi_aligned,
        ref_aligned,
        cross_begin,
        cross_end,
        eyeball_point,
    )
    iad_satellite = float(np.sum(np.abs(roi_aligned[:transition_point] - ref_aligned[:transition_point])))
    return iad_satellite, transition_point, roi_aligned, ref_aligned, tail_matched


def _satellite_plot_slice(spec_align, spec_r_align, transition_point):
    spec_align = np.asarray(spec_align, dtype=float)
    spec_r_align = np.asarray(spec_r_align, dtype=float)
    max_len = min(spec_align.size, spec_r_align.size)
    if max_len == 0:
        return np.array([], dtype=int), spec_align[:0], spec_r_align[:0]

    if transition_point is None:
        end = max_len
    else:
        end = max(1, min(int(transition_point), max_len))
    x_satellite = np.arange(end)
    return x_satellite, spec_align[:end], spec_r_align[:end]


def _plot_signed_iad_difference(self, x_values, roi_y, ref_y, color, label="IAD diff."):
    """Draw the old-style IAD residual in the reference spectrum color."""
    x_values = np.asarray(x_values, dtype=float)
    roi_y = np.asarray(roi_y, dtype=float)
    ref_y = np.asarray(ref_y, dtype=float)
    common_len = min(x_values.size, roi_y.size, ref_y.size)
    if common_len == 0:
        return
    x_values = x_values[:common_len]
    diff = roi_y[:common_len] - ref_y[:common_len]
    finite = np.isfinite(x_values) & np.isfinite(diff)
    if not np.any(finite):
        return
    baseline = np.zeros(common_len, dtype=float)
    self.ax_spectrum.axhline(0, color=color, linewidth=0.8, alpha=0.9, zorder=1)
    self.ax_spectrum.fill_between(
        x_values,
        baseline,
        diff,
        where=finite,
        color=color,
        alpha=0.55,
        interpolate=True,
        linewidth=0,
        label=label,
        zorder=1,
    )
    self.ax_spectrum.plot(x_values[finite], diff[finite], color=color, linewidth=0.6, zorder=2)


def _get_roi_curve_for_reference(self, ref_line):
    if ref_line.get('is_peak_fit', False):
        if not _ensure_roi_peak_fit(self):
            return None
        fit_result = self.last_peak_fit_profile
        return (
            np.asarray(fit_result.x, dtype=float),
            np.asarray(fit_result.normalized_fit, dtype=float),
            "ROI fit",
        )
    return (
        _spectrum_view._current_spectrum_x_axis(self, len(self.last_spectrum_roi)),
        np.asarray(self.last_spectrum_roi, dtype=float),
        "ROI",
    )


def _align_roi_to_reference(ref_x, roi_x, roi_y):
    ref_x = np.asarray(ref_x, dtype=float)
    roi_x = np.asarray(roi_x, dtype=float)
    roi_y = np.asarray(roi_y, dtype=float)
    if ref_x.shape == roi_x.shape and np.allclose(ref_x, roi_x):
        return roi_y
    if roi_x.size < 2:
        return roi_y
    order = np.argsort(roi_x)
    roi_x_sorted = roi_x[order]
    roi_y_sorted = roi_y[order]
    if np.nanmin(ref_x) < np.nanmin(roi_x_sorted) or np.nanmax(ref_x) > np.nanmax(roi_x_sorted):
        return roi_y
    return np.interp(ref_x, roi_x_sorted, roi_y_sorted)


def _redraw_peak_fit_reference_plot(self):
    roi_color = self.line_color.get() if hasattr(self, 'line_color') else 'black'
    self.ax_spectrum.clear()
    peak_fit_line = _get_current_peak_fit_line(self)
    if peak_fit_line is not None:
        peak_x, peak_y = peak_fit_line
        self.ax_spectrum.plot(
            peak_x,
            peak_y,
            color=roi_color,
            linestyle='-',
            linewidth=1.6,
            label="Peak fit",
        )
    for existing_name, existing_line in self.plotted_lines.items():
        if existing_line.get('visible') and existing_line.get('is_peak_fit', False):
            self.ax_spectrum.plot(
                existing_line['x_data'],
                existing_line['y_data'],
                label=existing_name,
                color=existing_line['color'],
                linestyle='-',
                linewidth=0.5,
            )
    self.ax_spectrum.set_title("")
    self.ax_spectrum.set_xlabel("Column Index")
    self.ax_spectrum.set_ylabel("Normalized intensity")


def _redraw_reference_display(self):
    has_peak_fit_ref = any(
        line_data.get('visible') and line_data.get('is_peak_fit', False)
        for line_data in self.plotted_lines.values()
    )
    if has_peak_fit_ref:
        _redraw_peak_fit_reference_plot(self)
    else:
        self.ax_spectrum.clear()
        if hasattr(self, 'last_spectrum_roi'):
            roi_y = np.asarray(self.last_spectrum_roi, dtype=float)
            roi_x = _spectrum_view._current_spectrum_x_axis(self, len(roi_y))
            roi_color = self.line_color.get() if hasattr(self, 'line_color') else 'black'
            try:
                line_width = float(self.line_width.get())
            except Exception:
                line_width = 0.5
            line_style = self.line_style.get() if hasattr(self, 'line_style') else '-'
            self.ax_spectrum.plot(
                roi_x,
                roi_y,
                color=roi_color,
                linestyle=line_style,
                linewidth=line_width,
                label="ROI",
            )
        for line_name, line_data in self.plotted_lines.items():
            if line_data.get('visible'):
                self.ax_spectrum.plot(
                    line_data['x_data'],
                    line_data['y_data'],
                    label=line_name,
                    color=line_data['color'],
                    linestyle='-',
                    linewidth=0.5,
                )
        self.ax_spectrum.set_title("")
    _spectrum_view._finish_spectrum_axes(
        self,
        xlabel="Column Index",
        ylabel="Normalized intensity",
        view_mode="reference",
        view_label="Reference comparison",
    )


def import_ref_data(self):
    """Import the reference spectrum data from a .txt or .chi file and plot it in the spectrum plot."""
    ref_file = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("CHI files", "*.chi"), ("All files", "*.*")])
    if ref_file:
        file_extension = ref_file.split('.')[-1].lower()
        try:
            if file_extension == 'txt':
                x_data, y_data, source_column = _parse_txt_spectrum(ref_file)
            elif file_extension == 'chi':
                x_data, y_data, source_column = _parse_chi_spectrum(ref_file)
            else:
                x_data, y_data, source_column = _parse_txt_spectrum(ref_file)
            x_data_array = np.array(x_data, dtype=float)
            y_data_array = np.array(y_data, dtype=float)
            finite_mask = np.isfinite(x_data_array) & np.isfinite(y_data_array)
            x_data_array = x_data_array[finite_mask]
            y_data_array = y_data_array[finite_mask]
            if y_data_array.size == 0:
                raise ValueError("No finite spectrum values found in the selected file.")
            is_peak_fit_ref = _is_peak_fit_column(source_column)
            if is_peak_fit_ref:
                _print_plot_line_axes(self)
                matched_x_axis = _current_peak_fit_x_axis(self, y_data_array.size)
                if matched_x_axis is not None:
                    x_data_array = matched_x_axis
                    x_repaired = False
                else:
                    x_data_array, x_repaired = _repair_x_axis_if_needed(x_data_array)
            else:
                x_data_array, x_repaired = _repair_x_axis_if_needed(x_data_array)
            x_data = x_data_array.tolist()
            ref_gap_corrected = False
            if hasattr(self, 'spect_processor') and self.spect_processor is not None:
                gap_defaults = self.parm.get('ccd_gap', {}) if hasattr(self, 'parm') else {}
                gap_enabled = getattr(self, 'gap_correction_enabled', gap_defaults.get('enabled', True))
                if not is_peak_fit_ref and gap_enabled and hasattr(self.spect_processor, 'correct_narrow_gaps'):
                    self.spect_processor.set_gap_correction_params(
                        max_width=gap_defaults.get('max_width', 8),
                        drop_fraction=gap_defaults.get('drop_fraction', 0.65),
                    )
                    y_data_array, ref_gap_mask = self.spect_processor.correct_narrow_gaps(y_data_array)
                    ref_gap_corrected = bool(np.any(ref_gap_mask))
            tail_matched = False
            if is_peak_fit_ref:
                y_data_norm = y_data_array.copy()
                target_y = _target_peak_fit_on_x(self, x_data_array)
                if target_y is not None:
                    y_data_norm, tail_matched = _tail_area_match_to_target(y_data_norm, target_y)
            else:
                y_data_norm = self.spect_processor.get_norm_spectrum(y_data_array)
            line_color = self.spect_processor.random_color()
            ref_filename = ref_file.split("/")[-1]
            ref_run_number_match = re.search(r"Run_(\d+)", ref_filename)
            if ref_run_number_match:
                ref_run_number = ref_run_number_match.group(1)
            else:
                ref_run_number = "?"
            label = f"Run {ref_run_number}"
            line_name = label
            self.plotted_lines[line_name] = {
                'x_data': x_data,
                'y_data': y_data_norm,
                'color': line_color,
                'run_number': ref_run_number,
                'visible': True,
                'checkbox': None,
                'color_circle': None,
                'is_ref': True,
                'is_peak_fit': is_peak_fit_ref,
                'source_column': source_column,
                'x_axis_repaired': x_repaired,
                'gap_corrected': ref_gap_corrected,
                'tail_matched': tail_matched,
            }
            if is_peak_fit_ref:
                _redraw_reference_display(self)
            else:
                self.ax_spectrum.plot(x_data, y_data_norm, label=label, color=line_color, linestyle='-', linewidth=0.5)
            self.add_line_to_list(line_name, line_color)
            _spectrum_view._finish_spectrum_axes(
                self,
                xlabel="Column Index",
                ylabel="Normalized intensity",
                view_mode="reference",
                view_label="Reference comparison",
            )
        except Exception as e:
            messagebox.showwarning("Import Ref. Data", str(e))
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
            _redraw_reference_display(self)
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
            _redraw_reference_display(self)

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
    if line_name not in self.plotted_lines:
        return
    if checkbox_var.get():
        if not self.plotted_lines[line_name]['visible']:
            self.plotted_lines[line_name]['visible'] = True
    else:
        self.plotted_lines[line_name]['visible'] = False
    _redraw_reference_display(self)

def plot_integrated_diff(self):
    input_data = _get_iad_input_data(self)
    if input_data is None:
        return
    ref_name, ref_line, ref_x, roi_y, ref_y = input_data
    roi_color = self.line_color.get() if hasattr(self, 'line_color') else 'black'
    ref_label, roi_label = _iad_spectrum_labels(self, ref_name, ref_line)
    self.ax_spectrum.clear()
    self.ax_spectrum.plot(ref_x, ref_y, color=ref_line['color'], linewidth=0.5, label=ref_label)
    self.ax_spectrum.plot(ref_x, roi_y, color=roi_color, linewidth=0.5, label=roi_label)
    _plot_signed_iad_difference(self, ref_x, roi_y, ref_y, ref_line['color'], label="IAD")
    _spectrum_view._finish_spectrum_axes(
        self,
        xlabel="Column Index",
        ylabel="Normalized intensity",
        view_mode="iad",
        view_label="IAD difference",
    )
    iad_value = np.sum(np.abs(roi_y - ref_y))
    _update_iad_entry(self, ref_name, 'iad_entry', iad_value)
    if hasattr(self, 'iad_value_label'):
        self.iad_value_label.config(text=f"IAD = {iad_value:.4f}")

def plot_fitted_integrated_diff(self):
    input_data = _get_iad_input_data(self)
    if input_data is None:
        return
    ref_name, ref_line, ref_x, roi_y, ref_y = input_data
    fit_config = _get_peak_fit_config(self)
    if fit_config is None:
        return
    try:
        fitted = calculate_fitted_iad(ref_x, roi_y, ref_y, fit_config)
    except PeakFitError as exc:
        messagebox.showwarning("Peak Fit", str(exc))
        print(f"Peak fitting failed: {exc}")
        return

    self.last_peak_fit_iad = fitted
    roi_color = self.line_color.get() if hasattr(self, 'line_color') else 'black'
    ref_label, roi_label = _iad_spectrum_labels(self, ref_name, ref_line)
    self.ax_spectrum.clear()
    self.ax_spectrum.plot(fitted.x, fitted.ref_fit, color=ref_line['color'], linewidth=1.0, label=ref_label)
    self.ax_spectrum.plot(fitted.x, fitted.roi_fit, color=roi_color, linewidth=1.0, label=roi_label)
    _plot_signed_iad_difference(self, fitted.x, fitted.roi_fit, fitted.ref_fit, ref_line['color'], label="Fitted Diff.")
    _spectrum_view._finish_spectrum_axes(
        self,
        xlabel="Column Index",
        ylabel="Normalized intensity",
        view_mode="iad",
        view_label="Fitted IAD difference",
    )
    _update_iad_entry(self, ref_name, 'iad_entry', fitted.iad_value)
    if hasattr(self, 'iad_value_label'):
        self.iad_value_label.config(text=f"Fitted IAD = {fitted.iad_value:.4f}")
    print(
        "Fitted IAD centers - ROI: {0}; REF: {1}".format(
            ", ".join(
                f"{label}={center:.2f}"
                for label, center in zip(fitted.roi_result.peak_labels, fitted.roi_result.centers)
            ),
            ", ".join(
                f"{label}={center:.2f}"
                for label, center in zip(fitted.ref_result.peak_labels, fitted.ref_result.centers)
            ),
        )
    )

def calculate_and_display_satellite_peak_iad(self):
    input_data = _get_iad_input_data(self)
    if input_data is None:
        return
    ref_name, ref_line, _ref_x, roi_y, ref_y = input_data
    satellite_controls = _get_satellite_controls(self, len(roi_y))
    if satellite_controls is None:
        return
    cross_begin, cross_end, eyeball_point = satellite_controls
    ref_label, roi_label = _iad_spectrum_labels(self, ref_name, ref_line)
    if ref_line.get('is_peak_fit', False):
        try:
            iad_satellite, transition_point, spec_align, spec_r_align, tail_matched = _calculate_peak_fit_satellite_iad(
                roi_y,
                ref_y,
                cross_begin,
                cross_end,
                eyeball_point,
            )
        except ValueError as exc:
            messagebox.showwarning("Satellite IAD", str(exc))
            return
        diff_label = "Fitted Satellite IAD"
        line_width = 1.0
        ref_line['satellite_tail_matched'] = tail_matched
    else:
        iad_satellite, transition_point, spec_align, spec_r_align = self.spect_processor.calculate_satellite_peak_iad(roi_y, ref_y, cross_begin, cross_end, eyeball_point)
        diff_label = "Satellite IAD"
        line_width = 0.5
    self.ax_spectrum.clear()
    roi_color = self.line_color.get() if hasattr(self, 'line_color') else 'black'
    self.ax_spectrum.clear()
    x_aligned = np.arange(len(spec_align))
    x_satellite, spec_satellite, spec_r_satellite = _satellite_plot_slice(
        spec_align,
        spec_r_align,
        transition_point,
    )
    self.ax_spectrum.plot(x_aligned, spec_r_align, color=ref_line['color'], linewidth=line_width, label=ref_label)
    self.ax_spectrum.plot(x_aligned, spec_align, color=roi_color, linewidth=line_width, label=roi_label)
    _plot_signed_iad_difference(self, x_satellite, spec_satellite, spec_r_satellite, ref_line['color'], label=diff_label)
    _spectrum_view._finish_spectrum_axes(
        self,
        xlabel="Aligned index",
        ylabel="Normalized intensity",
        view_mode="iad",
        view_label="Satellite IAD",
    )
    _update_iad_entry(self, ref_name, 'st_iad_entry', iad_satellite)
    if hasattr(self, 'sat_iad_value_label'):
        self.sat_iad_value_label.config(text=f"Satellite IAD = {iad_satellite:.4f}")

def calculate_and_display_fitted_satellite_peak_iad(self):
    input_data = _get_iad_input_data(self)
    if input_data is None:
        return
    ref_name, ref_line, ref_x, roi_y, ref_y = input_data
    fit_config = _get_peak_fit_config(self)
    if fit_config is None:
        return
    try:
        fitted = calculate_fitted_iad(ref_x, roi_y, ref_y, fit_config)
    except PeakFitError as exc:
        messagebox.showwarning("Peak Fit", str(exc))
        print(f"Peak fitting failed: {exc}")
        return

    satellite_controls = _get_satellite_controls(self, len(fitted.roi_fit))
    if satellite_controls is None:
        return
    cross_begin, cross_end, eyeball_point = satellite_controls
    iad_satellite, transition_point, spec_align, spec_r_align = self.spect_processor.calculate_satellite_peak_iad(
        fitted.roi_fit,
        fitted.ref_fit,
        cross_begin,
        cross_end,
        eyeball_point,
    )
    self.last_peak_fit_iad = fitted
    self.ax_spectrum.clear()
    roi_color = self.line_color.get() if hasattr(self, 'line_color') else 'black'
    ref_label, roi_label = _iad_spectrum_labels(self, ref_name, ref_line)
    self.ax_spectrum.plot(spec_r_align, color=ref_line['color'], linewidth=1.0, label=ref_label)
    self.ax_spectrum.plot(spec_align, color=roi_color, linewidth=1.0, label=roi_label)
    _plot_signed_iad_difference(
        self,
        np.arange(len(spec_align))[:transition_point],
        spec_align[:transition_point],
        spec_r_align[:transition_point],
        ref_line['color'],
        label="Fitted Satellite IAD",
    )
    _spectrum_view._finish_spectrum_axes(
        self,
        xlabel="Aligned index",
        ylabel="Normalized intensity",
        view_mode="iad",
        view_label="Fitted satellite IAD",
    )
    _update_iad_entry(self, ref_name, 'st_iad_entry', iad_satellite)
    if hasattr(self, 'sat_iad_value_label'):
        self.sat_iad_value_label.config(text=f"Fitted Satellite IAD = {iad_satellite:.4f}")

def pick_ref_color(self):
    color = askcolor()[1]
    if color:
        self.ref_spectrum_color = color
# iad_controls.py
# IAD and satellite peak calculation, reference data import, and line management for TIFFAnalyzer.

# All functions will be moved here unchanged in the next step.
