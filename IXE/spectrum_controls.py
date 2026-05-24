# -*- coding: utf-8 -*-
# spectrum_controls.py
# Spectrum plotting, moving average, and spectrum controls for TIFFAnalyzer.


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk, filedialog, simpledialog
from tkinter.colorchooser import askcolor
import csv
import os
import re

try:
    from .peak_fitting import PeakFitConfig, PeakFitError, fit_spectrum
    from . import pyqtgraph_spectrum
except ImportError:
    try:
        from peak_fitting import PeakFitConfig, PeakFitError, fit_spectrum
        import pyqtgraph_spectrum
    except ImportError:
        from IXE.peak_fitting import PeakFitConfig, PeakFitError, fit_spectrum
        from IXE import pyqtgraph_spectrum


def _entry_text(entry_widget):
    if entry_widget is None:
        return ""
    return entry_widget.get().strip()


def _get_spectrum_peak_fit_config(self):
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
    peak_shape = _selected_peak_fit_model(self)
    peak_defaults['n_peaks'] = n_peaks
    peak_defaults['x_min'] = x_min
    peak_defaults['x_max'] = x_max

    try:
        return PeakFitConfig(n_peaks=n_peaks, x_min=x_min, x_max=x_max, peak_shape=peak_shape)
    except PeakFitError as exc:
        messagebox.showwarning("Peak Fit", str(exc))
        return None


def _selected_peak_fit_model(self):
    peak_defaults = self.parm.setdefault('peak_fit', {}) if hasattr(self, 'parm') else {}
    model = peak_defaults.get('peak_shape', 'pseudo_voigt')
    if hasattr(self, 'peak_fit_model_var'):
        model = self.peak_fit_model_var.get()
    model = str(model or 'pseudo_voigt').strip().lower().replace('-', '_').replace(' ', '_')
    if model in ('pseudovoigt', 'pseudo'):
        model = 'pseudo_voigt'
    elif model in ('lorentz',):
        model = 'lorentzian'
    if model not in ('pseudo_voigt', 'lorentzian'):
        model = 'pseudo_voigt'
    if hasattr(self, 'peak_fit_model_var'):
        self.peak_fit_model_var.set(model)
    peak_defaults['peak_shape'] = model
    return model


def _peak_fit_model_label(model):
    return "Lorentzian" if model == "lorentzian" else "Pseudo-Voigt"


def _display_peak_label(label):
    """Convert Matplotlib math labels into compact Unicode labels for tables."""
    text = str(label)
    compact = re.sub(r"\s+", "", text)
    label_map = {
        r"$K_{\beta'}$": "Kβ′",
        r"$K_{\beta,\mathrm{res}}$": "Kβᵣₑₛ",
        r"$K_{\beta_{1,3}}$": "Kβ₁,₃",
    }
    return label_map.get(compact, text)


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
        total = np.nansum(np.abs(fit_result.best_fit))
        if not np.isfinite(total) or abs(total) <= np.finfo(float).eps:
            return None
        return total
    return raw_total / profile_total


def _peak_fit_parameter_rows(fit_result):
    labels = list(getattr(fit_result, 'peak_labels', []))
    centers = list(getattr(fit_result, 'centers', []))
    widths = list(getattr(fit_result, 'widths', []))
    amplitudes = list(getattr(fit_result, 'amplitudes', []))
    fractions = list(getattr(fit_result, 'fractions', []))
    model = getattr(fit_result, 'peak_shape', 'pseudo_voigt')
    model_label = _peak_fit_model_label(model)
    rows = []
    for index, center in enumerate(centers):
        raw_label = labels[index] if index < len(labels) else f"Peak {index + 1}"
        label = _display_peak_label(raw_label)
        width = widths[index] if index < len(widths) else np.nan
        area = amplitudes[index] if index < len(amplitudes) else np.nan
        fraction = fractions[index] if index < len(fractions) else (1.0 if model == 'lorentzian' else np.nan)
        rows.append((label, model_label, fraction, center, width, area))
    return rows


def _format_fit_value(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value):
        return ""
    return f"{value:.6g}"


def _clear_peak_fit_results_box(self):
    if hasattr(self, 'peak_fit_summary_var'):
        self.peak_fit_summary_var.set("Peak fit parameters: not fitted")
    if hasattr(self, 'peak_fit_table'):
        for item in self.peak_fit_table.get_children():
            self.peak_fit_table.delete(item)


def _update_peak_fit_results_box(self, fit_result):
    if not hasattr(self, 'peak_fit_table'):
        return
    _clear_peak_fit_results_box(self)
    rows = _peak_fit_parameter_rows(fit_result)
    model = getattr(fit_result, 'peak_shape', 'pseudo_voigt')
    if hasattr(self, 'peak_fit_summary_var'):
        if model == 'pseudo_voigt':
            self.peak_fit_summary_var.set(f"Pseudo-Voigt fit; Lorentzian components: {len(rows)}")
        else:
            self.peak_fit_summary_var.set(f"Lorentzian fit; components: {len(rows)}")
    for label, model_label, fraction, center, width, area in rows:
        self.peak_fit_table.insert(
            "",
            "end",
            values=(
                label,
                model_label,
                _format_fit_value(fraction),
                _format_fit_value(center),
                _format_fit_value(width),
                _format_fit_value(area),
            ),
        )


def _configure_gap_correction(self):
    if not hasattr(self, 'spect_processor') or self.spect_processor is None:
        return
    gap_defaults = self.parm.get('ccd_gap', {}) if hasattr(self, 'parm') else {}
    enabled = getattr(self, 'gap_correction_enabled', gap_defaults.get('enabled', True))
    max_width = gap_defaults.get('max_width', 8)
    drop_fraction = gap_defaults.get('drop_fraction', 0.65)
    self.spect_processor.toggle_gap_correction(enabled)
    self.spect_processor.set_gap_correction_params(max_width=max_width, drop_fraction=drop_fraction)
    if hasattr(self.spect_processor, 'set_detector_background_mask'):
        self.spect_processor.set_detector_background_mask(getattr(self, 'processed_detector_background_mask', None))
    if hasattr(self.spect_processor, 'set_edge_correction_params'):
        self.spect_processor.set_edge_correction_params(
            valid_fraction_threshold=gap_defaults.get('edge_valid_fraction', 0.80)
        )
    trace_defaults = self.parm.get('trace_extraction', {}) if hasattr(self, 'parm') else {}
    trace_enabled = getattr(self, 'trace_extraction_enabled', trace_defaults.get('enabled', False))
    if hasattr(self.spect_processor, 'toggle_trace_extraction'):
        self.spect_processor.toggle_trace_extraction(trace_enabled)
        self.spect_processor.set_trace_extraction_params(
            search_margin=trace_defaults.get('search_margin', 60)
        )

def _sync_background_roi(self):
    if not getattr(self, 'bg_subtraction_enabled', False):
        return
    if not hasattr(self, 'spect_processor') or self.spect_processor is None:
        return
    bg_ranges = self.get_bg_row_ranges() if hasattr(self, 'get_bg_row_ranges') else [self.get_bg_row_range()]
    bg_col_start, bg_col_end = self.get_roi_col_range()
    if not bg_ranges or bg_col_start >= bg_col_end:
        raise ValueError("Invalid background ROI coordinates")
    rois = [
        (row_start, row_end + 1, bg_col_start, bg_col_end + 1)
        for row_start, row_end in bg_ranges
        if row_end > row_start
    ]
    if hasattr(self.spect_processor, 'set_background_rois'):
        self.spect_processor.set_background_rois(rois)
    else:
        row_start, row_end, col_start, col_end = rois[0]
        self.spect_processor.set_background_roi(row_start, row_end, col_start, col_end)


def _active_roi_ranges(self):
    if hasattr(self, 'get_roi_row_ranges'):
        ranges = self.get_roi_row_ranges()
    else:
        ranges = [self.get_roi_row_range()]
    valid_ranges = []
    for row_begin, row_end in ranges:
        row_begin, row_end = sorted((int(row_begin), int(row_end)))
        if row_end >= row_begin:
            valid_ranges.append((row_begin, row_end))
    return valid_ranges


def _format_row_ranges(ranges):
    return "; ".join(f"{row_begin}-{row_end}" for row_begin, row_end in ranges) if ranges else "None"


def _active_spectrum_rois(self):
    row_ranges = _active_roi_ranges(self)
    column_begin, column_end = self.get_roi_col_range()
    if not row_ranges or column_end < column_begin:
        raise ValueError("Invalid ROI coordinates")
    rois = [
        (row_begin, row_end + 1, column_begin, column_end + 1)
        for row_begin, row_end in row_ranges
    ]
    return rois, column_begin, column_end


def _extract_active_roi_spectrum(self):
    rois, column_begin, column_end = _active_spectrum_rois(self)
    if hasattr(self.spect_processor, 'get_spectrum_for_rois'):
        spectrum = self.spect_processor.get_spectrum_for_rois(rois)
    else:
        row_begin, row_end, col_begin, col_end = rois[0]
        spectrum = self.spect_processor.get_spectrum(row_begin, row_end, col_begin, col_end)
    return spectrum, rois, column_begin, column_end


def _store_gap_state(self, normalized_spectrum, x_data=None):
    spectrum_len = np.asarray(normalized_spectrum).size
    gap_mask = getattr(self.spect_processor, 'last_gap_mask', None)
    gap_mask = np.asarray(gap_mask, dtype=bool) if gap_mask is not None else np.zeros(spectrum_len, dtype=bool)
    if gap_mask.size != spectrum_len:
        gap_mask = np.zeros(spectrum_len, dtype=bool)
    self.current_spectrum_gap_corrected = bool(getattr(self.spect_processor, 'gap_correction_enabled', False) and np.any(gap_mask))
    gap_indices = np.flatnonzero(gap_mask)
    self.current_spectrum_gap_indices = gap_indices.tolist()
    if x_data is not None:
        x_data = np.asarray(x_data, dtype=float)
        if x_data.size == np.asarray(normalized_spectrum).size:
            self.current_spectrum_x_axis = x_data.copy()
            self.current_spectrum_gap_columns = [
                int(value) if float(value).is_integer() else float(value)
                for value in x_data[gap_indices]
            ]
        else:
            self.current_spectrum_gap_columns = gap_indices.tolist()
    else:
        self.current_spectrum_gap_columns = gap_indices.tolist()
    raw_spectrum = getattr(self.spect_processor, 'last_uncorrected_spectrum', None)
    if raw_spectrum is not None and len(raw_spectrum) == len(normalized_spectrum):
        try:
            self.last_spectrum_roi_uncorrected = self.spect_processor.get_norm_spectrum(np.asarray(raw_spectrum, dtype=float))
        except Exception:
            self.last_spectrum_roi_uncorrected = normalized_spectrum.copy()
    else:
        self.last_spectrum_roi_uncorrected = normalized_spectrum.copy()
    self.current_spectrum_trace_extracted = bool(
        getattr(self.spect_processor, 'trace_extraction_enabled', False)
        and getattr(self.spect_processor, 'last_trace_centers', None) is not None
    )


def _ensure_spectrum_available(self):
    if hasattr(self, 'last_spectrum_roi'):
        return True
    if hasattr(self, 'spect_processor') and self.spect_processor is not None:
        plot_roi_spectrum(self)
        return hasattr(self, 'last_spectrum_roi')
    messagebox.showwarning("Spectrum", "Process an image and plot a spectrum first.")
    return False


def _ensure_peak_fit_available(self):
    if hasattr(self, 'last_peak_fit_profile'):
        return self.last_peak_fit_profile
    if not _ensure_spectrum_available(self):
        return None
    fit_config = _get_spectrum_peak_fit_config(self)
    if fit_config is None:
        return None
    y_data = np.asarray(self.last_spectrum_roi, dtype=float)
    x_data = _current_spectrum_x_axis(self, len(y_data))
    try:
        fit_result = fit_spectrum(x_data, y_data, fit_config)
    except PeakFitError as exc:
        messagebox.showwarning("Peak Fit", str(exc))
        print(f"Peak fitting failed: {exc}")
        return None
    self.last_peak_fit_profile = fit_result
    _update_peak_fit_results_box(self, fit_result)
    print(
        "Peak fit centers: {0}".format(
            ", ".join(
                f"{label}={center:.2f}"
                for label, center in zip(fit_result.peak_labels, fit_result.centers)
            )
        )
    )
    return fit_result


def _set_toggle_button_state(button, enabled, on_text, off_text):
    if button is None:
        return
    button.config(
        text=on_text if enabled else off_text,
        style='Active.TButton' if enabled else 'TButton',
    )


def _current_spectrum_x_axis(self, length, moving_average=0, prefer_stored=True):
    """Return detector-column x values for the currently displayed spectrum."""
    length = int(length)
    if length <= 0:
        return np.array([], dtype=float)
    stored_x = getattr(self, 'current_spectrum_x_axis', None)
    if prefer_stored and stored_x is not None:
        stored_x = np.asarray(stored_x, dtype=float)
        if stored_x.size == length:
            return stored_x.copy()
    try:
        column_begin, _column_end = self.get_roi_col_range()
    except Exception:
        column_begin = 0
    offset = max(int(moving_average), 1) - 1 if moving_average else 0
    start = int(column_begin) + offset
    return np.arange(start, start + length, dtype=float)


def _view_label(self, default="Spectrum"):
    return getattr(self, 'current_spectrum_view_label', default)


def _update_spectrum_status(self, view_label=None):
    """Show compact context for the spectrum view without changing analysis data."""
    if view_label is not None:
        self.current_spectrum_view_label = view_label
    if not hasattr(self, 'spectrum_status_label'):
        return

    if not hasattr(self, 'last_spectrum_roi') and view_label is None:
        self.spectrum_status_label.config(text="No spectrum plotted")
        return

    try:
        roi_ranges = _active_roi_ranges(self)
        col_begin, col_end = self.get_roi_col_range()
    except Exception:
        roi_ranges = []
        col_begin = col_end = None

    parts = [f"View: {_view_label(self)}"]
    if roi_ranges and col_begin is not None:
        parts.append(f"ROI rows {_format_row_ranges(roi_ranges)}")
        parts.append(f"cols {col_begin}-{col_end}")

    bg_enabled = bool(getattr(self, 'current_spectrum_bg_removed', getattr(self, 'bg_subtraction_enabled', False)))
    gap_enabled = bool(getattr(self, 'gap_correction_enabled', False))
    trace_enabled = bool(getattr(self, 'current_spectrum_trace_extracted', False))
    gap_columns = getattr(self, 'current_spectrum_gap_columns', [])

    parts.append("BG ON" if bg_enabled else "BG OFF")
    if gap_enabled:
        gap_text = "Gap ON"
        if gap_columns:
            gap_text += f" ({len(gap_columns)} cols)"
    else:
        gap_text = "Gap OFF"
    parts.append(gap_text)
    if trace_enabled:
        parts.append("Trace ON")

    self.spectrum_status_label.config(text=" | ".join(parts))


def _set_spectrum_view_mode(self, mode):
    """Update compact view-selector button styling, if present."""
    buttons = getattr(self, 'spectrum_view_buttons', {})
    for button_mode, button in buttons.items():
        style = 'CompactActive.TButton' if button_mode == mode else 'Compact.TButton'
        button.config(style=style)


def _coerce_mask(mask, expected_len):
    if mask is None:
        return np.zeros(expected_len, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.size != expected_len:
        return np.zeros(expected_len, dtype=bool)
    return mask


def _plot_gap_marker_group(self, x_data, marker_y, mask, color, label, marker='|'):
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return
    self.ax_spectrum.scatter(
        x_data[indices],
        np.full(indices.size, marker_y, dtype=float),
        s=85,
        color=color,
        marker=marker,
        alpha=0.78,
        linewidths=1.2,
        zorder=5,
        label=label,
    )


def _spectrum_point_scale(self):
    return float(getattr(self, 'spectrum_point_scale', 1.0))


def _spectrum_points(self, value):
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        numeric_value = float(plt.rcParams.get('font.size', 10))
    return numeric_value * _spectrum_point_scale(self)


def _set_spectrum_container_labels(self, xlabel, ylabel):
    if hasattr(self, 'spectrum_xlabel_label'):
        self.spectrum_xlabel_label.config(text=xlabel)
    if hasattr(self, 'spectrum_ylabel_text'):
        self.spectrum_ylabel_text = ylabel
    if hasattr(self, 'update_spectrum_ylabel'):
        self.update_spectrum_ylabel()


def _ensure_pyqtgraph_spectrum_viewer(self):
    if not pyqtgraph_spectrum.pyqtgraph_available():
        messagebox.showwarning(
            "Sharp Spectrum",
            "pyqtgraph with a Qt binding is required for the sharp spectrum viewer.",
        )
        return None
    viewer = getattr(self, 'pyqtgraph_spectrum_viewer', None)
    if viewer is None:
        viewer = pyqtgraph_spectrum.PyQtGraphSpectrumViewer()
        self.pyqtgraph_spectrum_viewer = viewer
    return viewer


def _use_pyqtgraph_primary_spectrum(self):
    return bool(getattr(self, 'use_pyqtgraph_spectrum_as_primary', False))


def _schedule_pyqtgraph_event_pump(self):
    if getattr(self, '_pyqtgraph_event_pump_active', False):
        return
    if not hasattr(self, 'root'):
        return

    self._pyqtgraph_event_pump_active = True

    def _pump():
        viewer = getattr(self, 'pyqtgraph_spectrum_viewer', None)
        if viewer is not None and viewer.is_visible():
            viewer.process_events()
            self.root.after(33, _pump)
            return
        self._pyqtgraph_event_pump_active = False

    self.root.after(33, _pump)


def _sync_pyqtgraph_spectrum_view(self, xlabel="Column Index", ylabel="Normalized intensity", show_legend=True):
    viewer = getattr(self, 'pyqtgraph_spectrum_viewer', None)
    if viewer is None and _use_pyqtgraph_primary_spectrum(self):
        viewer = _ensure_pyqtgraph_spectrum_viewer(self)
        if viewer is not None:
            viewer.show()
    elif viewer is not None and _use_pyqtgraph_primary_spectrum(self) and not viewer.is_visible():
        viewer.show()
    if viewer is None or not viewer.is_visible():
        return
    viewer.update_from_axes(self.ax_spectrum, xlabel=xlabel, ylabel=ylabel, show_legend=show_legend)
    _schedule_pyqtgraph_event_pump(self)


def open_pyqtgraph_spectrum_view(self):
    """Open a pyqtgraph spectrum viewer synchronized from the current spectrum axes."""
    viewer = _ensure_pyqtgraph_spectrum_viewer(self)
    if viewer is None:
        return
    viewer.show()
    _schedule_pyqtgraph_event_pump(self)
    if hasattr(self, 'ax_spectrum') and pyqtgraph_spectrum.axes_have_visible_data(self.ax_spectrum):
        xlabel = getattr(self, 'spectrum_xlabel_label', None)
        ylabel = getattr(self, 'spectrum_ylabel_text', "Normalized intensity")
        xlabel_text = xlabel.cget("text") if xlabel is not None else "Column Index"
        viewer.update_from_axes(self.ax_spectrum, xlabel=xlabel_text, ylabel=ylabel, show_legend=True)
    else:
        messagebox.showinfo("Sharp Spectrum", "Plot a spectrum first, then this window will mirror it.")


def _spectrum_canvas_size(self, event=None):
    fallback_width = int(getattr(self, 'spectrum_initial_width_px', 600))
    fallback_height = int(getattr(self, 'spectrum_initial_height_px', 400))
    if event is not None:
        width = int(getattr(event, 'width', fallback_width))
        height = int(getattr(event, 'height', fallback_height))
    elif hasattr(self, 'canvas_spectrum'):
        widget = self.canvas_spectrum.get_tk_widget()
        width = int(widget.winfo_width())
        height = int(widget.winfo_height())
    else:
        width = fallback_width
        height = fallback_height
    if width <= 1:
        width = fallback_width
    if height <= 1:
        height = fallback_height
    return width, height


def _configure_spectrum_render_target(self, event=None):
    if not hasattr(self, 'fig_spectrum'):
        return
    width_px, height_px = _spectrum_canvas_size(self, event)
    dpi = float(getattr(self, 'spectrum_fig_dpi', 300))
    self.fig_spectrum.set_dpi(dpi)
    self.fig_spectrum.set_size_inches(width_px / dpi, height_px / dpi, forward=False)
    _apply_spectrum_subplot_layout(self, width_px, height_px)
    self.spectrum_canvas_width_px = width_px
    self.spectrum_canvas_height_px = height_px


def _margin_fraction(pixels, total_pixels, minimum, maximum):
    if total_pixels <= 0:
        return minimum
    return min(max(float(pixels) / float(total_pixels), minimum), maximum)


def _apply_spectrum_subplot_layout(self, width_px=None, height_px=None):
    if not hasattr(self, 'fig_spectrum'):
        return
    if width_px is None or height_px is None:
        width_px, height_px = _spectrum_canvas_size(self)
    margins = getattr(
        self,
        'spectrum_subplot_margins_px',
        {'left': 45, 'bottom': 24, 'right': 10, 'top': 10},
    )
    left = _margin_fraction(margins.get('left', 45), width_px, 0.055, 0.24)
    right_margin = _margin_fraction(margins.get('right', 10), width_px, 0.012, 0.10)
    bottom = _margin_fraction(margins.get('bottom', 24), height_px, 0.07, 0.22)
    top_margin = _margin_fraction(margins.get('top', 10), height_px, 0.025, 0.12)
    right = 1.0 - right_margin
    top = 1.0 - top_margin

    if right - left < 0.42:
        center = (left + right) / 2.0
        left = max(0.075, center - 0.21)
        right = min(0.982, center + 0.21)
    if top - bottom < 0.42:
        center = (bottom + top) / 2.0
        bottom = max(0.12, center - 0.21)
        top = min(0.975, center + 0.21)

    self.fig_spectrum.subplots_adjust(left=left, bottom=bottom, right=right, top=top)


def _format_scaled_spectrum_tick(value, _position):
    scaled = value * 100.0
    if abs(scaled) < 1e-12:
        return "0"
    return f"{scaled:g}"


def _format_plain_spectrum_tick(value, _position):
    if abs(value) < 1e-12:
        return "0"
    return f"{value:g}"


def _apply_spectrum_y_tick_scale(self):
    if not hasattr(self, 'ax_spectrum'):
        return
    if getattr(self, '_spectrum_y_scale_text', None) is not None:
        try:
            self._spectrum_y_scale_text.remove()
        except ValueError:
            pass
        self._spectrum_y_scale_text = None

    bottom, top = self.ax_spectrum.get_ylim()
    max_abs = max(abs(float(bottom)), abs(float(top)))
    if not np.isfinite(max_abs) or max_abs <= 0.0 or max_abs > 1.0:
        self.ax_spectrum.yaxis.set_major_formatter(FuncFormatter(_format_plain_spectrum_tick))
        self.ax_spectrum.yaxis.get_offset_text().set_visible(False)
        return

    self.ax_spectrum.yaxis.set_major_formatter(FuncFormatter(_format_scaled_spectrum_tick))
    self.ax_spectrum.yaxis.get_offset_text().set_visible(False)
    self._spectrum_y_scale_text = self.ax_spectrum.text(
        -0.055,
        1.0,
        r'$\times 10^{-2}$',
        transform=self.ax_spectrum.transAxes,
        ha='left',
        va='top',
        fontsize=_spectrum_points(self, 10),
        family='Arial',
        color='0.20',
        clip_on=False,
    )


def on_spectrum_canvas_configure(self, event=None):
    _configure_spectrum_render_target(self, event)
    if hasattr(self, 'canvas_spectrum'):
        self.canvas_spectrum.draw_idle()


def _scale_spectrum_artist_units(self):
    if not hasattr(self, 'ax_spectrum'):
        return
    scale = _spectrum_point_scale(self)
    for line in self.ax_spectrum.lines:
        base_width = getattr(line, '_ixe_base_linewidth', None)
        if base_width is None:
            base_width = line.get_linewidth()
            setattr(line, '_ixe_base_linewidth', base_width)
        line.set_linewidth(max(float(base_width) * scale, 0.1))

        base_marker_size = getattr(line, '_ixe_base_markersize', None)
        if base_marker_size is None:
            base_marker_size = line.get_markersize()
            setattr(line, '_ixe_base_markersize', base_marker_size)
        line.set_markersize(float(base_marker_size) * scale)

    for collection in self.ax_spectrum.collections:
        if hasattr(collection, 'get_sizes') and hasattr(collection, 'set_sizes'):
            base_sizes = getattr(collection, '_ixe_base_sizes', None)
            if base_sizes is None:
                base_sizes = collection.get_sizes().copy()
                setattr(collection, '_ixe_base_sizes', base_sizes)
            if base_sizes.size:
                collection.set_sizes(base_sizes * scale * scale)

        if hasattr(collection, 'get_linewidths') and hasattr(collection, 'set_linewidths'):
            base_widths = getattr(collection, '_ixe_base_linewidths', None)
            if base_widths is None:
                base_widths = collection.get_linewidths()
                setattr(collection, '_ixe_base_linewidths', base_widths)
            collection.set_linewidths(np.asarray(base_widths, dtype=float) * scale)


def _plot_gap_markers(self, x_data, y_data):
    """Mark corrected columns near the plot baseline, split by ROI/background source."""
    if not hasattr(self, 'spect_processor') or self.spect_processor is None:
        return
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    if x_data.size != y_data.size or y_data.size == 0:
        return

    combined_mask = _coerce_mask(getattr(self.spect_processor, 'last_gap_mask', None), y_data.size)
    if not np.any(combined_mask):
        return

    roi_mask = _coerce_mask(getattr(self.spect_processor, 'last_roi_gap_mask', None), y_data.size)
    bg_mask = _coerce_mask(getattr(self.spect_processor, 'last_background_gap_mask', None), y_data.size)
    edge_mask = _coerce_mask(getattr(self.spect_processor, 'last_edge_mask', None), y_data.size)
    roi_edge_mask = _coerce_mask(getattr(self.spect_processor, 'last_roi_edge_mask', None), y_data.size)
    bg_edge_mask = _coerce_mask(getattr(self.spect_processor, 'last_background_edge_mask', None), y_data.size)
    if not np.any(roi_mask) and not np.any(bg_mask):
        roi_mask = combined_mask & ~edge_mask

    finite_y = y_data[np.isfinite(y_data)]
    if finite_y.size:
        y_min = float(np.nanmin(finite_y))
        y_max = float(np.nanmax(finite_y))
        span = max(y_max - y_min, np.finfo(float).eps)
        marker_y = y_min - 0.08 * span
        self._spectrum_gap_marker_y = marker_y
        self._spectrum_gap_marker_pad = 0.04 * span
    else:
        marker_y = 0.0
        self._spectrum_gap_marker_y = marker_y
        self._spectrum_gap_marker_pad = 0.02

    edge_both = roi_edge_mask & bg_edge_mask
    roi_edge_only = roi_edge_mask & ~bg_edge_mask
    bg_edge_only = bg_edge_mask & ~roi_edge_mask
    both_mask = roi_mask & bg_mask & ~edge_mask
    roi_only = roi_mask & ~bg_mask & ~edge_mask
    bg_only = bg_mask & ~roi_mask & ~edge_mask
    combined_only = combined_mask & ~(roi_mask | bg_mask | edge_mask)

    _plot_gap_marker_group(self, x_data, marker_y, roi_only, "#d62728", "ROI gap fill")
    _plot_gap_marker_group(self, x_data, marker_y, bg_only, "#ff7f0e", "BG gap fill")
    _plot_gap_marker_group(self, x_data, marker_y, both_mask, "#9467bd", "ROI+BG gap fill")
    _plot_gap_marker_group(self, x_data, marker_y, combined_only, "#d62728", "CCD gap fill")
    _plot_gap_marker_group(self, x_data, marker_y, roi_edge_only, "#8c1d18", "ROI edge fill")
    _plot_gap_marker_group(self, x_data, marker_y, bg_edge_only, "#b15928", "BG edge fill")
    _plot_gap_marker_group(self, x_data, marker_y, edge_both, "#6a3d9a", "ROI+BG edge fill")


def _finish_spectrum_axes(self, xlabel="Column Index", ylabel="Normalized intensity", show_legend=True, view_mode=None, view_label=None):
    _configure_spectrum_render_target(self)
    if view_mode is not None:
        _set_spectrum_view_mode(self, view_mode)
    if view_label is not None:
        _update_spectrum_status(self, view_label)
    else:
        _update_spectrum_status(self)
    _set_spectrum_container_labels(self, xlabel, ylabel)
    self.ax_spectrum.set_title("")
    self.ax_spectrum.set_xlabel("")
    self.ax_spectrum.set_ylabel("")
    self.ax_spectrum.tick_params(
        axis='both',
        which='major',
        width=_spectrum_points(self, 0.5),
        length=_spectrum_points(self, 2.5),
        pad=_spectrum_points(self, 1.5),
        labelsize=_spectrum_points(self, plt.rcParams.get('xtick.labelsize', 10)),
    )
    self.ax_spectrum.locator_params(axis='y', nbins=5)
    marker_y = getattr(self, '_spectrum_gap_marker_y', None)
    if marker_y is not None:
        bottom, top = self.ax_spectrum.get_ylim()
        pad = float(getattr(self, '_spectrum_gap_marker_pad', 0.0))
        self.ax_spectrum.set_ylim(min(bottom, marker_y - pad), top)
        delattr(self, '_spectrum_gap_marker_y')
        if hasattr(self, '_spectrum_gap_marker_pad'):
            delattr(self, '_spectrum_gap_marker_pad')
    _apply_spectrum_y_tick_scale(self)
    for spine in self.ax_spectrum.spines.values():
        spine.set_linewidth(_spectrum_points(self, 0.5))
    _scale_spectrum_artist_units(self)
    if show_legend:
        handles, labels = self.ax_spectrum.get_legend_handles_labels()
        if handles:
            self.ax_spectrum.legend(
                fontsize=_spectrum_points(self, plt.rcParams.get('legend.fontsize', 10)),
                loc='best',
                frameon=False,
            )
    self.canvas_spectrum.draw()
    _sync_pyqtgraph_spectrum_view(self, xlabel=xlabel, ylabel=ylabel, show_legend=show_legend)


def plot_roi_spectrum(self, line_color=None, line_style=None, line_width=None):
    """Plot the spectrum for the selected ROI with custom line style, color, and width."""
    if not hasattr(self, 'spect_processor'):
        print("Error: Process the image first")
        return

    try:
        # Clear previous spectrum plot
        self.ax_spectrum.clear()

        # Get spectrum data
        _configure_gap_correction(self)
        _sync_background_roi(self)
        spectrum, _rois, _column_begin, _column_end = _extract_active_roi_spectrum(self)
        # Normalize the spectrum before plotting
        norm_spectrum = self.spect_processor.get_norm_spectrum(spectrum)

        x_data = _current_spectrum_x_axis(self, len(norm_spectrum), moving_average=0, prefer_stored=False)

        # Store the last displayed ROI spectrum (normalized, raw)
        self.last_spectrum_roi = norm_spectrum.copy()
        self.current_spectrum_moving_avg = 0
        self.current_spectrum_bg_removed = self.bg_subtraction_enabled
        if hasattr(self, 'last_peak_fit_profile'):
            delattr(self, 'last_peak_fit_profile')
        _clear_peak_fit_results_box(self)
        _store_gap_state(self, norm_spectrum, x_data)

        # Get user-selected values for line color, style, and width
        line_color = line_color or self.line_color.get()
        line_style = line_style or self.line_style.get()
        try:
            line_width = float(line_width) if line_width else float(self.line_width.get())
        except ValueError:
            print("Invalid line width value. Using default.")
            line_width = 0.5  # Default value if invalid input

        # Plot the spectrum with the specified color, style, and width.
        self.ax_spectrum.plot(
            x_data,
            norm_spectrum,
            color=line_color,
            linestyle=line_style,
            linewidth=line_width,
            label=f"Run {self.run_number_label.cget('text').split(': ')[1]}",
        )
        _plot_gap_markers(self, x_data, norm_spectrum)
        _finish_spectrum_axes(self, view_mode='corrected', view_label='Corrected spectrum')
    except Exception as e:
        print(f"Error plotting spectrum: {e}")


def show_original_spectrum(self):
    """Show only the uncorrected extracted spectrum in the spectrum panel."""
    if not _ensure_spectrum_available(self):
        return
    y_data = getattr(self, 'last_spectrum_roi_uncorrected', self.last_spectrum_roi)
    y_data = np.asarray(y_data, dtype=float)
    x_data = _current_spectrum_x_axis(self, len(y_data))
    self.ax_spectrum.clear()
    self.ax_spectrum.plot(
        x_data,
        y_data,
        color='0.35',
        linewidth=0.8,
        label="BG-subtracted" if getattr(self, 'current_spectrum_bg_removed', False) else "Raw",
    )
    _finish_spectrum_axes(self, view_mode='original', view_label='Raw/BG spectrum')


def show_gap_corrected_spectrum(self):
    """Show only the current gap-corrected spectrum in the spectrum panel."""
    if not _ensure_spectrum_available(self):
        return
    y_data = np.asarray(self.last_spectrum_roi, dtype=float)
    x_data = _current_spectrum_x_axis(self, len(y_data))
    line_color = self.line_color.get() if hasattr(self, 'line_color') else 'black'
    self.ax_spectrum.clear()
    self.ax_spectrum.plot(
        x_data,
        y_data,
        color=line_color,
        linewidth=0.8,
        label="Gap-corrected",
    )
    _plot_gap_markers(self, x_data, y_data)
    _finish_spectrum_axes(self, view_mode='corrected', view_label='Corrected spectrum')


def show_peak_fit_view(self):
    """Show only the fitted peak profile and fitted components."""
    fit_result = _ensure_peak_fit_available(self)
    if fit_result is None:
        return
    line_color = self.line_color.get() if hasattr(self, 'line_color') else 'black'
    self.ax_spectrum.clear()
    self.ax_spectrum.plot(
        fit_result.x,
        fit_result.normalized_fit,
        color=line_color,
        linewidth=1.6,
        label="Peak fit",
    )

    scale = _component_scale(fit_result)
    if scale is not None:
        colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
        for index, peak_curve in enumerate(fit_result.peak_curves, start=1):
            component = np.asarray(peak_curve, dtype=float) / scale
            color = colors[index % len(colors)] if colors else None
            label = fit_result.peak_labels[index - 1] if index - 1 < len(fit_result.peak_labels) else f"Peak {index}"
            self.ax_spectrum.plot(
                fit_result.x,
                component,
                linestyle='--',
                linewidth=0.9,
                color=color,
                alpha=0.9,
                label=label,
            )

    _finish_spectrum_axes(self, view_mode='fit', view_label='Peak fit')


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
        
        # Redraw the canvas with the updated line color.
        _configure_spectrum_render_target(self)
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
        _configure_gap_correction(self)
        _sync_background_roi(self)
        spectrum, _rois, _column_begin, _column_end = _extract_active_roi_spectrum(self)

        # Apply the moving average to the spectrum data
        smoothed_spectrum = self.spect_processor.moving_average(spectrum)
        norm_smoothed_spectrum = self.spect_processor.get_norm_spectrum(smoothed_spectrum)
        x_data = _current_spectrum_x_axis(self, len(norm_smoothed_spectrum), moving_average=n_moveavg, prefer_stored=False)

        # Store the last displayed ROI spectrum (normalized, smoothed)
        self.last_spectrum_roi = norm_smoothed_spectrum.copy()
        self.current_spectrum_moving_avg = self.parm['n_moveavg']
        self.current_spectrum_bg_removed = self.bg_subtraction_enabled
        if hasattr(self, 'last_peak_fit_profile'):
            delattr(self, 'last_peak_fit_profile')
        _clear_peak_fit_results_box(self)
        _store_gap_state(self, norm_smoothed_spectrum, x_data)

        # Plot the smoothed spectrum in the spectrum axes
        self.ax_spectrum.clear()  # Clear the previous plot
        
        # Update title to reflect if BG is removed
        self.ax_spectrum.set_title("")
        
        self.ax_spectrum.plot(
            x_data,
            norm_smoothed_spectrum,
            color=self.line_color.get(),
            linestyle=self.line_style.get(),
            linewidth=float(self.line_width.get()),
            label=f"Run {self.run_number_label.cget('text').split(': ')[1]}",
        )
        _plot_gap_markers(self, x_data, norm_smoothed_spectrum)
        _finish_spectrum_axes(self, view_mode='corrected', view_label='Smoothed spectrum')

    except Exception as e:
        print(f"Error applying moving average: {e}")

def show_peak_fit_profile(self):
    """Fit the displayed ROI spectrum and overlay the fit profile on the original spectrum."""
    if not hasattr(self, 'last_spectrum_roi'):
        messagebox.showwarning("Peak Fit", "Plot a spectrum first.")
        return

    fit_config = _get_spectrum_peak_fit_config(self)
    if fit_config is None:
        return

    y_data = np.asarray(self.last_spectrum_roi, dtype=float)
    x_data = _current_spectrum_x_axis(self, len(y_data))
    try:
        fit_result = fit_spectrum(x_data, y_data, fit_config)
    except PeakFitError as exc:
        messagebox.showwarning("Peak Fit", str(exc))
        print(f"Peak fitting failed: {exc}")
        return

    self.last_peak_fit_profile = fit_result
    _update_peak_fit_results_box(self, fit_result)
    line_color = self.line_color.get() if hasattr(self, 'line_color') else 'black'
    background_y = getattr(self, 'last_spectrum_roi_uncorrected', y_data)
    if len(background_y) != len(y_data):
        background_y = y_data
    self.ax_spectrum.clear()
    self.ax_spectrum.plot(
        x_data,
        background_y,
        color='0.75',
        linewidth=1.0,
        alpha=0.8,
        label="BG-subtracted" if getattr(self, 'current_spectrum_bg_removed', False) else "Raw",
    )
    if getattr(self, 'current_spectrum_gap_corrected', False):
        self.ax_spectrum.plot(
            x_data,
            y_data,
            color='0.45',
            linewidth=0.8,
            alpha=0.9,
            label="Gap-corrected",
        )
    self.ax_spectrum.plot(
        fit_result.x,
        fit_result.normalized_fit,
        color=line_color,
        linewidth=1.6,
        label="Peak fit",
    )

    scale = _component_scale(fit_result)
    if scale is not None:
        colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
        for index, peak_curve in enumerate(fit_result.peak_curves, start=1):
            component = np.asarray(peak_curve, dtype=float) / scale
            color = colors[index % len(colors)] if colors else None
            label = fit_result.peak_labels[index - 1] if index - 1 < len(fit_result.peak_labels) else f"Peak {index}"
            self.ax_spectrum.plot(
                fit_result.x,
                component,
                linestyle='--',
                linewidth=0.9,
                color=color,
                alpha=0.9,
                label=label,
            )

    _finish_spectrum_axes(self, view_mode='fit', view_label='Peak fit')
    print(
        "Peak fit centers: {0}".format(
            ", ".join(
                f"{label}={center:.2f}"
                for label, center in zip(fit_result.peak_labels, fit_result.centers)
            )
        )
    )

def toggle_gap_correction(self):
    """Toggle mask-aware CCD gap handling before spectrum normalization."""
    gap_defaults = self.parm.setdefault('ccd_gap', {}) if hasattr(self, 'parm') else {}
    current = getattr(self, 'gap_correction_enabled', gap_defaults.get('enabled', True))
    self.gap_correction_enabled = not bool(current)
    gap_defaults['enabled'] = self.gap_correction_enabled
    if hasattr(self, 'spect_processor') and self.spect_processor is not None:
        _configure_gap_correction(self)
    if hasattr(self, 'gap_toggle'):
        _set_toggle_button_state(
            self.gap_toggle,
            self.gap_correction_enabled,
            "Gap Mask (ON)",
            "Gap Mask (OFF)",
        )
    if hasattr(self, 'last_spectrum_roi'):
        self.plot_roi_spectrum()

def toggle_trace_extraction(self):
    """Toggle column-by-column trace-following extraction."""
    trace_defaults = self.parm.setdefault('trace_extraction', {}) if hasattr(self, 'parm') else {}
    current = getattr(self, 'trace_extraction_enabled', trace_defaults.get('enabled', False))
    self.trace_extraction_enabled = not bool(current)
    trace_defaults['enabled'] = self.trace_extraction_enabled
    if hasattr(self, 'spect_processor') and self.spect_processor is not None and hasattr(self.spect_processor, 'toggle_trace_extraction'):
        self.spect_processor.toggle_trace_extraction(self.trace_extraction_enabled)
        self.spect_processor.set_trace_extraction_params(
            search_margin=trace_defaults.get('search_margin', 60)
        )
    if hasattr(self, 'trace_toggle'):
        text = "Trace ROI (ON)" if self.trace_extraction_enabled else "Trace ROI (OFF)"
        self.trace_toggle.config(text=text)
    if hasattr(self, 'last_spectrum_roi'):
        self.plot_roi_spectrum()

def toggle_background_removal(self):
    """Toggle background subtraction on/off."""
    self.bg_subtraction_enabled = not self.bg_subtraction_enabled
    if self.bg_subtraction_enabled:
        _set_toggle_button_state(self.bg_toggle, True, "BG Remove (ON)", "BG Remove (OFF)")
    else:
        _set_toggle_button_state(self.bg_toggle, False, "BG Remove (ON)", "BG Remove (OFF)")
    if not hasattr(self, 'spect_processor'):
        print("Error: Process the image first")
        self.bg_subtraction_enabled = False
        _set_toggle_button_state(self.bg_toggle, False, "BG Remove (ON)", "BG Remove (OFF)")
        return
    try:
        bg_ranges = self.get_bg_row_ranges() if hasattr(self, 'get_bg_row_ranges') else [self.get_bg_row_range()]
        bg_col_start, bg_col_end = self.get_roi_col_range()
        if (not bg_ranges or bg_col_start >= bg_col_end or bg_col_start < 0):
            raise ValueError("Invalid background ROI coordinates")
        rois = [
            (row_start, row_end + 1, bg_col_start, bg_col_end + 1)
            for row_start, row_end in bg_ranges
            if row_end > row_start and row_start >= 0
        ]
        if not rois:
            raise ValueError("Invalid background ROI coordinates")
        if hasattr(self.spect_processor, 'set_background_rois'):
            self.spect_processor.set_background_rois(rois)
        else:
            row_start, row_end, col_start, col_end = rois[0]
            self.spect_processor.set_background_roi(row_start, row_end, col_start, col_end)
        self.spect_processor.toggle_background_subtraction(self.bg_subtraction_enabled)
        if hasattr(self, 'last_spectrum_roi'):
            self.plot_roi_spectrum()
    except ValueError as ve:
        print(f"Invalid background ROI: {ve}")
        self.bg_subtraction_enabled = False
        _set_toggle_button_state(self.bg_toggle, False, "BG Remove (ON)", "BG Remove (OFF)")
    except Exception as e:
        print(f"Error toggling background removal: {e}")
        self.bg_subtraction_enabled = False
        _set_toggle_button_state(self.bg_toggle, False, "BG Remove (ON)", "BG Remove (OFF)")

def save_spectrum_data(self):
    """Save the spectrum currently displayed in the GUI."""
    if not hasattr(self, 'last_spectrum_roi'):
        print("Error: Plot a spectrum first")
        return
    
    try:
        y_data = self.last_spectrum_roi.copy()
        x_data = _current_spectrum_x_axis(self, len(y_data))
        moving_avg = getattr(self, 'current_spectrum_moving_avg', 0)
        bg_removed = getattr(self, 'current_spectrum_bg_removed', self.bg_subtraction_enabled)
        gap_corrected = getattr(self, 'current_spectrum_gap_corrected', False)
        gap_columns = getattr(self, 'current_spectrum_gap_columns', [])
        trace_extracted = getattr(self, 'current_spectrum_trace_extracted', False)
        roi_ranges = _active_roi_ranges(self)
        roi_ranges_text = _format_row_ranges(roi_ranges)
        column_begin, column_end = self.get_roi_col_range()
        bg_ranges = self.get_bg_row_ranges() if hasattr(self, 'get_bg_row_ranges') else [self.get_bg_row_range()]
        bg_ranges_text = "; ".join(f"{begin}-{end}" for begin, end in bg_ranges) if bg_ranges else "None"
        
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
        )
        
        if not save_path:
            return  # User cancelled

        with open(save_path, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter='	')
            writer.writerow([f"Spectrum (Moving Avg: {moving_avg})"])
            writer.writerow([f"Background removed: {bg_removed}"])
            writer.writerow([f"CCD gap corrected: {gap_corrected}"])
            if gap_columns:
                writer.writerow([f"CCD gap columns: {','.join(str(value) for value in gap_columns)}"])
            writer.writerow([f"Trace extracted: {trace_extracted}"])
            writer.writerow([f"ROI Rows: {roi_ranges_text}"])
            writer.writerow([f"ROI Columns: {column_begin}-{column_end}"])
            writer.writerow([f"BG Rows: {bg_ranges_text}"])
            writer.writerow(['X', 'Y'])
            for i in range(len(x_data)):
                writer.writerow([x_data[i], y_data[i]])
                
            print(f"Spectrum saved to {save_path}")
        
    except Exception as e:
        print(f"Error saving spectrum data: {e}")
    
def save_smoothed_spectrum_data(self):
    """Backward-compatible alias for saving the displayed spectrum."""
    return save_spectrum_data(self)


def save_peak_fit_parameters(self):
    """Save fitted peak position, width, area, and Lorentzian fraction."""
    if not hasattr(self, 'last_peak_fit_profile'):
        messagebox.showwarning("Save params", "Run Peak Fit before saving peak parameters.")
        return

    try:
        fit_result = self.last_peak_fit_profile
        filename = getattr(self, 'original_filepath', '').split("/")[-1]
        run_number_match = re.search(r"Run_(\d+)", filename)
        save_filename = (
            f"Run_{run_number_match.group(1)}_peak_params.txt"
            if run_number_match
            else "peak_fit_params.txt"
        )
        save_path = filedialog.asksaveasfilename(
            initialfile=save_filename,
            defaultextension='.txt',
        )
        if not save_path:
            return

        rows = _peak_fit_parameter_rows(fit_result)
        model = getattr(fit_result, 'peak_shape', 'pseudo_voigt')
        with open(save_path, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter='	')
            writer.writerow([f"Peak fit model: {_peak_fit_model_label(model)}"])
            if model == 'pseudo_voigt':
                writer.writerow([f"Lorentzian components: {len(rows)}"])
            writer.writerow(["Peak", "Model", "Lorentzian fraction", "Position", "Width", "Area"])
            for row in rows:
                label, model_label, fraction, center, width, area = row
                writer.writerow([label, model_label, fraction, center, width, area])
        print(f"Peak fit parameters saved to {save_path}")
    except Exception as e:
        print(f"Error saving peak fit parameters: {e}")

def save_peak_fit_data(self):
    """Save the most recent fitted peak profile and components."""
    if not hasattr(self, 'last_peak_fit_profile'):
        messagebox.showwarning("Save pkfit", "Run Peak Fit before saving the fitted profile.")
        return

    try:
        fit_result = self.last_peak_fit_profile
        x_data = np.asarray(fit_result.x, dtype=float)
        raw_y = np.asarray(fit_result.raw_y, dtype=float)
        fit_y = np.asarray(fit_result.normalized_fit, dtype=float)
        if x_data.size == 0 or fit_y.size == 0:
            messagebox.showwarning("Save pkfit", "Peak fit profile is empty.")
            return

        scale = _component_scale(fit_result)
        components = []
        for peak_curve in fit_result.peak_curves:
            if scale is None:
                components.append(np.full_like(fit_y, np.nan, dtype=float))
            else:
                components.append(np.asarray(peak_curve, dtype=float) / scale)

        moving_avg = getattr(self, 'current_spectrum_moving_avg', 0)
        bg_removed = getattr(self, 'current_spectrum_bg_removed', self.bg_subtraction_enabled)
        gap_corrected = getattr(self, 'current_spectrum_gap_corrected', False)
        gap_columns = getattr(self, 'current_spectrum_gap_columns', [])
        trace_extracted = getattr(self, 'current_spectrum_trace_extracted', False)
        roi_ranges = _active_roi_ranges(self)
        roi_ranges_text = _format_row_ranges(roi_ranges)
        column_begin, column_end = self.get_roi_col_range()
        bg_ranges = self.get_bg_row_ranges() if hasattr(self, 'get_bg_row_ranges') else [self.get_bg_row_range()]
        bg_ranges_text = "; ".join(f"{begin}-{end}" for begin, end in bg_ranges) if bg_ranges else "None"

        filename = self.original_filepath.split("/")[-1]
        run_number_match = re.search(r"Run_(\d+)", filename)
        if run_number_match:
            run_number = run_number_match.group(1)
        else:
            print("Error: Run number not found in filename.")
            return

        save_filename = f"Run_{run_number}_pkfit.txt"
        save_path = filedialog.asksaveasfilename(
            initialfile=save_filename,
            defaultextension='.txt',
        )
        if not save_path:
            return

        with open(save_path, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter='	')
            writer.writerow([f"Peak fitting spectrum (Moving Avg: {moving_avg})"])
            writer.writerow([f"Peak fit model: {_peak_fit_model_label(getattr(fit_result, 'peak_shape', 'pseudo_voigt'))}"])
            writer.writerow([f"Background removed: {bg_removed}"])
            writer.writerow([f"CCD gap corrected: {gap_corrected}"])
            if gap_columns:
                writer.writerow([f"CCD gap columns: {','.join(str(value) for value in gap_columns)}"])
            writer.writerow([f"Trace extracted: {trace_extracted}"])
            writer.writerow([f"ROI Rows: {roi_ranges_text}"])
            writer.writerow([f"ROI Columns: {column_begin}-{column_end}"])
            writer.writerow([f"BG Rows: {bg_ranges_text}"])
            writer.writerow([f"Fit X Range: {x_data[0]:.6g}-{x_data[-1]:.6g}"])
            display_peak_labels = [_display_peak_label(label) for label in fit_result.peak_labels]
            writer.writerow([
                "Centers",
                *[
                    f"{label}={center:.6g}"
                    for label, center in zip(display_peak_labels, fit_result.centers)
                ],
            ])
            writer.writerow(["Peak parameters"])
            writer.writerow(["Peak", "Lorentzian fraction", "Position", "Width", "Area"])
            for label, _model_label, fraction, center, width, area in _peak_fit_parameter_rows(fit_result):
                writer.writerow([label, fraction, center, width, area])
            writer.writerow(["X", "Raw Y", "Peak fit", *display_peak_labels])
            for row_index in range(len(x_data)):
                writer.writerow(
                    [
                        x_data[row_index],
                        raw_y[row_index],
                        fit_y[row_index],
                        *[component[row_index] for component in components],
                    ]
                )

        print(f"Peak fit spectrum saved to {save_path}")

    except Exception as e:
        print(f"Error saving peak fit data: {e}")

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
            filetypes=(
                ("PNG image", "*.png"),
                ("SVG vector", "*.svg"),
                ("PDF vector", "*.pdf"),
                ("All files", "*.*"),
            ),
        )

        if not save_path:
            return

        root, extension = os.path.splitext(save_path)
        if not extension:
            extension = '.png'
            save_path = f"{root}{extension}"
        extension = extension.lower()

        _configure_spectrum_render_target(self)
        if extension in ('.svg', '.pdf'):
            self.fig_spectrum.savefig(save_path, format=extension.lstrip('.'))
        else:
            export_dpi = max(300, int(getattr(self, 'spectrum_fig_dpi', 300)))
            self.fig_spectrum.savefig(save_path, dpi=export_dpi)
        print(f"Spectrum image saved to {save_path}")

    except Exception as e:
        print(f"Error saving spectrum image: {e}")
