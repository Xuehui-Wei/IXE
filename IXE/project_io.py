# -*- coding: utf-8 -*-
"""Project save/open helpers for IXE.

The project file is an HDF5 container with a JSON manifest plus numerical
arrays. This keeps analysis state portable without inventing many sidecar files.
"""

import json
import os
from datetime import datetime
from tkinter import filedialog, messagebox

import h5py
import numpy as np

try:
    from .peak_fitting import PeakFitResult
    from .spectrum_utils import SpectrumProcessor
    from . import iad_controls
    from . import spectrum_controls
except ImportError:
    try:
        from peak_fitting import PeakFitResult
        from spectrum_utils import SpectrumProcessor
        import iad_controls
        import spectrum_controls
    except ImportError:
        from IXE.peak_fitting import PeakFitResult
        from IXE.spectrum_utils import SpectrumProcessor
        from IXE import iad_controls
        from IXE import spectrum_controls


PROJECT_FORMAT = "IXE Project"
PROJECT_VERSION = 1


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _entry_text(widget, default=""):
    if widget is None:
        return default
    try:
        return widget.get()
    except Exception:
        return default


def _set_entry(widget, value):
    if widget is None:
        return
    try:
        widget.delete(0, "end")
        widget.insert(0, "" if value is None else str(value))
    except Exception:
        pass


def _write_array(group, name, values):
    if values is None:
        return False
    array = np.asarray(values)
    group.create_dataset(name, data=array, compression="gzip")
    return True


def _read_array(group, name, default=None):
    if group is None or name not in group:
        return default
    return np.asarray(group[name])


def _read_manifest(h5_file):
    raw = h5_file["manifest"][()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _current_run_number(self):
    label = self.run_number_label.cget("text") if hasattr(self, "run_number_label") else ""
    if ": " in label:
        return label.split(": ", 1)[1]
    return "project"


def _build_manifest(self):
    peak_fit_params = self.parm.setdefault("peak_fit", {})
    if hasattr(self, "peak_fit_model_var"):
        peak_fit_params["peak_shape"] = self.peak_fit_model_var.get()
    peak_count_text = _entry_text(getattr(self, "peak_fit_count", None), "").strip()
    if peak_count_text:
        try:
            peak_fit_params["n_peaks"] = int(peak_count_text)
        except ValueError:
            pass
    for attr_name, param_key in (("peak_fit_start", "x_min"), ("peak_fit_end", "x_max")):
        text = _entry_text(getattr(self, attr_name, None), "").strip()
        if not text:
            peak_fit_params[param_key] = None
            continue
        try:
            peak_fit_params[param_key] = float(text)
        except ValueError:
            pass
    references = []
    for line_name, line_data in getattr(self, "plotted_lines", {}).items():
        if not line_data.get("is_ref", False):
            continue
        iad_entry = line_data.get("iad_entry")
        st_iad_entry = line_data.get("st_iad_entry")
        checkbox_var = line_data.get("checkbox_var")
        visible = bool(checkbox_var.get()) if checkbox_var is not None else bool(line_data.get("visible", True))
        references.append(
            {
                "name": line_name,
                "color": line_data.get("color", "#000000"),
                "run_number": line_data.get("run_number", ""),
                "visible": visible,
                "is_ref": bool(line_data.get("is_ref", True)),
                "is_peak_fit": bool(line_data.get("is_peak_fit", False)),
                "source_column": line_data.get("source_column", "Y"),
                "x_axis_repaired": bool(line_data.get("x_axis_repaired", False)),
                "gap_corrected": bool(line_data.get("gap_corrected", False)),
                "tail_matched": bool(line_data.get("tail_matched", False)),
                "satellite_tail_matched": bool(line_data.get("satellite_tail_matched", False)),
                "iad_value": _entry_text(iad_entry),
                "satellite_iad_value": _entry_text(st_iad_entry),
            }
        )

    peak_fit_meta = None
    if hasattr(self, "last_peak_fit_profile"):
        fit = self.last_peak_fit_profile
        peak_fit_meta = {
            "peak_labels": list(getattr(fit, "peak_labels", [])),
            "centers": _json_safe(getattr(fit, "centers", [])),
            "sigmas": _json_safe(getattr(fit, "sigmas", [])),
            "amplitudes": _json_safe(getattr(fit, "amplitudes", [])),
            "fractions": _json_safe(getattr(fit, "fractions", [])),
            "widths": _json_safe(getattr(fit, "widths", [])),
            "peak_shape": getattr(fit, "peak_shape", "pseudo_voigt"),
            "physical_fit": bool(getattr(fit, "physical_fit", False)),
            "tail_baseline_enabled": bool(getattr(fit, "tail_baseline_enabled", False)),
            "tail_fraction": float(getattr(fit, "tail_fraction", 0.10)),
        }

    calibration_meta = {
        "filepath": getattr(self, "calibration_filepath", ""),
        "source_column": getattr(self, "calibration_source_column", ""),
        "fit_start": _entry_text(getattr(self, "cal_fit_start", None)),
        "fit_end": _entry_text(getattr(self, "cal_fit_end", None)),
        "energy_calibration": _json_safe(getattr(self, "energy_calibration", None)),
        "has_calibrated_display": bool(hasattr(self, "calibration_energy_axis")),
    }
    if hasattr(self, "calibration_fit"):
        fit = self.calibration_fit
        calibration_meta["fit"] = {
            "peak_labels": list(getattr(fit, "peak_labels", [])),
            "centers": _json_safe(getattr(fit, "centers", [])),
            "sigmas": _json_safe(getattr(fit, "sigmas", [])),
            "amplitudes": _json_safe(getattr(fit, "amplitudes", [])),
            "fractions": _json_safe(getattr(fit, "fractions", [])),
            "physical_fit": bool(getattr(fit, "physical_fit", False)),
            "tail_baseline_enabled": bool(getattr(fit, "tail_baseline_enabled", False)),
            "tail_fraction": float(getattr(fit, "tail_fraction", 0.10)),
        }

    return {
        "format": PROJECT_FORMAT,
        "version": PROJECT_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "original_filepath": getattr(self, "original_filepath", ""),
        "image_stack_files": _json_safe(getattr(self, "image_stack_files", [])),
        "image_stack_count": int(getattr(self, "image_stack_count", 1) or 1),
        "run_label": self.run_number_label.cget("text") if hasattr(self, "run_number_label") else "Run: Not Loaded",
        "run_label_spectrum": self.run_number_label_spectrum.cget("text") if hasattr(self, "run_number_label_spectrum") else "Run: Not Loaded",
        "parameters": _json_safe(getattr(self, "parm", {})),
        "ui": {
            "threshold": _json_safe(getattr(self, "parm", {}).get("threshold", 0)),
            "vmin": _entry_text(getattr(self, "vmin_entry", None)),
            "vmax": _entry_text(getattr(self, "vmax_entry", None)),
            "cmap": self.image_cmap.get() if hasattr(self, "image_cmap") else getattr(self, "parm", {}).get("cmap", "viridis"),
            "tilt": _entry_text(getattr(self, "tilt_entry", None)),
            "line_color": self.line_color.get() if hasattr(self, "line_color") else "black",
            "line_style": self.line_style.get() if hasattr(self, "line_style") else "-",
            "line_width": _entry_text(getattr(self, "line_width", None), "0.5"),
            "cross_begin": _entry_text(getattr(self, "cross_begin", None)),
            "cross_end": _entry_text(getattr(self, "cross_end", None)),
            "eye_ball_cross": _entry_text(getattr(self, "eye_ball_cross", None)),
            "bg_subtraction_enabled": bool(getattr(self, "bg_subtraction_enabled", False)),
            "gap_correction_enabled": bool(getattr(self, "gap_correction_enabled", True)),
            "trace_extraction_enabled": False,
            "current_spectrum_moving_avg": int(getattr(self, "current_spectrum_moving_avg", 0)),
            "current_spectrum_bg_removed": bool(getattr(self, "current_spectrum_bg_removed", False)),
            "current_spectrum_gap_corrected": bool(getattr(self, "current_spectrum_gap_corrected", False)),
            "current_spectrum_gap_columns": _json_safe(getattr(self, "current_spectrum_gap_columns", [])),
            "current_spectrum_trace_extracted": False,
            "calibration_filepath": getattr(self, "calibration_filepath", ""),
        },
        "references": references,
        "peak_fit": peak_fit_meta,
        "calibration": calibration_meta,
    }


def _write_project_arrays(self, h5_file):
    images = h5_file.create_group("images")
    _write_array(images, "raw", getattr(self, "imm", None))
    _write_array(images, "processed", getattr(self, "immm", None))

    masks = h5_file.create_group("masks")
    _write_array(masks, "raw_gap", getattr(self, "raw_gap_mask", None))
    _write_array(masks, "processed_gap", getattr(self, "processed_gap_mask", None))
    _write_array(masks, "detector_background", getattr(self, "processed_detector_background_mask", None))

    spectra = h5_file.create_group("spectra")
    _write_array(spectra, "last_spectrum_roi", getattr(self, "last_spectrum_roi", None))
    _write_array(spectra, "last_spectrum_roi_uncorrected", getattr(self, "last_spectrum_roi_uncorrected", None))

    if hasattr(self, "last_peak_fit_profile"):
        fit = self.last_peak_fit_profile
        fit_group = h5_file.create_group("peak_fit")
        _write_array(fit_group, "x", fit.x)
        _write_array(fit_group, "raw_y", fit.raw_y)
        _write_array(fit_group, "fit_y", getattr(fit, "fit_y", fit.raw_y))
        _write_array(fit_group, "tail_baseline", getattr(fit, "tail_baseline", None))
        _write_array(fit_group, "best_fit", fit.best_fit)
        _write_array(fit_group, "normalized_fit", fit.normalized_fit)
        _write_array(fit_group, "baseline", fit.baseline)
        if getattr(fit, "peak_curves", None):
            _write_array(fit_group, "peak_curves", np.vstack(fit.peak_curves))

    if hasattr(self, "calibration_x") and hasattr(self, "calibration_y"):
        cal_group = h5_file.create_group("calibration")
        _write_array(cal_group, "x", getattr(self, "calibration_x", None))
        _write_array(cal_group, "y", getattr(self, "calibration_y", None))
        _write_array(cal_group, "calibrated_energy_axis", getattr(self, "calibration_energy_axis", None))
        _write_array(cal_group, "calibrated_spectrum_y", getattr(self, "calibrated_spectrum_y", None))
        _write_array(cal_group, "calibrated_fit_energy", getattr(self, "calibrated_fit_energy", None))
        _write_array(cal_group, "calibrated_fit_y", getattr(self, "calibrated_fit_y", None))
        if getattr(self, "calibrated_components", None):
            _write_array(cal_group, "calibrated_components", np.vstack(self.calibrated_components))
        if hasattr(self, "calibration_fit"):
            fit = self.calibration_fit
            fit_group = cal_group.create_group("fit")
            _write_array(fit_group, "x", fit.x)
            _write_array(fit_group, "raw_y", fit.raw_y)
            _write_array(fit_group, "fit_y", getattr(fit, "fit_y", fit.raw_y))
            _write_array(fit_group, "tail_baseline", getattr(fit, "tail_baseline", None))
            _write_array(fit_group, "best_fit", fit.best_fit)
            _write_array(fit_group, "normalized_fit", fit.normalized_fit)
            _write_array(fit_group, "baseline", fit.baseline)
            if getattr(fit, "peak_curves", None):
                _write_array(fit_group, "peak_curves", np.vstack(fit.peak_curves))

    refs_group = h5_file.create_group("references")
    for index, (line_name, line_data) in enumerate(getattr(self, "plotted_lines", {}).items()):
        if not line_data.get("is_ref", False):
            continue
        ref_group = refs_group.create_group(f"ref_{index}")
        ref_group.attrs["name"] = line_name
        _write_array(ref_group, "x", line_data.get("x_data", []))
        _write_array(ref_group, "y", line_data.get("y_data", []))

    if hasattr(self, "last_peak_fit_iad"):
        iad = self.last_peak_fit_iad
        iad_group = h5_file.create_group("last_peak_fit_iad")
        _write_array(iad_group, "x", getattr(iad, "x", None))
        _write_array(iad_group, "roi_fit", getattr(iad, "roi_fit", None))
        _write_array(iad_group, "ref_fit", getattr(iad, "ref_fit", None))
        _write_array(iad_group, "roi_raw", getattr(iad, "roi_raw", None))
        _write_array(iad_group, "ref_raw", getattr(iad, "ref_raw", None))
        iad_group.attrs["iad_value"] = float(getattr(iad, "iad_value", np.nan))


def save_project(self):
    """Save the whole IXE analysis session as one .ixeproj file."""
    save_path = getattr(self, "current_project_path", "")
    if not save_path:
        run_number = _current_run_number(self).replace(" ", "_").replace(":", "")
        initialfile = f"Run_{run_number}.ixeproj" if run_number and run_number != "Not_Loaded" else "IXE_project.ixeproj"
        save_path = filedialog.asksaveasfilename(
            initialfile=initialfile,
            defaultextension=".ixeproj",
        )
        if not save_path:
            return
    try:
        manifest = _build_manifest(self)
        with h5py.File(save_path, "w") as h5_file:
            text_dtype = h5py.string_dtype(encoding="utf-8")
            h5_file.create_dataset("manifest", data=json.dumps(manifest, indent=2), dtype=text_dtype)
            h5_file.attrs["format"] = PROJECT_FORMAT
            h5_file.attrs["version"] = PROJECT_VERSION
            _write_project_arrays(self, h5_file)
        self.current_project_path = save_path
        print(f"Project saved to {save_path}")
        messagebox.showinfo("Export Project", f"Project saved:\n{save_path}")
    except Exception as exc:
        messagebox.showwarning("Export Project", str(exc))
        print(f"Error saving project: {exc}")


def _restore_peak_fit(h5_file, manifest):
    peak_meta = manifest.get("peak_fit")
    if not peak_meta or "peak_fit" not in h5_file:
        return None
    group = h5_file["peak_fit"]
    peak_curves = _read_array(group, "peak_curves", np.empty((0, 0), dtype=float))
    curves = [row.copy() for row in peak_curves] if peak_curves.size else []
    return PeakFitResult(
        x=_read_array(group, "x", np.array([], dtype=float)),
        raw_y=_read_array(group, "raw_y", np.array([], dtype=float)),
        fit_y=_read_array(group, "fit_y", _read_array(group, "raw_y", np.array([], dtype=float))),
        best_fit=_read_array(group, "best_fit", np.array([], dtype=float)),
        normalized_fit=_read_array(group, "normalized_fit", np.array([], dtype=float)),
        baseline=_read_array(group, "baseline", np.array([], dtype=float)),
        tail_baseline=_read_array(group, "tail_baseline", None),
        tail_baseline_enabled=bool(peak_meta.get("tail_baseline_enabled", False)),
        tail_fraction=float(peak_meta.get("tail_fraction", 0.10)),
        peak_curves=curves,
        centers=peak_meta.get("centers", []),
        sigmas=peak_meta.get("sigmas", []),
        amplitudes=peak_meta.get("amplitudes", []),
        fractions=peak_meta.get("fractions", []),
        peak_labels=peak_meta.get("peak_labels", []),
        widths=peak_meta.get("widths", []),
        peak_shape=peak_meta.get("peak_shape", "pseudo_voigt"),
        physical_fit=bool(peak_meta.get("physical_fit", False)),
    )


def _restore_calibration(self, h5_file, manifest):
    calibration_meta = manifest.get("calibration", {}) or {}
    cal_group = h5_file.get("calibration")
    if cal_group is not None:
        cal_x = _read_array(cal_group, "x")
        cal_y = _read_array(cal_group, "y")
        if cal_x is not None and cal_y is not None:
            self.calibration_x = cal_x
            self.calibration_y = cal_y
        calibrated_energy_axis = _read_array(cal_group, "calibrated_energy_axis")
        calibrated_spectrum_y = _read_array(cal_group, "calibrated_spectrum_y")
        calibrated_fit_energy = _read_array(cal_group, "calibrated_fit_energy")
        calibrated_fit_y = _read_array(cal_group, "calibrated_fit_y")
        calibrated_components = _read_array(cal_group, "calibrated_components")
        if calibrated_energy_axis is not None:
            self.calibration_energy_axis = calibrated_energy_axis
        if calibrated_spectrum_y is not None:
            self.calibrated_spectrum_y = calibrated_spectrum_y
        if calibrated_fit_energy is not None:
            self.calibrated_fit_energy = calibrated_fit_energy
        if calibrated_fit_y is not None:
            self.calibrated_fit_y = calibrated_fit_y
        if calibrated_components is not None and calibrated_components.size:
            self.calibrated_components = [row.copy() for row in calibrated_components]
    self.calibration_filepath = calibration_meta.get("filepath", "")
    self.calibration_source_column = calibration_meta.get("source_column", "")
    if hasattr(self, "calibration_file_label"):
        self.calibration_file_label.config(
            text=os.path.basename(self.calibration_filepath) if self.calibration_filepath else "No calibration file loaded"
        )
    _set_entry(getattr(self, "cal_fit_start", None), calibration_meta.get("fit_start", ""))
    _set_entry(getattr(self, "cal_fit_end", None), calibration_meta.get("fit_end", ""))

    fit_meta = calibration_meta.get("fit")
    if fit_meta and cal_group is not None and "fit" in cal_group:
        group = cal_group["fit"]
        peak_curves = _read_array(group, "peak_curves", np.empty((0, 0), dtype=float))
        curves = [row.copy() for row in peak_curves] if peak_curves.size else []
        self.calibration_fit = PeakFitResult(
            x=_read_array(group, "x", np.array([], dtype=float)),
            raw_y=_read_array(group, "raw_y", np.array([], dtype=float)),
            fit_y=_read_array(group, "fit_y", _read_array(group, "raw_y", np.array([], dtype=float))),
            best_fit=_read_array(group, "best_fit", np.array([], dtype=float)),
            normalized_fit=_read_array(group, "normalized_fit", np.array([], dtype=float)),
            baseline=_read_array(group, "baseline", np.array([], dtype=float)),
            tail_baseline=_read_array(group, "tail_baseline", None),
            tail_baseline_enabled=bool(fit_meta.get("tail_baseline_enabled", False)),
            tail_fraction=float(fit_meta.get("tail_fraction", 0.10)),
            peak_curves=curves,
            centers=fit_meta.get("centers", []),
            sigmas=fit_meta.get("sigmas", []),
            amplitudes=fit_meta.get("amplitudes", []),
            fractions=fit_meta.get("fractions", []),
            peak_labels=fit_meta.get("peak_labels", []),
            physical_fit=bool(fit_meta.get("physical_fit", False)),
        )
        centers = list(fit_meta.get("centers", []))
        if len(centers) >= 3:
            if hasattr(self, "cal_sat_energy_var"):
                self.cal_sat_energy_var.set(f"{float(centers[0]):.6g}")
            if hasattr(self, "cal_res_energy_var"):
                self.cal_res_energy_var.set(f"{float(centers[1]):.6g}")
            if hasattr(self, "cal_main_energy_var"):
                self.cal_main_energy_var.set(f"{float(centers[2]):.6g}")

    calibration = calibration_meta.get("energy_calibration")
    if calibration:
        self.energy_calibration = calibration
        assignments = (
            ("cal_sat_pixel_var", "pixel_satellite"),
            ("cal_res_pixel_var", "pixel_res"),
            ("cal_main_pixel_var", "pixel_main"),
            ("cal_sat_energy_var", "energy_satellite"),
            ("cal_res_energy_var", "energy_res"),
            ("cal_main_energy_var", "energy_main"),
            ("cal_slope_var", "slope"),
            ("cal_intercept_var", "intercept"),
            ("cal_residual_var", "residual_res_energy"),
        )
        for var_name, key in assignments:
            if hasattr(self, var_name) and key in calibration:
                getattr(self, var_name).set(f"{float(calibration[key]):.6g}")
    if hasattr(self, "cal_status_var"):
        if getattr(self, "energy_calibration", None):
            self.cal_status_var.set("Project calibration restored")
        elif hasattr(self, "calibration_x"):
            self.cal_status_var.set("Calibration spectrum restored")


def _restore_processor(self):
    if not hasattr(self, "immm"):
        self.spect_processor = None
        return
    gap_defaults = self.parm.get("ccd_gap", {})
    self.spect_processor = SpectrumProcessor(
        self.immm,
        int(self.parm.get("n_moveavg", 5)),
        invalid_pixel_mask=getattr(self, "processed_gap_mask", None),
        detector_background_mask=getattr(self, "processed_detector_background_mask", None),
        edge_valid_fraction_threshold=gap_defaults.get("edge_valid_fraction", 0.80),
    )
    self.spect_processor.toggle_gap_correction(getattr(self, "gap_correction_enabled", gap_defaults.get("enabled", True)))
    self.spect_processor.set_gap_correction_params(
        max_width=gap_defaults.get("max_width", 8),
        drop_fraction=gap_defaults.get("drop_fraction", 0.65),
    )
    trace_defaults = self.parm.get("trace_extraction", {})
    self.spect_processor.toggle_trace_extraction(getattr(self, "trace_extraction_enabled", trace_defaults.get("enabled", False)))
    self.spect_processor.set_trace_extraction_params(search_margin=trace_defaults.get("search_margin", 60))
    if getattr(self, "bg_subtraction_enabled", False):
        bg_ranges = self.get_bg_row_ranges() if hasattr(self, "get_bg_row_ranges") else [self.get_bg_row_range()]
        col_begin, col_end = self.get_roi_col_range()
        rois = [(begin, end + 1, col_begin, col_end + 1) for begin, end in bg_ranges if end > begin]
        if hasattr(self.spect_processor, "set_background_rois"):
            self.spect_processor.set_background_rois(rois)
        elif rois:
            self.spect_processor.set_background_roi(*rois[0])
        self.spect_processor.toggle_background_subtraction(True)


def _set_slider_values(self):
    row_count = self.immm.shape[0] if hasattr(self, "immm") else (self.imm.shape[0] if hasattr(self, "imm") else 1)
    col_count = self.immm.shape[1] if hasattr(self, "immm") else (self.imm.shape[1] if hasattr(self, "imm") else 1)
    row_begin = int(self.parm.get("row_begin", 0))
    row_end = int(self.parm.get("row_end", min(row_count - 1, row_begin + 1)))
    row2_begin = int(self.parm.get("row2_begin", row_begin))
    row2_end = int(self.parm.get("row2_end", row_end))
    col_begin = int(self.parm.get("column_begin", 0))
    col_end = int(self.parm.get("column_end", min(col_count - 1, col_begin + 1)))
    bg_begin = int(self.parm.get("bg_row_begin", 0))
    bg_end = int(self.parm.get("bg_row_end", bg_begin))
    bg2_begin = int(self.parm.get("bg2_row_begin", bg_begin))
    bg2_end = int(self.parm.get("bg2_row_end", bg_end))
    self.roi_second_enabled = bool(self.parm.get("roi_second_enabled", False))

    self.row_slider.set_limits(0, max(row_count - 1, 0), row_begin, row_end, row2_begin, row2_end, invoke=False)
    if hasattr(self, "_sync_roi_row_buttons"):
        self._sync_roi_row_buttons()
    self.column_slider.set_limits(0, max(col_count - 1, 0), col_begin, col_end, invoke=False)
    self.bg_row_slider.set_limits(0, max(row_count - 1, 0), bg_begin, bg_end, bg2_begin, bg2_end, invoke=False)
    self.on_roi_rows_changed(row_begin, row_end, row2_begin, row2_end, redraw=False)
    self.on_roi_cols_changed(col_begin, col_end, redraw=False)
    self.on_bg_rows_changed(bg_begin, bg_end, bg2_begin, bg2_end, redraw=False)


def _restore_references(self, h5_file, manifest):
    for widget in self.line_list_frame.winfo_children():
        widget.destroy()
    self.plotted_lines = {}
    refs_group = h5_file.get("references")
    references = manifest.get("references", [])
    for index, ref_meta in enumerate(references):
        ref_group = refs_group.get(f"ref_{index}") if refs_group is not None else None
        if ref_group is None:
            continue
        line_name = ref_meta["name"]
        line_color = ref_meta.get("color", "#000000")
        self.plotted_lines[line_name] = {
            "x_data": _read_array(ref_group, "x", np.array([], dtype=float)).tolist(),
            "y_data": _read_array(ref_group, "y", np.array([], dtype=float)).tolist(),
            "color": line_color,
            "run_number": ref_meta.get("run_number", ""),
            "visible": bool(ref_meta.get("visible", True)),
            "checkbox": None,
            "color_circle": None,
            "is_ref": bool(ref_meta.get("is_ref", True)),
            "is_peak_fit": bool(ref_meta.get("is_peak_fit", False)),
            "source_column": ref_meta.get("source_column", "Y"),
            "x_axis_repaired": bool(ref_meta.get("x_axis_repaired", False)),
            "gap_corrected": bool(ref_meta.get("gap_corrected", False)),
            "tail_matched": bool(ref_meta.get("tail_matched", False)),
            "satellite_tail_matched": bool(ref_meta.get("satellite_tail_matched", False)),
        }
        self.add_line_to_list(line_name, line_color)
        self.plotted_lines[line_name]["visible"] = bool(ref_meta.get("visible", True))
        checkbox_var = self.plotted_lines[line_name].get("checkbox_var")
        if checkbox_var is not None:
            checkbox_var.set(bool(ref_meta.get("visible", True)))
        _set_entry(self.plotted_lines[line_name].get("iad_entry"), ref_meta.get("iad_value", ""))
        _set_entry(self.plotted_lines[line_name].get("st_iad_entry"), ref_meta.get("satellite_iad_value", ""))


def _redraw_loaded_peak_fit(self):
    if not hasattr(self, "last_peak_fit_profile"):
        if getattr(self, "plotted_lines", {}):
            iad_controls._redraw_reference_display(self)
        return
    fit = self.last_peak_fit_profile
    if hasattr(spectrum_controls, "_update_peak_fit_results_box"):
        spectrum_controls._update_peak_fit_results_box(self, fit)
    self.ax_spectrum.clear()
    x_data = np.asarray(fit.x, dtype=float)
    background_y = getattr(self, "last_spectrum_roi_uncorrected", fit.raw_y)
    if len(background_y) != len(x_data):
        background_y = fit.raw_y
    self.ax_spectrum.plot(x_data, background_y, color="0.75", linewidth=1.0, alpha=0.8, label="Original")
    if getattr(self, "current_spectrum_gap_corrected", False) and hasattr(self, "last_spectrum_roi"):
        y_data = np.asarray(self.last_spectrum_roi, dtype=float)
        if len(y_data) == len(x_data):
            self.ax_spectrum.plot(x_data, y_data, color="0.45", linewidth=0.8, alpha=0.9, label="Gap-corrected")
    self.ax_spectrum.plot(x_data, fit.normalized_fit, color=self.line_color.get(), linewidth=1.6, label="Peak fit")
    scale = spectrum_controls._component_scale(fit)
    if scale is not None:
        colors = spectrum_controls.plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        for index, peak_curve in enumerate(fit.peak_curves, start=1):
            color = colors[index % len(colors)] if colors else None
            label = fit.peak_labels[index - 1] if index - 1 < len(fit.peak_labels) else f"Peak {index}"
            self.ax_spectrum.plot(
                x_data,
                np.asarray(peak_curve, dtype=float) / scale,
                linestyle="--",
                linewidth=0.9,
                color=color,
                alpha=0.9,
                label=label,
            )
    if getattr(self, "plotted_lines", {}):
        for line_name, line_data in self.plotted_lines.items():
            if line_data.get("visible"):
                self.ax_spectrum.plot(
                    line_data["x_data"],
                    line_data["y_data"],
                    label=line_name,
                    color=line_data["color"],
                    linestyle="-",
                    linewidth=0.5,
                )
    spectrum_controls._finish_spectrum_axes(
        self,
        xlabel="Column Index",
        ylabel="Normalized intensity",
        view_mode="fit",
        view_label="Peak fit",
    )


def open_project(self):
    """Open an IXE .ixeproj session file."""
    project_path = filedialog.askopenfilename(
        filetypes=[("IXE project", "*.ixeproj"), ("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")],
    )
    if not project_path:
        return
    try:
        with h5py.File(project_path, "r") as h5_file:
            manifest = _read_manifest(h5_file)
            if manifest.get("format") != PROJECT_FORMAT:
                raise ValueError("This file is not an IXE project.")

            self.reset_spectrum_state()
            for attr_name in ("imm", "immm", "raw_gap_mask", "processed_gap_mask", "processed_detector_background_mask"):
                if hasattr(self, attr_name):
                    delattr(self, attr_name)
            self.original_filepath = manifest.get("original_filepath", "")
            stack_files = list(manifest.get("image_stack_files", []))
            stack_count = int(manifest.get("image_stack_count", len(stack_files) if stack_files else 1) or 1)
            self.image_stack_files = stack_files or ([self.original_filepath] if self.original_filepath else [])
            self.image_stack_count = stack_count
            self.parm.update(manifest.get("parameters", {}))
            self.parm.setdefault("trace_extraction", {})["enabled"] = False
            ui = manifest.get("ui", {})

            images = h5_file.get("images")
            raw = _read_array(images, "raw")
            processed = _read_array(images, "processed")
            if raw is not None:
                self.imm = raw.astype(np.float32, copy=False)
            if processed is not None:
                self.immm = processed.astype(np.float32, copy=False)

            masks = h5_file.get("masks")
            raw_mask = _read_array(masks, "raw_gap")
            processed_mask = _read_array(masks, "processed_gap")
            detector_background_mask = _read_array(masks, "detector_background")
            if raw_mask is not None:
                self.raw_gap_mask = raw_mask.astype(bool, copy=False)
            if processed_mask is not None:
                self.processed_gap_mask = processed_mask.astype(bool, copy=False)
            if detector_background_mask is not None:
                self.processed_detector_background_mask = detector_background_mask.astype(bool, copy=False)

            self.bg_subtraction_enabled = bool(ui.get("bg_subtraction_enabled", False))
            self.gap_correction_enabled = bool(ui.get("gap_correction_enabled", True))
            self.trace_extraction_enabled = False
            self.current_spectrum_moving_avg = int(ui.get("current_spectrum_moving_avg", 0))
            self.current_spectrum_bg_removed = bool(ui.get("current_spectrum_bg_removed", False))
            self.current_spectrum_gap_corrected = bool(ui.get("current_spectrum_gap_corrected", False))
            self.current_spectrum_gap_columns = list(ui.get("current_spectrum_gap_columns", []))
            self.current_spectrum_trace_extracted = False

            if self.image_stack_count > 1 and self.image_stack_files:
                basenames = [os.path.basename(path) for path in self.image_stack_files]
                preview = "; ".join(basenames[:3])
                if len(basenames) > 3:
                    preview += f"; ... +{len(basenames) - 3} more"
                _set_entry(self.file_path_entry, f"{self.image_stack_count} TIFF stack: {preview}")
            else:
                _set_entry(self.file_path_entry, self.original_filepath)
            self.parm["threshold"] = ui.get("threshold", self.parm.get("threshold", 0))
            _set_entry(self.vmin_entry, ui.get("vmin", self.parm.get("vmin", 0)))
            _set_entry(self.vmax_entry, ui.get("vmax", self.parm.get("vmax", 100)))
            if hasattr(self, "image_cmap"):
                self.image_cmap.set(ui.get("cmap", self.parm.get("cmap", "viridis")))
                self.parm["cmap"] = self.image_cmap.get()
            if hasattr(self, "tilt_value"):
                self.tilt_value.set(str(ui.get("tilt", self.parm.get("tilt_cor", 0))))
            peak_fit_defaults = self.parm.get("peak_fit", {})
            if hasattr(self, "peak_fit_model_var"):
                self.peak_fit_model_var.set(peak_fit_defaults.get("peak_shape", "pseudo_voigt"))
            _set_entry(getattr(self, "peak_fit_count", None), peak_fit_defaults.get("n_peaks", 3))
            _set_entry(getattr(self, "peak_fit_start", None), peak_fit_defaults.get("x_min", ""))
            _set_entry(getattr(self, "peak_fit_end", None), peak_fit_defaults.get("x_max", ""))
            self.line_color.set(ui.get("line_color", "black"))
            self.line_style.set(ui.get("line_style", "-"))
            _set_entry(self.line_width, ui.get("line_width", "0.5"))
            _set_entry(self.cross_begin, ui.get("cross_begin", self.parm.get("PK intersect", {}).get("i_l", 50)))
            _set_entry(self.cross_end, ui.get("cross_end", self.parm.get("PK intersect", {}).get("i_r", 120)))
            _set_entry(self.eye_ball_cross, ui.get("eye_ball_cross", ""))
            self.calibration_filepath = ui.get("calibration_filepath", "")
            if hasattr(self, "calibration_file_label"):
                self.calibration_file_label.config(
                    text=os.path.basename(self.calibration_filepath) if self.calibration_filepath else "No calibration file loaded"
                )

            self.run_number_label.config(text=manifest.get("run_label", "Run: Not Loaded"))
            self.run_number_label_spectrum.config(text=manifest.get("run_label_spectrum", manifest.get("run_label", "Run: Not Loaded")))
            if hasattr(self, "bg_subtraction_var"):
                self.bg_subtraction_var.set(self.bg_subtraction_enabled)
            if hasattr(self, "gap_correction_var"):
                self.gap_correction_var.set(self.gap_correction_enabled)
            self.bg_toggle.config(text="BG Remove", style="TButton")
            self.gap_toggle.config(text="Gap Mask", style="TButton")
            if hasattr(self, "trace_toggle"):
                self.trace_toggle.config(text="Trace ROI (ON)" if self.trace_extraction_enabled else "Trace ROI (OFF)")

            _set_slider_values(self)
            _restore_processor(self)

            spectra = h5_file.get("spectra")
            last_roi = _read_array(spectra, "last_spectrum_roi")
            last_uncorrected = _read_array(spectra, "last_spectrum_roi_uncorrected")
            if last_roi is not None:
                self.last_spectrum_roi = last_roi
            if last_uncorrected is not None:
                self.last_spectrum_roi_uncorrected = last_uncorrected

            peak_fit = _restore_peak_fit(h5_file, manifest)
            if peak_fit is not None:
                self.last_peak_fit_profile = peak_fit

            _restore_calibration(self, h5_file, manifest)
            _restore_references(self, h5_file, manifest)

        display_data = getattr(self, "immm", getattr(self, "imm", None))
        if display_data is not None:
            title = "Processed Image" if hasattr(self, "immm") else "Raw Image"
            if not hasattr(self, "immm") and getattr(self, "image_stack_count", 1) > 1:
                title = "Stacked Raw Image"
            self.display_image(display_data, self.ax_image, title)
        _redraw_loaded_peak_fit(self)
        self.current_project_path = project_path
        print(f"Project opened from {project_path}")
        messagebox.showinfo("Open Project", f"Project opened:\n{project_path}")
    except Exception as exc:
        messagebox.showwarning("Open Project", str(exc))
        print(f"Error opening project: {exc}")
