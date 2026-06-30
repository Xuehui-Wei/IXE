# -*- coding: utf-8 -*-
"""Qt/pyqtgraph analyzer with embedded CCD and spectrum displays."""

import re
import csv
import json
import os
import sys
from datetime import datetime

import numpy as np
from PIL import Image
import scipy.ndimage

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph.exporters import ImageExporter, SVGExporter

try:
    import h5py
except ImportError:  # pragma: no cover - project save/open reports this at runtime
    h5py = None

try:
    from .peak_fitting import PeakFitConfig, PeakFitError, PeakFitResult, fit_spectrum
    from .spectrum_utils import SpectrumProcessor
    from .image_processing import (
        detect_detector_gap_mask,
        estimate_image_tilt_angle,
        rotate_detector_background_mask,
        rotate_detector_gap_mask,
    )
except ImportError:
    from peak_fitting import PeakFitConfig, PeakFitError, PeakFitResult, fit_spectrum
    from spectrum_utils import SpectrumProcessor
    from image_processing import (
        detect_detector_gap_mask,
        estimate_image_tilt_angle,
        rotate_detector_background_mask,
        rotate_detector_gap_mask,
    )


pg.setConfigOptions(antialias=True, imageAxisOrder='row-major')
_QT_SIGNAL = getattr(QtCore, 'Signal', getattr(QtCore, 'pyqtSignal'))
PROJECT_FORMAT = "IXE Project"
PROJECT_VERSION = 1
IAD_MC_DEFAULT_TRIALS = 100
IAD_MC_MIN_SUCCESS = 40
IAD_MC_SEED = 20260612


def _left_mouse_button():
    return getattr(getattr(QtCore.Qt, 'MouseButton', QtCore.Qt), 'LeftButton')


def _pointing_cursor():
    return getattr(getattr(QtCore.Qt, 'CursorShape', QtCore.Qt), 'PointingHandCursor')


def _wait_cursor():
    return getattr(getattr(QtCore.Qt, 'CursorShape', QtCore.Qt), 'WaitCursor')


def _painter_antialiasing():
    return getattr(getattr(QtGui.QPainter, 'RenderHint', QtGui.QPainter), 'Antialiasing')


def _read_tiff_array(filepath):
    with Image.open(filepath) as image:
        return np.asarray(image, dtype=np.float32)


def _read_two_column_spectrum(filepath):
    x_data = []
    y_data = []
    with open(filepath, mode="r", newline="") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "\t" in stripped:
                parts = [value.strip() for value in stripped.split("\t")]
            elif "," in stripped:
                parts = [value.strip() for value in next(csv.reader([stripped]))]
            else:
                parts = stripped.split()
            if len(parts) < 2:
                continue
            try:
                x_value = float(parts[0])
                y_value = float(parts[1])
            except (TypeError, ValueError):
                continue
            x_data.append(x_value)
            y_data.append(y_value)
    x = np.asarray(x_data, dtype=float)
    y = np.asarray(y_data, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2:
        raise ValueError("No usable two-column spectrum was found.")
    order = np.argsort(x)
    return x[order], y[order]


def _split_spectrum_row(line):
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return []
    if "\t" in stripped:
        return [value.strip() for value in stripped.split("\t")]
    if "," in stripped:
        try:
            return [value.strip() for value in next(csv.reader([stripped]))]
        except csv.Error:
            return [value.strip() for value in stripped.split(",")]
    return stripped.split()


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _header_name(value):
    return re.sub(r"\s+", " ", str(value).strip().lower())


_SUBSCRIPT_TRANSLATION = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def _peak_label_key(value):
    text = str(value or "").strip().lower().translate(_SUBSCRIPT_TRANSLATION)
    text = text.replace("\\beta", "beta").replace("β", "beta")
    text = text.replace("\\mathrm", "").replace("mathrm", "")
    text = text.replace("prime", "'").replace("′", "'")
    return re.sub(r"[\s\$\{\}_\\,\-]+", "", text)


def _is_main_kbeta_label(value):
    key = _peak_label_key(value)
    return "k" in key and "beta" in key and ("1,3" in key or "13" in key)


def _read_pkfit_rows(filepath):
    with open(filepath, mode="r", newline="") as handle:
        return [_split_spectrum_row(line) for line in handle]


def _extract_pkfit_xy(rows):
    x_data = []
    y_data = []
    header_index = None
    header = []
    for row_index, row in enumerate(rows):
        normalized = [_header_name(value) for value in row]
        if "x" not in normalized or "peak fit" not in normalized:
            continue
        x_column = normalized.index("x")
        y_column = normalized.index("peak fit")
        header_index = row_index
        header = row
        for data_row in rows[row_index + 1:]:
            if len(data_row) <= max(x_column, y_column):
                continue
            x_value = _as_float(data_row[x_column])
            y_value = _as_float(data_row[y_column])
            if x_value is not None and y_value is not None:
                x_data.append(x_value)
                y_data.append(y_value)
        break

    x = np.asarray(x_data, dtype=float)
    y = np.asarray(y_data, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2:
        raise ValueError(
            "Import Ref. expects a pkfit file exported by Export PKfit with numeric X and Peak fit columns."
        )
    order = np.argsort(x)
    return x[order], y[order], header_index, header


def _extract_pkfit_fit_error(rows):
    return _extract_pkfit_scalar(rows, ("fit error", "relative fit error"))


def _extract_pkfit_scalar(rows, keys):
    normalized_keys = {_header_name(key) for key in keys}
    for row in rows:
        if len(row) < 2:
            continue
        key = _header_name(row[0])
        if key not in normalized_keys:
            continue
        value = _as_float(row[1])
        if value is not None and np.isfinite(value):
            return float(value)
    return np.nan


def _extract_pkfit_input_reference_fit_errors(rows):
    in_input_block = False
    fit_error_column = None
    fit_errors = []
    for row in rows:
        normalized = [_header_name(value) for value in row]
        row_label = _header_name(" ".join(str(value) for value in row))
        if not normalized:
            continue
        if row_label == "input references":
            in_input_block = True
            fit_error_column = None
            continue
        if not in_input_block:
            continue
        if row_label == "peak parameters" or _is_spectrum_table_header(row):
            break
        if fit_error_column is None:
            if "fit error" in normalized:
                fit_error_column = normalized.index("fit error")
            elif "relative fit error" in normalized:
                fit_error_column = normalized.index("relative fit error")
            continue
        if len(row) <= fit_error_column:
            continue
        value = _as_float(row[fit_error_column])
        if value is not None and np.isfinite(value):
            fit_errors.append(float(value))
    return np.asarray(fit_errors, dtype=float)


def _extract_pkfit_table_columns(rows, column_candidates):
    for row_index, row in enumerate(rows):
        if not row or not _is_spectrum_table_header(row):
            continue
        normalized = [_header_name(value) for value in row]
        indices = {}
        for key, candidates in column_candidates.items():
            for candidate in candidates:
                candidate = _header_name(candidate)
                if candidate in normalized:
                    indices[key] = normalized.index(candidate)
                    break
        if "x" not in indices:
            continue
        values = {key: [] for key in indices}
        for data_row in rows[row_index + 1:]:
            if len(data_row) <= max(indices.values()):
                continue
            x_value = _as_float(data_row[indices["x"]])
            if x_value is None:
                continue
            for key, column in indices.items():
                value = _as_float(data_row[column])
                values[key].append(np.nan if value is None else float(value))
        if len(values.get("x", [])) >= 2:
            return {key: np.asarray(column_values, dtype=float) for key, column_values in values.items()}
    return {}


def _extract_pkfit_average_scatter_error(rows):
    columns = _extract_pkfit_table_columns(
        rows,
        {
            "x": ("x", "energy"),
            "mean": ("peak fit", "y", "raw y", "intensity"),
            "std": ("std",),
            "valid_n": ("valid n",),
        },
    )
    if not all(key in columns for key in ("x", "mean", "std", "valid_n")):
        return np.nan
    return _reference_mean_scatter_error(columns["x"], columns["mean"], columns["std"], columns["valid_n"])


def _rows_mark_reference_average(rows):
    for row in rows[:20]:
        if len(row) < 2:
            continue
        key = _header_name(row[0])
        value = _header_name(row[1])
        if key == "reference average" and value in ("true", "yes", "1"):
            return True
    return False


def _find_pkfit_main_position(rows, x, y, header_index, header):
    for row_index, row in enumerate(rows):
        normalized = [_header_name(value) for value in row]
        if "peak" not in normalized or "position" not in normalized:
            continue
        peak_column = normalized.index("peak")
        position_column = normalized.index("position")
        numeric_positions = []
        for data_row in rows[row_index + 1:]:
            if _is_spectrum_table_header(data_row):
                break
            if len(data_row) <= max(peak_column, position_column):
                continue
            position = _as_float(data_row[position_column])
            if position is None:
                continue
            numeric_positions.append(position)
            if _is_main_kbeta_label(data_row[peak_column]):
                return float(position)
        if len(numeric_positions) >= 3:
            return float(numeric_positions[-1])

    for row in rows:
        for cell in row:
            if "=" not in str(cell):
                continue
            label, value = str(cell).split("=", 1)
            if _is_main_kbeta_label(label):
                position = _as_float(value)
                if position is not None:
                    return float(position)

    if header_index is not None and header:
        normalized = [_header_name(value) for value in header]
        main_columns = [index for index, value in enumerate(normalized) if _is_main_kbeta_label(value)]
        if main_columns:
            x_column = normalized.index("x")
            component_column = main_columns[0]
            component_x = []
            component_y = []
            for data_row in rows[header_index + 1:]:
                if len(data_row) <= max(x_column, component_column):
                    continue
                x_value = _as_float(data_row[x_column])
                y_value = _as_float(data_row[component_column])
                if x_value is not None and y_value is not None and np.isfinite(y_value):
                    component_x.append(x_value)
                    component_y.append(y_value)
            if component_y:
                component_x = np.asarray(component_x, dtype=float)
                component_y = np.asarray(component_y, dtype=float)
                return float(component_x[int(np.nanargmax(component_y))])

    finite = np.isfinite(x) & np.isfinite(y)
    if np.any(finite):
        x_finite = np.asarray(x, dtype=float)[finite]
        y_finite = np.asarray(y, dtype=float)[finite]
        return float(x_finite[int(np.nanargmax(y_finite))])
    raise ValueError("Could not determine the Kβ1,3 main-peak position from the pkfit file.")


def _read_pkfit_peak_fit_spectrum(filepath):
    rows = _read_pkfit_rows(filepath)
    x, y, _header_index, _header = _extract_pkfit_xy(rows)
    return x, y


def _read_pkfit_iad_reference(filepath):
    rows = _read_pkfit_rows(filepath)
    x, y, _header_index, _header = _extract_pkfit_xy(rows)
    is_average = _rows_mark_reference_average(rows)
    fit_error = _extract_pkfit_fit_error(rows)
    input_fit_errors = _extract_pkfit_input_reference_fit_errors(rows)
    scatter_columns = _extract_pkfit_table_columns(
        rows,
        {
            "x": ("x", "energy"),
            "mean": ("peak fit", "y", "raw y", "intensity"),
            "std": ("std",),
            "valid_n": ("valid n",),
        },
    )

    average_fit_error = _extract_pkfit_scalar(
        rows,
        ("average reference fit error", "avg reference fit error"),
    )
    if not np.isfinite(average_fit_error):
        average_fit_error = _average_reference_fit_error(input_fit_errors)
    if not is_average and not np.isfinite(average_fit_error):
        average_fit_error = fit_error

    scatter_error = _extract_pkfit_scalar(
        rows,
        ("reference mean scatter error", "reference scatter error"),
    )
    if not np.isfinite(scatter_error):
        scatter_error = _extract_pkfit_average_scatter_error(rows)

    total_error = _extract_pkfit_scalar(
        rows,
        ("total average reference error", "total reference error"),
    )
    if not np.isfinite(total_error):
        if is_average:
            total_error = _quadrature(average_fit_error, scatter_error)
        else:
            total_error = fit_error

    metadata = {
        "is_average": is_average,
        "fit_error": float(fit_error),
        "input_fit_errors": input_fit_errors,
        "average_reference_fit_error": float(average_fit_error),
        "reference_mean_scatter_error": float(scatter_error),
        "total_reference_error": float(total_error),
        "reference_scatter_x": scatter_columns.get("x", np.array([], dtype=float)),
        "reference_scatter_mean": scatter_columns.get("mean", np.array([], dtype=float)),
        "reference_scatter_std": scatter_columns.get("std", np.array([], dtype=float)),
        "reference_scatter_valid_n": scatter_columns.get("valid_n", np.array([], dtype=float)),
    }
    return x, y, metadata


def _read_pkfit_reference_profile(filepath):
    rows = _read_pkfit_rows(filepath)
    x, y, header_index, header = _extract_pkfit_xy(rows)
    main_position = _find_pkfit_main_position(rows, x, y, header_index, header)
    fit_error = _extract_pkfit_fit_error(rows)
    return x, y, main_position, fit_error


def _is_pkfit_reference_average(filepath):
    try:
        rows = _read_pkfit_rows(filepath)
    except Exception:
        return False
    return _rows_mark_reference_average(rows)


def _preferred_spectrum_y_column(headers):
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


def _read_calibration_spectrum(filepath):
    with open(filepath, mode="r", newline="") as handle:
        rows = [_split_spectrum_row(line) for line in handle]

    x_data = []
    y_data = []
    source_column = "Y"
    for row_index, row in enumerate(rows):
        if not row or not _is_spectrum_table_header(row):
            continue
        x_column = 0
        y_column = _preferred_spectrum_y_column(row)
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
        x_data, y_data = _read_two_column_spectrum(filepath)
        source_column = "Y"
    else:
        x_data = np.asarray(x_data, dtype=float)
        y_data = np.asarray(y_data, dtype=float)

    finite = np.isfinite(x_data) & np.isfinite(y_data)
    x = np.asarray(x_data, dtype=float)[finite]
    y = np.asarray(y_data, dtype=float)[finite]
    if x.size < 8:
        raise ValueError("Calibration spectrum has too few finite points.")
    order = np.argsort(x)
    return x[order], y[order], source_column


def _energy_axis(calibration, x_values):
    return float(calibration["slope"]) * np.asarray(x_values, dtype=float) + float(calibration["intercept"])


def _normalize_overlay(y_values):
    y_values = np.asarray(y_values, dtype=float)
    if y_values.size == 0:
        return y_values
    finite = np.isfinite(y_values)
    if not np.any(finite):
        return np.zeros_like(y_values)
    fill = float(np.nanmedian(y_values[finite]))
    y = np.nan_to_num(y_values, nan=fill, posinf=fill, neginf=fill)
    y = y - float(np.nanmin(y))
    max_value = float(np.nanmax(y))
    if max_value <= np.finfo(float).eps:
        return np.zeros_like(y)
    return y / max_value


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


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


def _fit_metadata(fit_result):
    return {
        "peak_labels": list(getattr(fit_result, "peak_labels", [])),
        "centers": _json_safe(getattr(fit_result, "centers", [])),
        "sigmas": _json_safe(getattr(fit_result, "sigmas", [])),
        "amplitudes": _json_safe(getattr(fit_result, "amplitudes", [])),
        "fractions": _json_safe(getattr(fit_result, "fractions", [])),
        "widths": _json_safe(getattr(fit_result, "widths", [])),
        "peak_shape": getattr(fit_result, "peak_shape", "pseudo_voigt"),
        "physical_fit": bool(getattr(fit_result, "physical_fit", False)),
        "tail_baseline_enabled": bool(getattr(fit_result, "tail_baseline_enabled", False)),
        "tail_fraction": float(getattr(fit_result, "tail_fraction", 0.10)),
        "relative_fit_error": float(getattr(fit_result, "relative_fit_error", np.nan)),
        "fit_residual_area": float(getattr(fit_result, "fit_residual_area", np.nan)),
        "fit_total_intensity": float(getattr(fit_result, "fit_total_intensity", np.nan)),
    }


def _write_fit_group(parent, name, fit_result):
    group = parent.create_group(name)
    _write_array(group, "x", getattr(fit_result, "x", None))
    _write_array(group, "raw_y", getattr(fit_result, "raw_y", None))
    _write_array(group, "fit_y", getattr(fit_result, "fit_y", None))
    _write_array(group, "tail_baseline", getattr(fit_result, "tail_baseline", None))
    _write_array(group, "fit_residual", getattr(fit_result, "fit_residual", None))
    _write_array(group, "normalized_fit_residual", getattr(fit_result, "normalized_fit_residual", None))
    _write_array(group, "best_fit", getattr(fit_result, "best_fit", None))
    _write_array(group, "normalized_fit", getattr(fit_result, "normalized_fit", None))
    _write_array(group, "baseline", getattr(fit_result, "baseline", None))
    if getattr(fit_result, "peak_curves", None):
        _write_array(group, "peak_curves", np.vstack(fit_result.peak_curves))
    return group


def _read_fit_group(group, metadata):
    if group is None:
        return None
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
        tail_baseline_enabled=bool(metadata.get("tail_baseline_enabled", False)),
        tail_fraction=float(metadata.get("tail_fraction", 0.10)),
        peak_curves=curves,
        centers=metadata.get("centers", []),
        sigmas=metadata.get("sigmas", []),
        amplitudes=metadata.get("amplitudes", []),
        fractions=metadata.get("fractions", []),
        peak_labels=metadata.get("peak_labels", []),
        widths=metadata.get("widths", []),
        peak_shape=metadata.get("peak_shape", "pseudo_voigt"),
        physical_fit=bool(metadata.get("physical_fit", False)),
    )


def _run_label_from_path(path):
    basename = os.path.basename(path)
    match = re.search(r"Run[_\s-]*(\d+)", basename, re.IGNORECASE)
    return f"Run {match.group(1)}" if match else basename


def _run_label_from_paths(paths):
    run_numbers = []
    for path in paths:
        match = re.search(r"Run_(\d+)", path)
        if match:
            run_numbers.append(match.group(1))
    if not run_numbers:
        return f"Stack ({len(paths)} images)" if len(paths) > 1 else "Run: Not Loaded"
    if len(paths) == 1:
        return f"Run {run_numbers[0]}"
    if len(set(run_numbers)) == 1:
        return f"Run {run_numbers[0]} stack ({len(paths)} images)"
    return f"Run stack {run_numbers[0]}-{run_numbers[-1]} ({len(paths)} images)"


def _normalize_spectrum(spectrum):
    spectrum = np.asarray(spectrum, dtype=float)
    if spectrum.size == 0:
        return spectrum.copy()
    finite = np.isfinite(spectrum)
    if not np.any(finite):
        return np.zeros_like(spectrum)
    baseline = float(np.nanmin(spectrum[finite]))
    normalized = spectrum - baseline
    normalized[~np.isfinite(normalized)] = 0.0
    total = float(np.nansum(normalized))
    if total <= np.finfo(float).eps:
        return np.zeros_like(normalized)
    return normalized / total


def _trapezoid_area(x_values, y_values):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if np.count_nonzero(finite) < 2:
        return float(np.nansum(y_values[finite]))
    x = x_values[finite]
    y = y_values[finite]
    order = np.argsort(x)
    integrator = getattr(np, "trapezoid", np.trapz)
    return float(integrator(y[order], x[order]))


def _area_normalize_spectrum(x_values, y_values, valid_mask=None, return_transform=False):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    normalized = np.full_like(y_values, np.nan, dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    if np.count_nonzero(valid) < 2:
        if return_transform:
            return normalized, np.nan, np.nan
        return normalized

    baseline = float(np.nanmin(y_values[valid]))
    shifted = np.full_like(y_values, np.nan, dtype=float)
    shifted[valid] = np.maximum(y_values[valid] - baseline, 0.0)
    area = _trapezoid_area(x_values[valid], shifted[valid])
    if not np.isfinite(area) or area <= np.finfo(float).eps:
        area = float(np.nansum(shifted[valid]))
    if not np.isfinite(area) or area <= np.finfo(float).eps:
        if return_transform:
            return normalized, baseline, area
        return normalized
    normalized[valid] = shifted[valid] / area
    if return_transform:
        return normalized, baseline, area
    return normalized


def _area_normalize_pair(x_values, first_y, second_y):
    x_values = np.asarray(x_values, dtype=float)
    first_y = np.asarray(first_y, dtype=float)
    second_y = np.asarray(second_y, dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(first_y) & np.isfinite(second_y)
    first_norm = _area_normalize_spectrum(x_values, first_y, valid)
    second_norm = _area_normalize_spectrum(x_values, second_y, valid)
    valid = np.isfinite(x_values) & np.isfinite(first_norm) & np.isfinite(second_norm)
    return x_values[valid], first_norm[valid], second_norm[valid]


def _integrated_absolute_difference(x_values, first_y, second_y):
    x_values = np.asarray(x_values, dtype=float)
    diff = np.abs(np.asarray(first_y, dtype=float) - np.asarray(second_y, dtype=float))
    finite = np.isfinite(x_values) & np.isfinite(diff)
    if np.count_nonzero(finite) < 2:
        return float(np.nansum(diff[finite]))
    return _trapezoid_area(x_values[finite], diff[finite])


def _quadrature(*values):
    finite_values = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            finite_values.append(value)
    if not finite_values:
        return np.nan
    finite_values = np.asarray(finite_values, dtype=float)
    return float(np.sqrt(np.sum(finite_values * finite_values)))


def _average_reference_fit_error(fit_errors):
    fit_errors = np.asarray(fit_errors, dtype=float)
    fit_errors = fit_errors[np.isfinite(fit_errors)]
    if fit_errors.size == 0:
        return np.nan
    return float(np.sqrt(np.sum(fit_errors * fit_errors)) / fit_errors.size)


def _reference_mean_scatter_error(x_values, mean_y, std_y, valid_counts):
    x_values = np.asarray(x_values, dtype=float)
    mean_y = np.asarray(mean_y, dtype=float)
    std_y = np.asarray(std_y, dtype=float)
    valid_counts = np.asarray(valid_counts, dtype=float)
    sem = np.full_like(std_y, np.nan, dtype=float)
    valid_n = np.isfinite(valid_counts) & (valid_counts > 0)
    sem[valid_n] = std_y[valid_n] / np.sqrt(valid_counts[valid_n])
    valid = np.isfinite(x_values) & np.isfinite(mean_y) & np.isfinite(sem)
    if np.count_nonzero(valid) < 2:
        return np.nan
    denominator = _trapezoid_area(x_values[valid], np.abs(mean_y[valid]))
    numerator = _trapezoid_area(x_values[valid], np.abs(sem[valid]))
    if not np.isfinite(denominator) or denominator <= np.finfo(float).eps:
        return np.nan
    return float(numerator / denominator)


def _reference_fractional_scatter_error(reference, fraction):
    try:
        fraction = float(fraction)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(fraction) or fraction <= 0:
        return np.nan
    fraction = min(fraction, 1.0)
    x_values = np.asarray(reference.get("reference_scatter_x", []), dtype=float)
    mean_y = np.asarray(reference.get("reference_scatter_mean", []), dtype=float)
    std_y = np.asarray(reference.get("reference_scatter_std", []), dtype=float)
    valid_counts = np.asarray(reference.get("reference_scatter_valid_n", []), dtype=float)
    if not (x_values.size and x_values.shape == mean_y.shape == std_y.shape == valid_counts.shape):
        return np.nan
    sem = np.full_like(std_y, np.nan, dtype=float)
    valid_n = np.isfinite(valid_counts) & (valid_counts > 0)
    sem[valid_n] = std_y[valid_n] / np.sqrt(valid_counts[valid_n])
    valid = np.isfinite(x_values) & np.isfinite(mean_y) & np.isfinite(sem)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size < 2:
        return np.nan
    satellite_count = max(2, int(round(valid_indices.size * fraction)))
    satellite_indices = valid_indices[:min(satellite_count, valid_indices.size)]
    denominator = _trapezoid_area(x_values[valid], np.abs(mean_y[valid]))
    numerator = _trapezoid_area(x_values[satellite_indices], np.abs(sem[satellite_indices]))
    if not np.isfinite(denominator) or denominator <= np.finfo(float).eps:
        return np.nan
    return float(numerator / denominator)


def _tail_baseline_level(y_values, edge_fraction=0.08):
    y_values = np.asarray(y_values, dtype=float)
    finite = np.isfinite(y_values)
    if y_values.size < 4 or not np.any(finite):
        return 0.0
    edge_count = max(3, int(round(y_values.size * float(edge_fraction))))
    edge_count = min(edge_count, max(3, y_values.size // 3))
    tail_values = np.concatenate([y_values[:edge_count], y_values[-edge_count:]])
    tail_values = tail_values[np.isfinite(tail_values)]
    if tail_values.size == 0:
        return 0.0
    return float(np.nanmedian(tail_values))


def _align_iad_baselines(roi_y, ref_y, edge_fraction=0.08):
    roi_y = np.asarray(roi_y, dtype=float)
    ref_y = np.asarray(ref_y, dtype=float)
    if roi_y.shape != ref_y.shape or roi_y.size < 4:
        return roi_y.copy(), ref_y.copy()
    roi_baseline = _tail_baseline_level(roi_y, edge_fraction=edge_fraction)
    ref_baseline = _tail_baseline_level(ref_y, edge_fraction=edge_fraction)
    return roi_y - roi_baseline, ref_y - ref_baseline


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

    roi_max = int(np.nanargmax(roi_y))
    ref_max = int(np.nanargmax(ref_y))
    shift = ref_max - roi_max
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


def _safe_int_pair(first, second):
    first, second = sorted((int(first), int(second)))
    return first, second


def _peak_fit_model_label(model):
    return "Lorentzian" if str(model) == "lorentzian" else "Pseudo-Voigt"


def _display_peak_label(label):
    text = str(label)
    compact = re.sub(r"\s+", "", text)
    label_map = {
        r"$K_{\beta'}$": "Kβ′",
        r"$K_{\beta,\mathrm{res}}$": "Kβᵣₑₛ",
        r"$K_{\beta_{1,3}}$": "Kβ₁,₃",
    }
    return label_map.get(compact, text)


def _format_fit_value(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value):
        return ""
    return f"{value:.6g}"


def _format_optional_float(value, precision=10):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value):
        return ""
    return f"{value:.{precision}g}"


def _format_fit_error(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value):
        return ""
    if value < 0.001:
        return f"{value:.3e}"
    return f"{100.0 * value:.2f}%"


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


def _finite_min_max(arrays):
    finite_parts = []
    for values in arrays:
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array)]
        if finite.size:
            finite_parts.append(finite)
    if not finite_parts:
        return 0.0, 1.0
    combined = np.concatenate(finite_parts)
    return float(np.nanmin(combined)), float(np.nanmax(combined))


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


def _score_background_band(data, invalid, start, height, col_begin, col_end, roi_center):
    stop = min(start + height, data.shape[0])
    if stop <= start:
        return None
    band = np.asarray(data[start:stop, col_begin:col_end + 1], dtype=float)
    if invalid is not None and invalid.shape == data.shape:
        band = np.where(invalid[start:stop, col_begin:col_end + 1], np.nan, band)
    finite = band[np.isfinite(band)]
    if finite.size < max(12, band.size * 0.25):
        return None
    median = float(np.nanmedian(finite))
    p90 = float(np.nanpercentile(finite, 90))
    spread = float(np.nanpercentile(finite, 75) - np.nanpercentile(finite, 25))
    distance = abs((start + stop - 1) / 2.0 - roi_center)
    return median + 0.35 * p90 + 0.75 * spread + 0.002 * distance


def _best_background_band(data, invalid, candidates, height, col_begin, col_end, roi_center):
    best = None
    for start in candidates:
        score = _score_background_band(data, invalid, int(start), height, col_begin, col_end, roi_center)
        if score is None:
            continue
        if best is None or score < best[0]:
            best = (score, int(start))
    if best is None:
        return None
    start = best[1]
    return start, min(start + height - 1, data.shape[0] - 1)


class CurrentPanelStack(QtWidgets.QStackedWidget):
    """Stack that keeps the controller area stable across page changes."""

    def _stable_size_hint(self, hint_name):
        width = 0
        height = 0
        for index in range(self.count()):
            widget = self.widget(index)
            if widget is None:
                continue
            hint = getattr(widget, hint_name)()
            width = max(width, hint.width())
            height = max(height, hint.height())
        if width > 0 or height > 0:
            return QtCore.QSize(width, height)
        return super().sizeHint()

    def sizeHint(self):
        return self._stable_size_hint("sizeHint")

    def minimumSizeHint(self):
        return self._stable_size_hint("minimumSizeHint")


class ButtonPanelWidget(QtWidgets.QWidget):
    """Button-driven panel switcher with native QPushButton styling."""

    def __init__(self, parent=None, role="main"):
        super().__init__(parent)
        self._role = role
        self._buttons = []
        self._stack = CurrentPanelStack()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2 if role == "sub" else 5)

        self._button_row = QtWidgets.QHBoxLayout()
        self._button_row.setContentsMargins(0, 0, 0, 0)
        self._button_row.setSpacing(0)
        if role == "sub":
            self._button_row.addStretch(1)
            self._button_row.addStretch(1)
        else:
            self._button_row.addStretch(1)
        layout.addLayout(self._button_row)
        divider = QtWidgets.QFrame()
        divider.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        divider.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        divider.setFixedHeight(1)
        divider.setStyleSheet(
            "background-color: #c9cfd6; border: 0;"
            if role == "main"
            else "background-color: #d6dce2; border: 0;"
        )
        layout.addWidget(divider)
        layout.addWidget(self._stack)
        self._apply_role_style()

    def _apply_role_style(self):
        if self._role == "main":
            self.setStyleSheet(
                "QPushButton[panelButton=\"true\"] {"
                "background: #eef1f4; color: #20262c; border: 1px solid #b9c1c8;"
                "border-radius: 0px; padding: 5px 16px; font-weight: 600;"
                "}"
                "QPushButton[panelButton=\"true\"]:checked {"
                "background: #1f6fb2; color: white; border-color: #15598f;"
                "border-radius: 0px;"
                "}"
                "QPushButton[panelButton=\"true\"]:hover:!checked { background: #e2e7eb; border-radius: 0px; }"
            )
        else:
            self.setStyleSheet(
                "QPushButton[panelButton=\"true\"] {"
                "background: #f4f7fa; color: #1f2933; border: 1px solid #bec8d2;"
                "border-radius: 0px; padding: 3px 14px; min-height: 22px;"
                "}"
                "QPushButton[panelButton=\"true\"]:checked {"
                "background: #d9eaff; color: #0d4f86; border-color: #6da7dc; font-weight: 600;"
                "border-radius: 0px;"
                "}"
                "QPushButton[panelButton=\"true\"]:hover:!checked { background: #ebf1f6; border-radius: 0px; }"
            )

    def addPanel(self, widget, text):
        index = self._stack.addWidget(widget)
        button = QtWidgets.QPushButton(text)
        button.setCheckable(True)
        button.setAutoDefault(False)
        button.setProperty("panelButton", True)
        button.style().unpolish(button)
        button.style().polish(button)
        if self._role == "main":
            button.setMinimumWidth(142)
        else:
            button.setMinimumWidth(132)
            if text in ("Spectrum Plotting", "Spectrum Reference"):
                button.setMinimumWidth(172)
        button.clicked.connect(lambda _checked=False, panel_index=index: self.setCurrentIndex(panel_index))
        insert_index = max(self._button_row.count() - 1, 0) if self._role == "sub" else len(self._buttons)
        self._button_row.insertWidget(insert_index, button)
        self._buttons.append(button)
        if index == 0:
            self.setCurrentIndex(0)
        return index

    def addHeaderWidget(self, widget, spacing=10):
        insert_index = max(self._button_row.count() - 1, 0)
        if spacing:
            self._button_row.insertSpacing(insert_index, spacing)
            insert_index += 1
        self._button_row.insertWidget(insert_index, widget)

    def setCurrentIndex(self, index):
        self._stack.setCurrentIndex(index)
        for button_index, button in enumerate(self._buttons):
            button.blockSignals(True)
            button.setChecked(button_index == index)
            button.blockSignals(False)
        self._stack.updateGeometry()
        self.updateGeometry()

    def count(self):
        return self._stack.count()

    def widget(self, index):
        return self._stack.widget(index)


class QtRangeBar(QtWidgets.QWidget):
    """Two-handle Qt range bar used in place of the old Tk RangeSlider."""

    rangeChanged = _QT_SIGNAL(int, int)

    def __init__(self, parent=None, height=44):
        super().__init__(parent)
        self.min_value = 0
        self.max_value = 1
        self.left_value = 0
        self.right_value = 1
        self._active_handle = None
        self._height = int(height)
        self._track_margin = 24
        self._handle_radius = 7
        self.setMinimumSize(260, self._height)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.setCursor(_pointing_cursor())

    def sizeHint(self):
        return QtCore.QSize(340, self._height)

    def setLimits(self, min_value, max_value, left_value=None, right_value=None, emit=False):
        self.min_value = int(min_value)
        self.max_value = max(int(max_value), self.min_value)
        left = self.left_value if left_value is None else left_value
        right = self.right_value if right_value is None else right_value
        self.setValues(left, right, emit=emit)

    def setValues(self, left_value, right_value, emit=False):
        self.left_value, self.right_value = self._clamp_pair(left_value, right_value)
        self.update()
        if emit:
            self.rangeChanged.emit(self.left_value, self.right_value)

    def _clamp(self, value):
        return min(max(int(round(value)), self.min_value), self.max_value)

    def _clamp_pair(self, left_value, right_value):
        left = self._clamp(left_value)
        right = self._clamp(right_value)
        if self.max_value <= self.min_value:
            return self.min_value, self.max_value
        if left >= right:
            if left >= self.max_value:
                left = self.max_value - 1
                right = self.max_value
            else:
                right = left + 1
        return left, right

    def _track_bounds(self):
        width = max(self.width(), self._track_margin * 2 + 1)
        return self._track_margin, width - self._track_margin

    def _value_span(self):
        return max(self.max_value - self.min_value, 1)

    def _value_to_x(self, value):
        left, right = self._track_bounds()
        fraction = (int(value) - self.min_value) / float(self._value_span())
        return left + fraction * (right - left)

    def _x_to_value(self, x_pos):
        left, right = self._track_bounds()
        x_pos = min(max(float(x_pos), left), right)
        fraction = (x_pos - left) / max(right - left, 1.0)
        return int(round(self.min_value + fraction * self._value_span()))

    def _event_x(self, event):
        point = event.position() if hasattr(event, 'position') else event.pos()
        return float(point.x())

    def _nearest_handle(self, x_pos):
        left_distance = abs(x_pos - self._value_to_x(self.left_value))
        right_distance = abs(x_pos - self._value_to_x(self.right_value))
        return 'left' if left_distance <= right_distance else 'right'

    def _set_active_value(self, x_pos):
        value = self._x_to_value(x_pos)
        if self._active_handle == 'left':
            value = min(value, self.right_value - 1) if self.max_value > self.min_value else self.min_value
            self.left_value = self._clamp(value)
        elif self._active_handle == 'right':
            value = max(value, self.left_value + 1) if self.max_value > self.min_value else self.max_value
            self.right_value = self._clamp(value)
        self.update()
        self.rangeChanged.emit(self.left_value, self.right_value)

    def mousePressEvent(self, event):
        if event.button() != _left_mouse_button():
            return
        self._active_handle = self._nearest_handle(self._event_x(event))
        self._set_active_value(self._event_x(event))

    def mouseMoveEvent(self, event):
        if self._active_handle:
            self._set_active_value(self._event_x(event))

    def mouseReleaseEvent(self, event):
        if self._active_handle:
            self._set_active_value(self._event_x(event))
            self._active_handle = None

    def _draw_value_label(self, painter, text, x_pos, y_pos, color):
        font = QtGui.QFont("Arial", 10)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        text_width = metrics.horizontalAdvance(text)
        rect = QtCore.QRectF(x_pos - text_width / 2.0 - 6, y_pos - 10, text_width + 12, 18)
        rect.moveLeft(max(2.0, min(rect.left(), self.width() - rect.width() - 2.0)))
        painter.setPen(QtGui.QPen(color, 1.0))
        painter.setBrush(QtGui.QColor(248, 251, 248, 230))
        painter.drawRoundedRect(rect, 4, 4)
        painter.setPen(QtGui.QPen(color))
        painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, text)

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(_painter_antialiasing(), True)
        y_pos = self.height() / 2.0 + 4
        track_left, track_right = self._track_bounds()
        left_x = self._value_to_x(self.left_value)
        right_x = self._value_to_x(self.right_value)

        painter.setPen(QtGui.QPen(QtGui.QColor("#9da7a1"), 2.0))
        painter.drawLine(QtCore.QPointF(track_left, y_pos), QtCore.QPointF(track_right, y_pos))
        active_pen = QtGui.QPen(QtGui.QColor("#1f5c3f"), 3.0)
        active_pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        painter.setPen(active_pen)
        painter.drawLine(QtCore.QPointF(left_x, y_pos), QtCore.QPointF(right_x, y_pos))

        painter.setPen(QtGui.QPen(QtGui.QColor("white"), 1.0))
        painter.setBrush(QtGui.QColor("#1f5c3f"))
        for x_pos in (left_x, right_x):
            painter.drawEllipse(QtCore.QPointF(x_pos, y_pos), self._handle_radius, self._handle_radius)
        self._draw_value_label(
            painter,
            f"{self.left_value}-{self.right_value}",
            (left_x + right_x) / 2.0,
            max(11.0, y_pos - 18.0),
            QtGui.QColor("#1f5c3f"),
        )


class QtDualRangeBar(QtRangeBar):
    """Four-handle Qt range bar for paired ROI/BG strips."""

    rangesChanged = _QT_SIGNAL(int, int, int, int)

    def __init__(self, parent=None, height=54, second_enabled=True):
        super().__init__(parent=parent, height=height)
        self.first_left = 0
        self.first_right = 1
        self.second_left = 0
        self.second_right = 1
        self.second_enabled = bool(second_enabled)
        self.setMinimumSize(260, self._height)

    def setSecondEnabled(self, enabled):
        self.second_enabled = bool(enabled)
        if self._active_handle and self._active_handle.startswith('second') and not self.second_enabled:
            self._active_handle = None
        self.update()

    def setLimits(
        self,
        min_value,
        max_value,
        first_left=None,
        first_right=None,
        second_left=None,
        second_right=None,
        emit=False,
    ):
        self.min_value = int(min_value)
        self.max_value = max(int(max_value), self.min_value)
        self.setValues(
            self.first_left if first_left is None else first_left,
            self.first_right if first_right is None else first_right,
            self.second_left if second_left is None else second_left,
            self.second_right if second_right is None else second_right,
            emit=emit,
        )

    def setValues(self, first_left, first_right, second_left, second_right, emit=False):
        self.first_left, self.first_right = self._clamp_pair(first_left, first_right)
        self.second_left, self.second_right = self._clamp_pair(second_left, second_right)
        self.update()
        if emit:
            self.rangesChanged.emit(self.first_left, self.first_right, self.second_left, self.second_right)

    def _handle_values(self):
        values = {
            'first_left': self.first_left,
            'first_right': self.first_right,
        }
        if self.second_enabled:
            values.update({
                'second_left': self.second_left,
                'second_right': self.second_right,
            })
        return values

    def _nearest_handle(self, x_pos):
        return min(
            self._handle_values(),
            key=lambda handle: abs(x_pos - self._value_to_x(self._handle_values()[handle])),
        )

    def _set_active_value(self, x_pos):
        value = self._x_to_value(x_pos)
        if self._active_handle == 'first_left':
            self.first_left = self._clamp(min(value, self.first_right - 1))
        elif self._active_handle == 'first_right':
            self.first_right = self._clamp(max(value, self.first_left + 1))
        elif self._active_handle == 'second_left':
            self.second_left = self._clamp(min(value, self.second_right - 1))
        elif self._active_handle == 'second_right':
            self.second_right = self._clamp(max(value, self.second_left + 1))
        self.update()
        self.rangesChanged.emit(self.first_left, self.first_right, self.second_left, self.second_right)

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(_painter_antialiasing(), True)
        track_left, track_right = self._track_bounds()
        y_pos = self.height() / 2.0 + 2

        painter.setPen(QtGui.QPen(QtGui.QColor("#9da7a1"), 2.0))
        painter.drawLine(QtCore.QPointF(track_left, y_pos), QtCore.QPointF(track_right, y_pos))

        first_left_x = self._value_to_x(self.first_left)
        first_right_x = self._value_to_x(self.first_right)
        second_left_x = self._value_to_x(self.second_left)
        second_right_x = self._value_to_x(self.second_right)

        first_color = QtGui.QColor("#7b3294")
        second_color = QtGui.QColor("#008837")
        first_pen = QtGui.QPen(first_color, 4.0)
        second_pen = QtGui.QPen(second_color, 4.0)
        first_pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        second_pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)

        painter.setPen(first_pen)
        painter.drawLine(QtCore.QPointF(first_left_x, y_pos - 4), QtCore.QPointF(first_right_x, y_pos - 4))
        if self.second_enabled:
            painter.setPen(second_pen)
            painter.drawLine(QtCore.QPointF(second_left_x, y_pos + 4), QtCore.QPointF(second_right_x, y_pos + 4))

        painter.setPen(QtGui.QPen(QtGui.QColor("white"), 1.0))
        painter.setBrush(first_color)
        for x_pos in (first_left_x, first_right_x):
            painter.drawEllipse(QtCore.QPointF(x_pos, y_pos), self._handle_radius, self._handle_radius)
        if self.second_enabled:
            painter.setBrush(second_color)
            for x_pos in (second_left_x, second_right_x):
                painter.drawEllipse(QtCore.QPointF(x_pos, y_pos), self._handle_radius, self._handle_radius)

        self._draw_value_label(
            painter,
            f"{self.first_left}-{self.first_right}",
            (first_left_x + first_right_x) / 2.0,
            max(10.0, y_pos - 18.0),
            first_color,
        )
        if self.second_enabled:
            self._draw_value_label(
                painter,
                f"{self.second_left}-{self.second_right}",
                (second_left_x + second_right_x) / 2.0,
                min(self.height() - 10.0, y_pos + 19.0),
                second_color,
            )


class QtXESAnalyzer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IXE Analyzer - Qt")
        self.resize(1320, 860)

        self.parm = {
            'n_moveavg': 5,
            'PK intersect': {'i_l': 50, 'i_r': 120},
            'eye_ball_cross': None,
            'peak_fit': {
                'n_peaks': 3,
                'x_min': None,
                'x_max': None,
                'peak_shape': 'pseudo_voigt',
                'physical_fit': True,
                'tail_baseline': True,
            },
            'ccd_gap': {
                'enabled': True,
                'max_width': 8,
                'drop_fraction': 0.65,
                'mask_max_width': 16,
                'mask_drop_fraction': 0.55,
                'mask_dilate': 1,
                'mask_center_window': 0.22,
                'mask_row_center_window': 0.22,
                'mask_col_center_window': 0.20,
                'edge_valid_fraction': 0.80,
            },
        }
        self.filepaths = []
        self.reference_spectra = []
        self.reference_average_spectra = []
        self.last_reference_average = None
        self.imm = None
        self.immm = None
        self.raw_gap_mask = None
        self.processed_gap_mask = None
        self.processed_detector_background_mask = None
        self.current_tilt_text = "Tilt: --"
        self.spect_processor = None
        self.last_x = np.array([], dtype=float)
        self.last_y = np.array([], dtype=float)
        self.current_tilt_text = "Tilt: --"

        self._image_region_items = []
        self._spectrum_items = []
        self._legend = None
        self._setup_ui()

    def _setup_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layout.addWidget(self._build_controls())

        control_separator = QtWidgets.QFrame()
        control_separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        control_separator.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        control_separator.setFixedHeight(1)
        control_separator.setStyleSheet("background-color: #c7cdd2; border: 0;")
        layout.addWidget(control_separator)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_image_panel())
        splitter.addWidget(self._build_spectrum_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([520, 780])
        layout.addWidget(splitter, stretch=1)

        self.statusBar().showMessage("Ready")

    def _build_controls(self):
        panels = ButtonPanelWidget()
        panels.setObjectName("controllerPanel")
        panels.setStyleSheet(
            "#controllerPanel {"
            "background-color: #f5f6f7;"
            "border: 1px solid #d7dce0;"
            "border-radius: 3px;"
            "}"
        )
        panels.layout().setContentsMargins(6, 6, 6, 6)
        panels.addPanel(self._build_image_controls_tab(), "Image Processing")
        panels.addPanel(self._build_spectrum_controls_tab(), "Spectrum Analysis")
        panels.addPanel(self._build_iad_controls_tab(), "IAD Calculation")
        panels.addPanel(self._build_calibration_controls_tab(), "Calibration")
        self.global_save_project_button = QtWidgets.QPushButton("Save Project")
        self.global_save_project_button.setAutoDefault(False)
        self.global_save_project_button.setFixedWidth(116)
        self.global_save_project_button.setToolTip("Save the full IXE project")
        self.global_save_project_button.clicked.connect(self.save_project)
        self._style_active_button(self.global_save_project_button)
        panels.addHeaderWidget(self.global_save_project_button, spacing=12)
        return panels

    def _build_image_controls_tab(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        file_row = QtWidgets.QHBoxLayout()
        file_row.setContentsMargins(0, 0, 0, 0)
        file_row.setSpacing(8)

        self.import_button = QtWidgets.QPushButton("Import Image")
        self.import_stack_button = QtWidgets.QPushButton("Import Stack Image")
        self.import_button.setFixedWidth(148)
        self.import_stack_button.setFixedWidth(176)
        self.import_button.clicked.connect(self.import_tiff)
        self.import_stack_button.clicked.connect(self.import_tiff_stack)
        self.file_path_edit = QtWidgets.QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.file_path_edit.setPlaceholderText("No file loaded")
        self.file_path_edit.setMinimumWidth(360)
        self.file_path_edit.setMaximumWidth(760)
        self.file_path_edit.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.run_label = QtWidgets.QLabel("No image loaded")
        self.run_label.setMinimumWidth(88)

        file_row.addWidget(self.import_button)
        file_row.addWidget(self.import_stack_button)
        file_row.addWidget(QtWidgets.QLabel("File Path:"))
        file_row.addWidget(self.file_path_edit, stretch=1)
        file_row.addWidget(self.run_label)
        file_row.addStretch(1)
        layout.addLayout(file_row)

        self.vmin_spin = QtWidgets.QDoubleSpinBox()
        self.vmin_spin.setRange(-1e9, 1e9)
        self.vmin_spin.setDecimals(0)
        self.vmin_spin.setFixedWidth(72)
        self.vmin_spin.setValue(0.0)
        self.vmax_spin = QtWidgets.QDoubleSpinBox()
        self.vmax_spin.setRange(-1e9, 1e9)
        self.vmax_spin.setDecimals(0)
        self.vmax_spin.setFixedWidth(72)
        self.vmax_spin.setValue(100.0)
        self.vmin_spin.valueChanged.connect(self.refresh_image)
        self.vmax_spin.valueChanged.connect(self.refresh_image)

        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(["viridis", "plasma", "inferno", "magma", "cividis", "gray", "Greys", "turbo"])
        self.cmap_combo.setFixedWidth(104)
        self.cmap_combo.currentTextChanged.connect(self.update_cmap)
        self.manual_tilt_spin = QtWidgets.QDoubleSpinBox()
        self.manual_tilt_spin.setRange(-20.0, 20.0)
        self.manual_tilt_spin.setDecimals(2)
        self.manual_tilt_spin.setSingleStep(0.05)
        self.manual_tilt_spin.setSuffix(" deg")
        self.manual_tilt_spin.setFixedWidth(102)
        self.manual_tilt_spin.setValue(0.0)

        self.process_button = QtWidgets.QPushButton("Tilt Correction")
        self.process_button.setFixedWidth(132)
        self.process_button.clicked.connect(self.process_image)
        self._style_active_button(self.process_button)
        self.apply_tilt_button = QtWidgets.QPushButton("Manual Tilt")
        self.apply_tilt_button.setFixedWidth(104)
        self.apply_tilt_button.clicked.connect(self.apply_manual_tilt)

        gap_button = QtWidgets.QPushButton("Gap Mask")
        gap_button.clicked.connect(self.show_gap_mask)
        save_image_button = QtWidgets.QPushButton("Export Image")
        save_image_button.clicked.connect(self.save_image_png)
        open_project_button = QtWidgets.QPushButton("Open Project")
        open_project_button.clicked.connect(self.open_project)
        save_project_button = QtWidgets.QPushButton("Export Project")
        save_project_button.clicked.connect(self.save_project)
        for button in (gap_button, save_image_button, open_project_button, save_project_button):
            button.setFixedWidth(132)

        action_row = QtWidgets.QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(8)
        action_row.addWidget(self.process_button)
        action_row.addWidget(QtWidgets.QLabel("Tilt:"))
        action_row.addWidget(self.manual_tilt_spin)
        action_row.addWidget(self.apply_tilt_button)
        action_row.addWidget(gap_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        project_row = QtWidgets.QHBoxLayout()
        project_row.setContentsMargins(0, 0, 0, 0)
        project_row.setSpacing(8)
        project_row.addWidget(open_project_button)
        project_row.addWidget(save_project_button)
        project_row.addWidget(save_image_button)
        project_row.addStretch(1)
        layout.addLayout(project_row)
        layout.addStretch(1)
        return panel

    def _build_spectrum_controls_tab(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(4, 1, 4, 4)
        layout.setSpacing(1)

        self.spectrum_control_tabs = ButtonPanelWidget(role="sub")
        self.spectrum_control_tabs.addPanel(self._build_spectrum_plotting_tab(), "Spectrum Plotting")
        self.spectrum_control_tabs.addPanel(self._build_spectrum_fitting_tab(), "Peak Fitting")
        self.spectrum_control_tabs.addPanel(self._build_spectrum_reference_tab(), "Spectrum Reference")
        layout.addWidget(self.spectrum_control_tabs)
        return panel

    def _build_spectrum_plotting_tab(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(3)

        selector_grid = QtWidgets.QGridLayout()
        selector_grid.setContentsMargins(0, 0, 0, 0)
        selector_grid.setHorizontalSpacing(8)
        selector_grid.setVerticalSpacing(0)
        selector_grid.setColumnStretch(1, 1)

        self.roi_row_start = self._spin()
        self.roi_row_end = self._spin()
        self.roi2_row_start = self._spin()
        self.roi2_row_end = self._spin()
        self.roi_col_start = self._spin()
        self.roi_col_end = self._spin()
        self.bg_row_start = self._spin()
        self.bg_row_end = self._spin()
        self.bg2_row_start = self._spin()
        self.bg2_row_end = self._spin()

        for spin in (
            self.roi_row_start,
            self.roi_row_end,
            self.roi2_row_start,
            self.roi2_row_end,
            self.roi_col_start,
            self.roi_col_end,
            self.bg_row_start,
            self.bg_row_end,
            self.bg2_row_start,
            self.bg2_row_end,
        ):
            spin.valueChanged.connect(self.refresh_regions)

        self.roi2_check = QtWidgets.QCheckBox("ROI 2")
        self.roi2_check.setVisible(False)
        self.roi2_check.stateChanged.connect(self._update_roi2_widgets)
        self.roi2_check.stateChanged.connect(self.refresh_regions)

        self.roi_row_bar = QtDualRangeBar(height=46, second_enabled=False)
        self.roi_row_bar.rangesChanged.connect(self._on_roi_row_bar_changed)
        self.roi_col_bar = QtRangeBar(height=36)
        self.roi_col_bar.rangeChanged.connect(self._on_roi_col_bar_changed)
        self.bg_row_bar = QtDualRangeBar(height=46, second_enabled=True)
        self.bg_row_bar.rangesChanged.connect(self._on_bg_row_bar_changed)

        self.roi_add_button = QtWidgets.QPushButton("Add a row")
        self.roi_add_button.setFixedWidth(92)
        self.roi_add_button.clicked.connect(self.add_roi_row_range)
        self.roi_remove_button = QtWidgets.QPushButton("Subtract a row")
        self.roi_remove_button.setFixedWidth(116)
        self.roi_remove_button.clicked.connect(self.remove_roi_row_range)

        roi_actions = QtWidgets.QHBoxLayout()
        roi_actions.setContentsMargins(0, 0, 0, 0)
        roi_actions.setSpacing(4)
        roi_actions.addWidget(self.roi_add_button)
        roi_actions.addWidget(self.roi_remove_button)

        roi_label = QtWidgets.QLabel("ROI Rows:")
        roi_label.setFixedWidth(102)
        roi_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        selector_grid.addWidget(roi_label, 0, 0)
        selector_grid.addWidget(self.roi_row_bar, 0, 1)
        selector_grid.addLayout(roi_actions, 0, 2)

        self.plot_button = QtWidgets.QPushButton("Plot")
        self.plot_button.setFixedWidth(110)
        self.plot_button.clicked.connect(self.plot_spectrum)
        self._style_active_button(self.plot_button)

        self.gap_enabled_check = QtWidgets.QCheckBox()
        self.gap_enabled_check.setChecked(True)
        self.gap_enabled_check.stateChanged.connect(self.toggle_gap_correction)
        self.gap_check = QtWidgets.QPushButton("Gap Mask")
        self.gap_check.setFixedWidth(132)
        self.gap_check.clicked.connect(self.toggle_gap_correction)

        col_actions = QtWidgets.QHBoxLayout()
        col_actions.setContentsMargins(0, 0, 0, 0)
        col_actions.setSpacing(6)
        col_actions.addWidget(self.plot_button)
        col_actions.addWidget(self.gap_enabled_check)
        col_actions.addWidget(self.gap_check)

        col_label = QtWidgets.QLabel("ROI Columns:")
        col_label.setFixedWidth(102)
        col_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        selector_grid.addWidget(col_label, 1, 0)
        selector_grid.addWidget(self.roi_col_bar, 1, 1)
        selector_grid.addLayout(col_actions, 1, 2)

        self.auto_bg_button = QtWidgets.QPushButton("Auto BG")
        self.auto_bg_button.setFixedWidth(110)
        self.auto_bg_button.clicked.connect(self.auto_background)
        self.bg_enabled_check = QtWidgets.QCheckBox()
        self.bg_enabled_check.setChecked(False)
        self.bg_enabled_check.stateChanged.connect(self.toggle_background_removal)
        self.bg_check = QtWidgets.QPushButton("BG Remove")
        self.bg_check.setFixedWidth(132)
        self.bg_check.clicked.connect(self.toggle_background_removal)

        bg_actions = QtWidgets.QHBoxLayout()
        bg_actions.setContentsMargins(0, 0, 0, 0)
        bg_actions.setSpacing(6)
        bg_actions.addWidget(self.auto_bg_button)
        bg_actions.addWidget(self.bg_enabled_check)
        bg_actions.addWidget(self.bg_check)

        bg_label = QtWidgets.QLabel("BG Rows:")
        bg_label.setFixedWidth(102)
        bg_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        selector_grid.addWidget(bg_label, 2, 0)
        selector_grid.addWidget(self.bg_row_bar, 2, 1)
        selector_grid.addLayout(bg_actions, 2, 2)
        layout.addLayout(selector_grid)

        self.line_color = QtGui.QColor("#202020")
        color_button = QtWidgets.QPushButton("Pick color")
        color_button.setFixedWidth(104)
        color_button.clicked.connect(self.pick_line_color)
        self.line_style_combo = QtWidgets.QComboBox()
        self.line_style_combo.addItems(["-", "--", "-.", ":"])
        self.line_style_combo.setFixedWidth(64)
        self.line_style_combo.currentTextChanged.connect(self.update_plot_style)
        self.line_width_spin = QtWidgets.QDoubleSpinBox()
        self.line_width_spin.setRange(0.1, 10.0)
        self.line_width_spin.setDecimals(2)
        self.line_width_spin.setSingleStep(0.1)
        self.line_width_spin.setFixedWidth(74)
        self.line_width_spin.setValue(1.6)
        self.line_width_spin.valueChanged.connect(self.update_plot_style)
        save_spectrum_button = QtWidgets.QPushButton("Export Spectrum")
        save_spectrum_button.setFixedWidth(132)
        save_spectrum_button.clicked.connect(self.save_spectrum_data)
        self.save_svg_button = QtWidgets.QPushButton("Export Image")
        self.save_svg_button.setFixedWidth(116)
        self.save_svg_button.clicked.connect(self.save_spectrum_svg)

        line_row = QtWidgets.QHBoxLayout()
        line_row.setContentsMargins(0, 0, 0, 0)
        line_row.setSpacing(8)
        line_row.addWidget(QtWidgets.QLabel("Line Properties"))
        line_row.addWidget(color_button)
        line_row.addWidget(QtWidgets.QLabel("Line Style:"))
        line_row.addWidget(self.line_style_combo)
        line_row.addWidget(QtWidgets.QLabel("Line Width:"))
        line_row.addWidget(self.line_width_spin)
        group_divider = QtWidgets.QFrame()
        group_divider.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        group_divider.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        group_divider.setFixedWidth(1)
        group_divider.setStyleSheet("background-color: #c9cfd6; border: 0;")
        line_row.addSpacing(6)
        line_row.addWidget(group_divider)
        line_row.addSpacing(6)
        line_row.addWidget(save_spectrum_button)
        line_row.addWidget(self.save_svg_button)
        line_row.addStretch(1)
        layout.addLayout(line_row)

        self._update_roi2_widgets()
        return panel

    def _build_spectrum_fitting_tab(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Fit Model:"))
        self.peak_fit_model_group = QtWidgets.QButtonGroup(self)
        self.pseudo_voigt_radio = QtWidgets.QRadioButton("Pseudo-Voigt")
        self.lorentzian_radio = QtWidgets.QRadioButton("Lorentzian")
        self.pseudo_voigt_radio.setChecked(True)
        self.peak_fit_model_group.addButton(self.pseudo_voigt_radio)
        self.peak_fit_model_group.addButton(self.lorentzian_radio)
        self.pseudo_voigt_radio.toggled.connect(self._update_peak_fit_model)
        controls.addWidget(self.pseudo_voigt_radio)
        controls.addWidget(self.lorentzian_radio)
        peak_defaults = self.parm.get('peak_fit', {})
        self.physical_fit_check = QtWidgets.QCheckBox("Physical Fit")
        self.physical_fit_check.setChecked(bool(peak_defaults.get('physical_fit', True)))
        self.physical_fit_check.stateChanged.connect(self._update_peak_fit_model)
        controls.addWidget(self.physical_fit_check)
        self.tail_baseline_check = QtWidgets.QCheckBox("Tail Baseline")
        self.tail_baseline_check.setChecked(bool(peak_defaults.get('tail_baseline', True)))
        self.tail_baseline_check.stateChanged.connect(self._update_peak_fit_model)
        controls.addWidget(self.tail_baseline_check)

        self.peak_fit_count = QtWidgets.QSpinBox()
        self.peak_fit_count.setRange(1, 12)
        self.peak_fit_count.setValue(int(peak_defaults.get('n_peaks', 3)))
        self.peak_fit_start = QtWidgets.QLineEdit()
        self.peak_fit_start.setFixedWidth(80)
        self.peak_fit_end = QtWidgets.QLineEdit()
        self.peak_fit_end.setFixedWidth(80)

        def fit_divider():
            divider = QtWidgets.QFrame()
            divider.setFrameShape(QtWidgets.QFrame.Shape.VLine)
            divider.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
            divider.setFixedWidth(1)
            divider.setStyleSheet("background-color: #c9cfd6; border: 0;")
            return divider

        controls.addSpacing(10)
        controls.addWidget(QtWidgets.QLabel("Peaks:"))
        controls.addWidget(self.peak_fit_count)
        controls.addSpacing(10)
        controls.addWidget(QtWidgets.QLabel("Fit Range:"))
        controls.addWidget(self.peak_fit_start)
        controls.addWidget(QtWidgets.QLabel("-"))
        controls.addWidget(self.peak_fit_end)

        peak_fit_button = QtWidgets.QPushButton("Peak Fit")
        peak_fit_button.setFixedWidth(96)
        peak_fit_button.clicked.connect(self.show_peak_fit_profile)
        self._style_active_button(peak_fit_button)
        save_pkfit_button = QtWidgets.QPushButton("Export PKfit")
        save_pkfit_button.setFixedWidth(116)
        save_pkfit_button.clicked.connect(self.save_peak_fit_data)
        save_params_button = QtWidgets.QPushButton("Export Params")
        save_params_button.setFixedWidth(116)
        save_params_button.clicked.connect(self.save_peak_fit_parameters)
        controls.addSpacing(8)
        controls.addWidget(fit_divider())
        controls.addSpacing(8)
        controls.addWidget(peak_fit_button)
        controls.addSpacing(8)
        controls.addWidget(fit_divider())
        controls.addSpacing(8)
        controls.addWidget(QtWidgets.QLabel("Fit Error:"))
        self.peak_fit_error_entry = QtWidgets.QLineEdit()
        self.peak_fit_error_entry.setReadOnly(True)
        self.peak_fit_error_entry.setFixedWidth(88)
        self.peak_fit_error_entry.setToolTip(
            "integral(abs(ROI corrected - Peak fit)) / integral(abs(ROI corrected))"
        )
        controls.addWidget(self.peak_fit_error_entry)
        controls.addSpacing(8)
        controls.addWidget(fit_divider())
        controls.addSpacing(8)
        controls.addWidget(save_pkfit_button)
        controls.addWidget(save_params_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.peak_fit_summary_label = QtWidgets.QLabel("Peak fit parameters: not fitted")
        layout.addWidget(self.peak_fit_summary_label)
        self.peak_fit_table = QtWidgets.QTableWidget(0, 6)
        self.peak_fit_table.setHorizontalHeaderLabels(["Peak", "Model", "L frac", "Position", "Width", "Area"])
        self.peak_fit_table.verticalHeader().setVisible(False)
        self.peak_fit_table.verticalHeader().setDefaultSectionSize(20)
        self.peak_fit_table.verticalHeader().setMinimumSectionSize(18)
        self.peak_fit_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.peak_fit_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.peak_fit_table.setMaximumHeight(88)
        self.peak_fit_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.peak_fit_table)
        return panel

    def _build_spectrum_reference_tab(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        button_row = QtWidgets.QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(8)
        import_button = QtWidgets.QPushButton("Import")
        import_button.clicked.connect(self.import_average_references)
        remove_button = QtWidgets.QPushButton("Remove")
        remove_button.clicked.connect(self.remove_average_references)
        average_button = QtWidgets.QPushButton("Average")
        average_button.clicked.connect(self.average_reference_spectra)
        self._style_active_button(average_button)
        save_button = QtWidgets.QPushButton("Export Reference")
        save_button.clicked.connect(self.save_average_reference)
        for button, width in (
            (import_button, 96),
            (remove_button, 96),
            (average_button, 104),
            (save_button, 134),
        ):
            button.setFixedWidth(width)
            button_row.addWidget(button)
        self.reference_average_summary = QtWidgets.QLabel("No reference pkfit imported")
        self.reference_average_summary.setStyleSheet("color: #444;")
        button_row.addSpacing(10)
        button_row.addWidget(self.reference_average_summary)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.reference_average_table = QtWidgets.QTableWidget(0, 5)
        self.reference_average_table.setHorizontalHeaderLabels(["Use", "Run", "Kβ₁,₃ px", "Fit error", "Shift"])
        self.reference_average_table.verticalHeader().setVisible(False)
        self.reference_average_table.verticalHeader().setDefaultSectionSize(20)
        self.reference_average_table.verticalHeader().setMinimumSectionSize(18)
        self.reference_average_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.reference_average_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.reference_average_table.setMaximumHeight(128)
        header = self.reference_average_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Fixed)
        header.resizeSection(0, 44)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.reference_average_table, stretch=1)
        return panel

    def _build_iad_controls_tab(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(4)

        button_row = QtWidgets.QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(8)
        for text, width, slot in (
            ("Import Ref.", 116, self.import_reference_spectrum),
            ("Remove", 76, self.remove_iad_reference),
            ("Color", 76, self.pick_iad_reference_color),
        ):
            button = QtWidgets.QPushButton(text)
            button.setFixedWidth(width)
            button.clicked.connect(slot)
            button_row.addWidget(button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        action_row = QtWidgets.QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(8)

        def iad_divider():
            divider = QtWidgets.QFrame()
            divider.setFrameShape(QtWidgets.QFrame.Shape.VLine)
            divider.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
            divider.setFixedWidth(1)
            divider.setStyleSheet("background-color: #c9cfd6; border: 0;")
            return divider

        for text, width, slot in (
            ("IAD", 82, self.plot_integrated_diff),
            ("IAD Error", 104, self.calculate_iad_error),
        ):
            button = QtWidgets.QPushButton(text)
            button.setFixedWidth(width)
            button.clicked.connect(slot)
            action_row.addWidget(button)
        action_row.addWidget(iad_divider())
        for text, width, slot in (
            ("Satellite IAD", 126, self.calculate_satellite_iad),
            ("Satellite IAD error", 158, self.calculate_satellite_iad_error),
        ):
            button = QtWidgets.QPushButton(text)
            button.setFixedWidth(width)
            button.clicked.connect(slot)
            action_row.addWidget(button)
        action_row.addWidget(iad_divider())
        export_iad_button = QtWidgets.QPushButton("Export IAD Values")
        export_iad_button.setFixedWidth(152)
        export_iad_button.clicked.connect(self.save_iad_results)
        action_row.addWidget(export_iad_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        self.cross_begin = QtWidgets.QLineEdit(str(self.parm['PK intersect']['i_l']))
        self.cross_begin.setFixedWidth(54)
        self.cross_end = QtWidgets.QLineEdit(str(self.parm['PK intersect']['i_r']))
        self.cross_end.setFixedWidth(54)
        self.eye_ball_cross = QtWidgets.QLineEdit("")
        self.eye_ball_cross.setFixedWidth(54)
        cross_row = QtWidgets.QHBoxLayout()
        cross_row.setContentsMargins(0, 0, 0, 0)
        cross_row.setSpacing(6)
        cross_row.addWidget(QtWidgets.QLabel("Spectra Crossing Range:"))
        cross_row.addWidget(self.cross_begin)
        cross_row.addWidget(QtWidgets.QLabel("-"))
        cross_row.addWidget(self.cross_end)
        cross_row.addSpacing(16)
        cross_row.addWidget(QtWidgets.QLabel("Manual crossing:"))
        cross_row.addWidget(self.eye_ball_cross)
        self.iad_baseline_check = QtWidgets.QCheckBox("Align baseline")
        self.iad_baseline_check.setChecked(True)
        cross_row.addSpacing(16)
        cross_row.addWidget(self.iad_baseline_check)
        self.iad_mc_trials_spin = QtWidgets.QSpinBox()
        self.iad_mc_trials_spin.setRange(20, 1000)
        self.iad_mc_trials_spin.setSingleStep(50)
        self.iad_mc_trials_spin.setValue(IAD_MC_DEFAULT_TRIALS)
        self.iad_mc_trials_spin.setFixedWidth(78)
        self.iad_mc_trials_spin.setToolTip("Monte Carlo re-fit trials used by IAD Error and Satellite IAD error.")
        self.iad_mc_trials_spin.valueChanged.connect(self._clear_iad_mc_cache)
        cross_row.addSpacing(16)
        cross_row.addWidget(QtWidgets.QLabel("Monte Carlo trials:"))
        cross_row.addWidget(self.iad_mc_trials_spin)
        cross_row.addStretch(1)
        layout.addLayout(cross_row)

        self.iad_line_table = QtWidgets.QTableWidget(0, 6)
        self.iad_line_table.setHorizontalHeaderLabels(["Use", "Reference", "IAD", "IAD error", "Satellite IAD", "Satellite IAD error"])
        self.iad_line_table.verticalHeader().setVisible(False)
        self.iad_line_table.verticalHeader().setDefaultSectionSize(20)
        self.iad_line_table.verticalHeader().setMinimumSectionSize(18)
        self.iad_line_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.iad_line_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.iad_line_table.setMaximumHeight(118)
        header = self.iad_line_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.iad_line_table.setColumnWidth(0, 44)
        self.iad_line_table.setColumnWidth(2, 84)
        self.iad_line_table.setColumnWidth(3, 94)
        self.iad_line_table.setColumnWidth(4, 108)
        self.iad_line_table.setColumnWidth(5, 92)
        layout.addWidget(self.iad_line_table, stretch=1)
        return panel

    def _build_calibration_controls_tab(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(5)

        cal_buttons = [
            ("Import Cal.", 96, self.import_calibration_spectrum),
            ("Fit Cal.", 76, self.fit_calibration_spectrum),
            ("Calc Map", 88, self.calculate_energy_calibration),
            ("Show Cal. Fit", 106, self.show_calibration_fit),
            ("Compare Cal.", 108, self.compare_calibration_overlay),
            ("Apply Cal.", 88, self.apply_energy_calibration),
            ("Export Cal.", 88, self.save_calibrated_spectrum_data),
        ]
        button_row = QtWidgets.QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(6)
        for index, (text, width, slot) in enumerate(cal_buttons):
            button = QtWidgets.QPushButton(text)
            button.setFixedWidth(width)
            button.clicked.connect(slot)
            button_row.addWidget(button)
            if index in (2, 4):
                group_divider = QtWidgets.QFrame()
                group_divider.setFrameShape(QtWidgets.QFrame.Shape.VLine)
                group_divider.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
                group_divider.setFixedWidth(1)
                group_divider.setStyleSheet("background-color: #c9cfd6; border: 0;")
                button_row.addSpacing(5)
                button_row.addWidget(group_divider)
                button_row.addSpacing(5)
        file_label = QtWidgets.QLabel("Cal file:")
        file_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.calibration_file_label = QtWidgets.QLineEdit("No calibration file loaded")
        self.calibration_file_label.setReadOnly(True)
        self.calibration_file_label.setFixedWidth(330)
        self.calibration_file_label.setStyleSheet("color: #555; background: #f8f9fa;")
        button_row.addSpacing(12)
        button_row.addWidget(file_label)
        button_row.addWidget(self.calibration_file_label)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.cal_fit_start = QtWidgets.QLineEdit()
        self.cal_fit_start.setFixedWidth(72)
        self.cal_fit_end = QtWidgets.QLineEdit()
        self.cal_fit_end.setFixedWidth(72)
        fit_row = QtWidgets.QHBoxLayout()
        fit_row.setContentsMargins(0, 0, 0, 0)
        fit_row.setSpacing(6)
        fit_label = QtWidgets.QLabel("Cal. fit range:")
        fit_label.setFixedWidth(102)
        fit_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        fit_row.addWidget(fit_label)
        fit_row.addWidget(self.cal_fit_start)
        fit_row.addWidget(QtWidgets.QLabel("-"))
        fit_row.addWidget(self.cal_fit_end)
        fit_row.addSpacing(14)
        fit_hint = QtWidgets.QLabel("Use Kβ′ + Kβ₁,₃ anchors; Kβᵣₑₛ is residual check")
        fit_hint.setStyleSheet("color: #666; font-size: 12px;")
        fit_row.addWidget(fit_hint)
        fit_row.addStretch(1)
        layout.addLayout(fit_row)

        self.calibration_outputs = {}
        output_grid = QtWidgets.QGridLayout()
        output_grid.setContentsMargins(0, 0, 0, 0)
        output_grid.setHorizontalSpacing(14)
        output_grid.setVerticalSpacing(5)
        peak_columns = [
            ("Kβ′", "Current px Kβ′", "Cal E Kβ′"),
            ("Kβᵣₑₛ", "Current px Kβᵣₑₛ", "Cal E Kβᵣₑₛ"),
            ("Kβ₁,₃", "Current px Kβ₁,₃", "Cal E Kβ₁,₃"),
        ]
        blank = QtWidgets.QLabel("")
        blank.setFixedWidth(102)
        output_grid.addWidget(blank, 0, 0)
        for column_index, (peak_label, _current_key, _cal_key) in enumerate(peak_columns, start=1):
            header = QtWidgets.QLabel(peak_label)
            header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            header.setStyleSheet("font-weight: 600; color: #333;")
            output_grid.addWidget(header, 0, column_index)
        for row_index, (row_label_text, key_offset) in enumerate((("Current px:", 1), ("Cal E:", 2)), start=1):
            row_label = QtWidgets.QLabel(row_label_text)
            row_label.setFixedWidth(102)
            row_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            output_grid.addWidget(row_label, row_index, 0)
            for column_index, (_peak_label, current_key, cal_key) in enumerate(peak_columns, start=1):
                key = current_key if key_offset == 1 else cal_key
                entry = QtWidgets.QLineEdit()
                entry.setReadOnly(True)
                entry.setFixedWidth(126)
                self.calibration_outputs[key] = entry
                output_grid.addWidget(entry, row_index, column_index)
        output_row = QtWidgets.QHBoxLayout()
        output_row.setContentsMargins(0, 0, 0, 0)
        output_row.addLayout(output_grid)
        output_row.addStretch(1)
        layout.addLayout(output_row)

        self.cal_slope_entry = QtWidgets.QLineEdit()
        self.cal_slope_entry.setReadOnly(True)
        self.cal_slope_entry.setFixedWidth(112)
        self.cal_intercept_entry = QtWidgets.QLineEdit()
        self.cal_intercept_entry.setReadOnly(True)
        self.cal_intercept_entry.setFixedWidth(112)
        self.cal_residual_entry = QtWidgets.QLineEdit()
        self.cal_residual_entry.setReadOnly(True)
        self.cal_residual_entry.setFixedWidth(104)
        equation_row = QtWidgets.QHBoxLayout()
        equation_row.setContentsMargins(0, 0, 0, 0)
        equation_row.setSpacing(6)
        equation_label = QtWidgets.QLabel("E =")
        equation_label.setFixedWidth(102)
        equation_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        equation_row.addWidget(equation_label)
        equation_row.addWidget(self.cal_slope_entry)
        equation_row.addWidget(QtWidgets.QLabel("* pixel +"))
        equation_row.addWidget(self.cal_intercept_entry)
        equation_row.addSpacing(18)
        equation_row.addWidget(QtWidgets.QLabel("Kβᵣₑₛ residual:"))
        equation_row.addWidget(self.cal_residual_entry)
        equation_row.addStretch(1)
        layout.addLayout(equation_row)
        layout.addStretch(1)
        return panel

    def _style_active_button(self, button):
        button.setStyleSheet(
            "QPushButton {"
            "font-weight: 600; color: #0b5d3b; background: #e2f2e6;"
            "border: 1px solid #7fae8c; border-radius: 4px; padding: 4px 10px;"
            "}"
            "QPushButton:hover { background: #d4eadb; }"
        )

    def _set_toggle_style(self, button, checked, on_text, off_text):
        button.setText(on_text if checked else off_text)
        if checked:
            self._style_active_button(button)
        else:
            button.setStyleSheet("")

    def _not_migrated(self, feature):
        QtWidgets.QMessageBox.information(
            self,
            feature,
            f"{feature} controls are restored in this Qt layout, but this action has not been migrated yet.",
        )

    def _normalize_imported_spectrum(self, y_data):
        y_data = np.asarray(y_data, dtype=float)
        if self.spect_processor is not None:
            try:
                return self.spect_processor.get_norm_spectrum(y_data)
            except Exception:
                pass
        return _normalize_spectrum(y_data)

    def _new_spectrum_legend(self):
        legend = pg.LegendItem(offset=(-12, 12), labelTextColor=(40, 40, 40), labelTextSize="12pt")
        legend.setParentItem(self.spectrum_plot.getViewBox())
        legend.anchor((1, 0), (1, 0), offset=(-12, 12))
        return legend

    def _ensure_spectrum_legend(self):
        if self._legend is None:
            self._legend = self._new_spectrum_legend()

    def _add_imported_spectrum_curve(self, x_data, y_data, label, color="#4daf4a", clear_if_empty=False):
        if clear_if_empty and self.last_x.size == 0:
            self._clear_spectrum()
        self._ensure_spectrum_legend()
        curve = pg.PlotDataItem(x_data, y_data, pen=pg.mkPen(color, width=1.3))
        self.spectrum_plot.addItem(curve)
        self._spectrum_items.append(curve)
        self._legend.addItem(curve, label)
        y_arrays = [y_data]
        x_values = np.asarray(x_data, dtype=float)
        if self.last_x.size:
            x_values = np.concatenate([self.last_x, x_values])
            y_arrays.append(self.last_y)
        self._set_spectrum_view_range(x_values, y_arrays)

    def _insert_average_reference_row(self, reference):
        if not hasattr(self, 'reference_average_table'):
            return
        row = self.reference_average_table.rowCount()
        self.reference_average_table.insertRow(row)
        use_item = QtWidgets.QTableWidgetItem("")
        use_item.setFlags(
            QtCore.Qt.ItemFlag.ItemIsUserCheckable
            | QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsSelectable
        )
        use_item.setCheckState(QtCore.Qt.CheckState.Checked)
        use_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.reference_average_table.setItem(row, 0, use_item)
        fit_error = reference.get('fit_error', np.nan)
        values = [
            reference.get('label', f"Ref {row + 1}"),
            f"{float(reference.get('main_position', np.nan)):.6g}",
            _format_fit_error(fit_error) or "--",
            "",
        ]
        for column, value in enumerate(values, start=1):
            item = QtWidgets.QTableWidgetItem(value)
            item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.reference_average_table.setItem(row, column, item)
        self.reference_average_table.setRowHeight(row, 20)

    def _checked_average_reference_indices(self):
        table = getattr(self, 'reference_average_table', None)
        if table is None:
            return list(range(len(self.reference_average_spectra)))
        indices = []
        for row in range(min(table.rowCount(), len(self.reference_average_spectra))):
            use_item = table.item(row, 0)
            if use_item is None or use_item.checkState() == QtCore.Qt.CheckState.Checked:
                indices.append(row)
        return indices

    def _set_average_reference_table_value(self, row, column, value):
        table = getattr(self, 'reference_average_table', None)
        if table is None or row >= table.rowCount():
            return
        item = table.item(row, column)
        if item is None:
            item = QtWidgets.QTableWidgetItem()
            item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            table.setItem(row, column, item)
        item.setText(value)

    def import_average_references(self):
        initial_dir = os.path.dirname(self._default_save_path("reference_pkfit.txt"))
        if not os.path.isdir(initial_dir):
            initial_dir = ""
        paths, _selected = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Import reference pkfit spectra",
            initial_dir,
            "Peak fit files (*.txt *.csv);;All files (*.*)",
        )
        if not paths:
            return

        palette = ["#4daf4a", "#377eb8", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#999999"]
        failures = []
        imported = 0
        for path in paths:
            try:
                x_data, y_data, main_position, fit_error = _read_pkfit_reference_profile(path)
            except Exception as exc:
                failures.append(f"{os.path.basename(path)}: {exc}")
                continue
            y_norm = _area_normalize_spectrum(x_data, y_data)
            reference = {
                "path": path,
                "x": np.asarray(x_data, dtype=float),
                "y": np.asarray(y_data, dtype=float),
                "y_norm": y_norm,
                "main_position": float(main_position),
                "fit_error": float(fit_error),
                "label": _run_label_from_path(path),
                "color": palette[len(self.reference_average_spectra) % len(palette)],
                "last_shift": np.nan,
            }
            self.reference_average_spectra.append(reference)
            self._insert_average_reference_row(reference)
            imported += 1

        self.last_reference_average = None
        if imported:
            self._draw_reference_average_inputs()
            self.reference_average_summary.setText(f"{len(self.reference_average_spectra)} reference pkfit spectra")
            self.statusBar().showMessage(f"Imported {imported} reference pkfit spectra")
        if failures:
            QtWidgets.QMessageBox.warning(self, "Import Reference", "\n".join(failures[:6]))

    def remove_average_references(self):
        table = getattr(self, 'reference_average_table', None)
        if table is None or not self.reference_average_spectra:
            return
        rows = sorted({index.row() for index in table.selectedIndexes()}, reverse=True)
        if not rows:
            current = table.currentRow()
            if current >= 0:
                rows = [current]
        if not rows:
            return
        for row in rows:
            if 0 <= row < len(self.reference_average_spectra):
                self.reference_average_spectra.pop(row)
                table.removeRow(row)
        self.last_reference_average = None
        self.reference_average_summary.setText(f"{len(self.reference_average_spectra)} reference pkfit spectra")
        self._draw_reference_average_inputs()

    def _draw_reference_average_inputs(self):
        self._clear_spectrum()
        self.spectrum_plot.setLabel('bottom', 'Column Index', **{'font-size': '13pt', 'font-family': 'Arial'})
        self.spectrum_plot.setLabel('left', 'Normalized intensity', **{'font-size': '13pt', 'font-family': 'Arial'})
        self._legend = self._new_spectrum_legend()
        x_values = []
        y_arrays = []
        for reference in self.reference_average_spectra:
            x_data = np.asarray(reference.get('x', []), dtype=float)
            y_data = np.asarray(reference.get('y_norm', reference.get('y', [])), dtype=float)
            if x_data.size < 2 or y_data.size != x_data.size:
                continue
            color = QtGui.QColor(reference.get('color', "#4daf4a"))
            color.setAlpha(185)
            line = pg.PlotDataItem(x_data, y_data, pen=pg.mkPen(color, width=1.2))
            self.spectrum_plot.addItem(line)
            self._spectrum_items.append(line)
            self._legend.addItem(line, reference.get('label', "Reference"))
            x_values.append(x_data)
            y_arrays.append(y_data)
        if x_values and y_arrays:
            self._set_spectrum_view_range(np.concatenate(x_values), y_arrays)
        self._set_spectrum_view_mode("reference")
        self.spectrum_title.setText("Reference Spectra")
        self.spectrum_status.setText("Imported pkfit spectra")

    def _reference_common_grid(self, selected):
        first = self.reference_average_spectra[selected[0]]
        x_data = np.asarray(first.get('x', []), dtype=float)
        finite = np.isfinite(x_data)
        x_data = x_data[finite]
        if x_data.size < 2:
            raise ValueError("The first checked reference does not contain a valid X grid.")
        return np.sort(x_data)

    def _aligned_reference_stack(self, selected):
        main_positions = np.asarray(
            [self.reference_average_spectra[index]['main_position'] for index in selected],
            dtype=float,
        )
        if main_positions.size == 0 or not np.all(np.isfinite(main_positions)):
            raise ValueError("All checked references must have a valid Kβ1,3 peak position.")
        target_main = float(np.nanmedian(main_positions))
        common_x = self._reference_common_grid(selected)
        stack = []
        metadata = []
        for index in selected:
            reference = self.reference_average_spectra[index]
            x_data = np.asarray(reference.get('x', []), dtype=float)
            y_data = np.asarray(reference.get('y_norm', reference.get('y', [])), dtype=float)
            finite = np.isfinite(x_data) & np.isfinite(y_data)
            x_data = x_data[finite]
            y_data = y_data[finite]
            if x_data.size < 2:
                continue
            shift = target_main - float(reference['main_position'])
            x_aligned = x_data + shift
            order = np.argsort(x_aligned)
            x_aligned = x_aligned[order]
            y_data = y_data[order]
            unique_x, unique_indices = np.unique(x_aligned, return_index=True)
            x_aligned = unique_x
            y_data = y_data[unique_indices]
            if x_aligned.size < 2:
                continue
            y_interp = np.interp(common_x, x_aligned, y_data)
            y_interp[(common_x < x_aligned[0]) | (common_x > x_aligned[-1])] = np.nan
            y_interp = _area_normalize_spectrum(common_x, y_interp)
            if np.count_nonzero(np.isfinite(y_interp)) < 2:
                continue
            stack.append(y_interp)
            reference['last_shift'] = float(shift)
            metadata.append(
                {
                    "index": index,
                    "label": reference.get("label", f"Ref {index + 1}"),
                    "path": reference.get("path", ""),
                    "main_position": float(reference.get("main_position", np.nan)),
                    "fit_error": float(reference.get("fit_error", np.nan)),
                    "shift": float(shift),
                }
            )
        if not stack:
            raise ValueError("No checked reference could be interpolated onto the common grid.")
        stack = np.vstack(stack)
        valid_counts = np.sum(np.isfinite(stack), axis=0)
        safe_stack = np.where(np.isfinite(stack), stack, 0.0)
        avg_raw = np.full(common_x.shape, np.nan, dtype=float)
        np.divide(np.sum(safe_stack, axis=0), valid_counts, out=avg_raw, where=valid_counts > 0)
        avg, avg_baseline, avg_area = _area_normalize_spectrum(common_x, avg_raw, return_transform=True)
        deviations = np.where(np.isfinite(stack), stack - avg_raw, 0.0)
        variance = np.full(common_x.shape, np.nan, dtype=float)
        np.divide(np.sum(deviations * deviations, axis=0), valid_counts, out=variance, where=valid_counts > 0)
        std = np.sqrt(variance)
        if np.isfinite(avg_area) and avg_area > np.finfo(float).eps:
            std = std / avg_area
        return common_x, stack, avg, std, valid_counts, target_main, metadata, avg_baseline, avg_area

    def average_reference_spectra(self):
        selected = self._checked_average_reference_indices()
        if not selected:
            QtWidgets.QMessageBox.information(self, "Average Reference", "Check at least one reference spectrum.")
            return
        try:
            common_x, stack, avg, std, valid_counts, target_main, metadata, avg_baseline, avg_area = self._aligned_reference_stack(selected)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Average Reference", str(exc))
            return
        input_fit_errors = [item.get("fit_error", np.nan) for item in metadata]
        average_fit_error = _average_reference_fit_error(input_fit_errors)
        scatter_error = _reference_mean_scatter_error(common_x, avg, std, valid_counts)
        total_reference_error = _quadrature(average_fit_error, scatter_error)
        self.last_reference_average = {
            "x": common_x,
            "stack": stack,
            "avg": avg,
            "std": std,
            "valid_counts": valid_counts,
            "target_main_position": target_main,
            "target_method": "median",
            "area_normalized": True,
            "average_baseline": avg_baseline,
            "average_area_before_normalization": avg_area,
            "average_reference_fit_error": average_fit_error,
            "reference_mean_scatter_error": scatter_error,
            "total_reference_error": total_reference_error,
            "references": metadata,
        }
        for reference_index, reference in enumerate(self.reference_average_spectra):
            shift = reference.get('last_shift', np.nan)
            self._set_average_reference_table_value(reference_index, 4, "" if not np.isfinite(shift) else f"{shift:.6g}")
        self._draw_reference_average_result()
        valid_pixels = int(np.count_nonzero(valid_counts > 0))
        self.reference_average_summary.setText(
            f"Averaged {len(metadata)} spectra; target Kβ₁,₃ = {target_main:.6g}; valid px = {valid_pixels}"
        )
        self.statusBar().showMessage(f"Averaged {len(metadata)} checked reference spectra")

    def _draw_reference_average_result(self):
        result = getattr(self, 'last_reference_average', None)
        if not result:
            self._draw_reference_average_inputs()
            return
        self._clear_spectrum()
        self.spectrum_plot.setLabel('bottom', 'Column Index', **{'font-size': '13pt', 'font-family': 'Arial'})
        self.spectrum_plot.setLabel('left', 'Normalized intensity', **{'font-size': '13pt', 'font-family': 'Arial'})
        self._legend = self._new_spectrum_legend()
        y_arrays = []
        x_arrays = []
        checked = {item['index'] for item in result.get('references', [])}
        for index, reference in enumerate(self.reference_average_spectra):
            if index not in checked:
                continue
            x_data = np.asarray(reference.get('x', []), dtype=float)
            y_data = np.asarray(reference.get('y_norm', reference.get('y', [])), dtype=float)
            color = QtGui.QColor(reference.get('color', "#4daf4a"))
            color.setAlpha(105)
            line = pg.PlotDataItem(x_data, y_data, pen=pg.mkPen(color, width=1.0))
            self.spectrum_plot.addItem(line)
            self._spectrum_items.append(line)
            self._legend.addItem(line, reference.get('label', f"Run {index + 1}"))
            x_arrays.append(x_data)
            y_arrays.append(y_data)

        common_x = np.asarray(result['x'], dtype=float)
        avg = np.asarray(result['avg'], dtype=float)
        std = np.asarray(result['std'], dtype=float)
        lower = avg - std
        upper = avg + std
        lower_curve = pg.PlotCurveItem(common_x, lower, pen=pg.mkPen(None))
        upper_curve = pg.PlotCurveItem(common_x, upper, pen=pg.mkPen(None))
        fill_color = QtGui.QColor("#202020")
        fill_color.setAlpha(35)
        std_fill = pg.FillBetweenItem(upper_curve, lower_curve, brush=pg.mkBrush(fill_color))
        std_fill.setZValue(-5)
        avg_line = pg.PlotDataItem(common_x, avg, pen=pg.mkPen("#202020", width=2.0))
        self.spectrum_plot.addItem(std_fill)
        self.spectrum_plot.addItem(lower_curve)
        self.spectrum_plot.addItem(upper_curve)
        self.spectrum_plot.addItem(avg_line)
        self._spectrum_items.extend([std_fill, lower_curve, upper_curve, avg_line])
        self._legend.addItem(avg_line, "Average Ref")
        x_arrays.append(common_x)
        y_arrays.extend([avg, lower, upper])
        self._set_spectrum_view_range(np.concatenate(x_arrays), y_arrays)
        self._set_spectrum_view_mode("reference")
        self.spectrum_title.setText("Reference Spectra")
        self.spectrum_status.setText("Average reference from checked pkfit spectra")

    def save_average_reference(self):
        if getattr(self, 'last_reference_average', None) is None:
            self.average_reference_spectra()
        result = getattr(self, 'last_reference_average', None)
        if not result:
            return
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export averaged reference pkfit",
            self._default_save_path("Average_reference_pkfit.txt"),
            "Text data (*.txt);;CSV data (*.csv);;All files (*.*)",
        )
        if not path:
            return
        delimiter = "," if path.lower().endswith(".csv") else "\t"
        avg = np.asarray(result['avg'], dtype=float)
        std = np.asarray(result['std'], dtype=float)
        valid_counts = np.asarray(result['valid_counts'], dtype=int)
        with open(path, mode="w", newline="") as handle:
            writer = csv.writer(handle, delimiter=delimiter)
            writer.writerow(["Peak fit model", "Averaged reference"])
            writer.writerow(["Reference average", True])
            writer.writerow(["Area normalized", bool(result.get("area_normalized", False))])
            writer.writerow(["Spectra number", len(result.get('references', []))])
            writer.writerow(["Target statistic", result.get("target_method", "median")])
            writer.writerow(["Target Kβ1,3 position", result.get("target_main_position", np.nan)])
            writer.writerow(["Average baseline before normalization", result.get("average_baseline", np.nan)])
            writer.writerow(["Average area before normalization", result.get("average_area_before_normalization", np.nan)])
            writer.writerow(["Average reference fit error", result.get("average_reference_fit_error", np.nan)])
            writer.writerow(["Reference mean scatter error", result.get("reference_mean_scatter_error", np.nan)])
            writer.writerow(["Total average reference error", result.get("total_reference_error", np.nan)])
            writer.writerow(["Input references"])
            writer.writerow(["Use", "Run", "Kβ1,3 position", "Fit error", "Shift", "Reference file"])
            for reference in result.get('references', []):
                writer.writerow(
                    [
                        True,
                        reference.get("label", ""),
                        reference.get("main_position", ""),
                        reference.get("fit_error", ""),
                        reference.get("shift", ""),
                        reference.get("path", ""),
                    ]
                )
            writer.writerow(["Peak parameters"])
            writer.writerow(["Peak", "Lorentzian fraction", "Position", "Width", "Area"])
            writer.writerow(["Kβ₁,₃", "", result.get("target_main_position", np.nan), "", ""])
            writer.writerow(["X", "Raw Y", "Peak fit", "Mean - 1 std", "Mean + 1 std", "Std", "Valid N"])
            for index, x_value in enumerate(result['x']):
                writer.writerow(
                    [
                        x_value,
                        avg[index],
                        avg[index],
                        avg[index] - std[index],
                        avg[index] + std[index],
                        std[index],
                        int(valid_counts[index]),
                    ]
                )
        self.statusBar().showMessage(f"Exported averaged reference pkfit: {path}")

    def _insert_iad_reference_row(self, reference):
        if not hasattr(self, 'iad_line_table'):
            return
        row = self.iad_line_table.rowCount()
        self.iad_line_table.insertRow(row)
        use_item = QtWidgets.QTableWidgetItem("")
        use_item.setFlags(
            QtCore.Qt.ItemFlag.ItemIsUserCheckable
            | QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsSelectable
        )
        use_item.setCheckState(QtCore.Qt.CheckState.Checked)
        use_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.iad_line_table.setItem(row, 0, use_item)

        values = [reference['label'], "", "", "", ""]
        for column, value in enumerate(values, start=1):
            item = QtWidgets.QTableWidgetItem(value)
            item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.iad_line_table.setItem(row, column, item)
        self.iad_line_table.setRowHeight(row, 20)

    def _checked_iad_reference_indices(self):
        table = getattr(self, 'iad_line_table', None)
        if table is None:
            return []
        indices = []
        for row in range(min(table.rowCount(), len(self.reference_spectra))):
            use_item = table.item(row, 0)
            if use_item is None or use_item.checkState() == QtCore.Qt.CheckState.Checked:
                indices.append(row)
        return indices

    def _set_iad_table_value(self, row, column, value):
        if not hasattr(self, 'iad_line_table') or row >= self.iad_line_table.rowCount():
            return
        item = self.iad_line_table.item(row, column)
        if item is None:
            item = QtWidgets.QTableWidgetItem()
            item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.iad_line_table.setItem(row, column, item)
        item.setText(value)

    def pick_iad_reference_color(self):
        table = getattr(self, 'iad_line_table', None)
        if table is None or not self.reference_spectra:
            QtWidgets.QMessageBox.information(self, "Reference Color", "Import a reference first.")
            return
        rows = sorted({index.row() for index in table.selectedIndexes()})
        if not rows:
            current = table.currentRow()
            if current >= 0:
                rows = [current]
        rows = [row for row in rows if 0 <= row < len(self.reference_spectra)]
        if not rows:
            QtWidgets.QMessageBox.information(self, "Reference Color", "Select a reference row first.")
            return
        initial = QtGui.QColor(self.reference_spectra[rows[0]].get('color', "#4daf4a"))
        color = QtWidgets.QColorDialog.getColor(initial, self, "Pick IAD reference color")
        if not color.isValid():
            return
        for row in rows:
            self.reference_spectra[row]['color'] = color.name()
        mode = getattr(self, 'current_spectrum_view', "")
        if mode == "iad":
            if hasattr(self, 'last_iad_plot'):
                self._draw_iad_comparison(*self.last_iad_plot)
        elif rows:
            reference = self.reference_spectra[rows[-1]]
            self._clear_spectrum()
            legend_label = reference['label'] if reference.get('is_average') else f"Ref, {reference['label']}"
            self._add_imported_spectrum_curve(
                reference['x'],
                reference['y'],
                legend_label,
                color=reference['color'],
                clear_if_empty=True,
            )
            self._update_spectrum_header("Reference")
        self.statusBar().showMessage("Updated IAD reference color")

    def _ensure_current_peak_fit(self):
        if hasattr(self, 'last_peak_fit_profile'):
            return True
        if self.last_x.size == 0:
            QtWidgets.QMessageBox.information(self, "IAD", "Plot a spectrum and run Peak Fit first.")
            return False
        self.show_peak_fit_profile()
        return hasattr(self, 'last_peak_fit_profile')

    def _current_peak_fit_xy(self):
        if not self._ensure_current_peak_fit():
            return None
        fit_result = self.last_peak_fit_profile
        x_data = np.asarray(fit_result.x, dtype=float)
        y_data = np.asarray(fit_result.normalized_fit, dtype=float)
        finite = np.isfinite(x_data) & np.isfinite(y_data)
        x_data = x_data[finite]
        y_data = y_data[finite]
        if x_data.size < 2:
            return None
        order = np.argsort(x_data)
        return x_data[order], y_data[order]

    def _current_peak_fit_error(self):
        if not hasattr(self, 'last_peak_fit_profile'):
            return np.nan
        value = getattr(self.last_peak_fit_profile, 'relative_fit_error', np.nan)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return np.nan
        return value if np.isfinite(value) else np.nan

    def _iad_mc_trial_count(self):
        if hasattr(self, 'iad_mc_trials_spin'):
            return int(self.iad_mc_trials_spin.value())
        return IAD_MC_DEFAULT_TRIALS

    def _clear_iad_mc_cache(self, *_args):
        if hasattr(self, 'last_iad_mc_sample'):
            del self.last_iad_mc_sample

    def _peak_fit_config_from_result(self, fit_result):
        try:
            return PeakFitConfig(
                n_peaks=max(1, len(getattr(fit_result, 'centers', []) or [])),
                peak_shape=getattr(fit_result, 'peak_shape', 'pseudo_voigt'),
                physical_fit=bool(getattr(fit_result, 'physical_fit', False)),
                tail_baseline=bool(getattr(fit_result, 'tail_baseline_enabled', False)),
                tail_fraction=float(getattr(fit_result, 'tail_fraction', 0.10)),
            )
        except PeakFitError:
            return None

    def _monte_carlo_trial_fits(self):
        if not hasattr(self, 'last_peak_fit_profile'):
            return [], {"trials": 0, "success": 0, "failures": 0}

        trial_count = self._iad_mc_trial_count()
        fit_result = self.last_peak_fit_profile
        cached = getattr(self, 'last_iad_mc_sample', None)
        if (
            cached
            and cached.get('source_id') == id(fit_result)
            and cached.get('trials') == trial_count
        ):
            return cached.get('fits', []), {
                "trials": cached.get('trials', 0),
                "success": cached.get('success', 0),
                "failures": cached.get('failures', 0),
            }

        config = self._peak_fit_config_from_result(fit_result)
        if config is None:
            return [], {"trials": trial_count, "success": 0, "failures": trial_count}

        x_data = np.asarray(fit_result.x, dtype=float)
        fitted_y = np.asarray(fit_result.normalized_fit, dtype=float)
        corrected_y = np.asarray(getattr(fit_result, 'fit_y', np.full_like(fitted_y, np.nan)), dtype=float)
        tail_baseline = np.asarray(getattr(fit_result, 'tail_baseline', np.zeros_like(fitted_y)), dtype=float)
        residual = corrected_y - fitted_y
        finite_residual = residual[np.isfinite(residual)]
        if finite_residual.size < 8:
            return [], {"trials": trial_count, "success": 0, "failures": trial_count}
        finite_residual = finite_residual - float(np.nanmean(finite_residual))

        rng = np.random.default_rng(IAD_MC_SEED)
        fits = []
        failures = 0
        for _trial_index in range(trial_count):
            sampled_residual = rng.choice(finite_residual, size=x_data.size, replace=True)
            trial_raw_y = fitted_y + tail_baseline + sampled_residual
            try:
                trial_fit = fit_spectrum(x_data, trial_raw_y, config)
            except PeakFitError:
                failures += 1
                continue
            trial_x = np.asarray(trial_fit.x, dtype=float)
            trial_y = np.asarray(trial_fit.normalized_fit, dtype=float)
            finite = np.isfinite(trial_x) & np.isfinite(trial_y)
            if np.count_nonzero(finite) < 2:
                failures += 1
                continue
            order = np.argsort(trial_x[finite])
            fits.append((trial_x[finite][order], trial_y[finite][order]))

        self.last_iad_mc_sample = {
            "source_id": id(fit_result),
            "trials": trial_count,
            "success": len(fits),
            "failures": failures,
            "fits": fits,
        }
        return fits, {"trials": trial_count, "success": len(fits), "failures": failures}

    def _reference_error_terms(self, reference):
        if reference.get('is_average'):
            fit_error = reference.get('average_reference_fit_error', np.nan)
            scatter_error = reference.get('reference_mean_scatter_error', np.nan)
            total_error = reference.get('total_reference_error', np.nan)
            try:
                total_error = float(total_error)
            except (TypeError, ValueError):
                total_error = np.nan
            if not np.isfinite(total_error):
                total_error = _quadrature(fit_error, scatter_error)
        else:
            fit_error = reference.get('fit_error', np.nan)
            scatter_error = np.nan
            total_error = fit_error
        return fit_error, scatter_error, total_error

    def _iad_uncertainty(self, reference):
        trial_fits, mc_meta = self._monte_carlo_trial_fits()
        iad_values = []
        for trial_x, trial_y in trial_fits:
            aligned = self._iad_arrays_from_fit(trial_x, trial_y, reference, warn=False)
            if aligned is None:
                continue
            x_data, roi_y, ref_y = aligned
            iad_values.append(_integrated_absolute_difference(x_data, roi_y, ref_y))
        iad_values = np.asarray(iad_values, dtype=float)
        iad_values = iad_values[np.isfinite(iad_values)]
        trial_count = int(mc_meta.get("trials", self._iad_mc_trial_count()))
        min_success = min(IAD_MC_MIN_SUCCESS, max(8, trial_count // 5))
        if iad_values.size >= min_success:
            sample_mc_error = float(np.nanstd(iad_values, ddof=1))
            mc_mean = float(np.nanmean(iad_values))
            mc_median = float(np.nanmedian(iad_values))
        else:
            sample_mc_error = np.nan
            mc_mean = np.nan
            mc_median = np.nan

        ref_fit_score, ref_scatter_error, ref_total_score = self._reference_error_terms(reference)
        total_error = _quadrature(sample_mc_error, ref_scatter_error)
        return {
            "iad_error": total_error,
            "sample_mc_error": sample_mc_error,
            "sample_residual_score": self._current_peak_fit_error(),
            "reference_fit_score": ref_fit_score,
            "reference_scatter_error": ref_scatter_error,
            "reference_total_score": ref_total_score,
            "mc_trials": trial_count,
            "mc_success": int(iad_values.size),
            "mc_failures": int(mc_meta.get("failures", 0)),
            "mc_mean": mc_mean,
            "mc_median": mc_median,
        }

    def _satellite_iad_uncertainty(self, reference):
        trial_fits, mc_meta = self._monte_carlo_trial_fits()
        satellite_values = []
        transition_fractions = []
        for trial_x, trial_y in trial_fits:
            aligned = self._iad_arrays_from_fit(trial_x, trial_y, reference, warn=False)
            prepared = self._satellite_iad_arrays_from_aligned(aligned, warn=False)
            if prepared is None:
                continue
            x_aligned, roi_aligned, ref_aligned, transition_point = prepared
            satellite_values.append(
                _integrated_absolute_difference(
                    x_aligned[:transition_point],
                    roi_aligned[:transition_point],
                    ref_aligned[:transition_point],
                )
            )
            if len(x_aligned) > 0:
                transition_fractions.append(float(transition_point) / float(len(x_aligned)))

        satellite_values = np.asarray(satellite_values, dtype=float)
        satellite_values = satellite_values[np.isfinite(satellite_values)]
        transition_fractions = np.asarray(transition_fractions, dtype=float)
        transition_fractions = transition_fractions[np.isfinite(transition_fractions)]
        trial_count = int(mc_meta.get("trials", self._iad_mc_trial_count()))
        min_success = min(IAD_MC_MIN_SUCCESS, max(8, trial_count // 5))
        if satellite_values.size >= min_success:
            sample_mc_error = float(np.nanstd(satellite_values, ddof=1))
            mc_mean = float(np.nanmean(satellite_values))
            mc_median = float(np.nanmedian(satellite_values))
        else:
            sample_mc_error = np.nan
            mc_mean = np.nan
            mc_median = np.nan

        if transition_fractions.size:
            satellite_fraction = float(np.nanmedian(transition_fractions))
        else:
            satellite_fraction = np.nan
        ref_satellite_scatter = _reference_fractional_scatter_error(reference, satellite_fraction)
        if not np.isfinite(ref_satellite_scatter):
            ref_satellite_scatter = reference.get('reference_mean_scatter_error', np.nan)
        total_error = _quadrature(sample_mc_error, ref_satellite_scatter)
        return {
            "satellite_iad_error": total_error,
            "satellite_sample_mc_error": sample_mc_error,
            "satellite_reference_scatter_error": ref_satellite_scatter,
            "satellite_fraction": satellite_fraction,
            "sample_residual_score": self._current_peak_fit_error(),
            "mc_trials": trial_count,
            "mc_success": int(satellite_values.size),
            "mc_failures": int(mc_meta.get("failures", 0)),
            "mc_mean": mc_mean,
            "mc_median": mc_median,
        }

    def _iad_baseline_alignment_enabled(self):
        return not hasattr(self, 'iad_baseline_check') or self.iad_baseline_check.isChecked()

    def _iad_arrays_from_fit(self, current_x, current_y, reference, warn=True):
        current_x = np.asarray(current_x, dtype=float)
        current_y = np.asarray(current_y, dtype=float)
        current_finite = np.isfinite(current_x) & np.isfinite(current_y)
        current_x = current_x[current_finite]
        current_y = current_y[current_finite]
        if current_x.size < 2:
            return None
        current_order = np.argsort(current_x)
        current_x = current_x[current_order]
        current_y = current_y[current_order]
        ref_x = np.asarray(reference['x'], dtype=float)
        ref_y = np.asarray(reference['y'], dtype=float)
        finite = np.isfinite(ref_x) & np.isfinite(ref_y)
        ref_x = ref_x[finite]
        ref_y = ref_y[finite]
        if ref_x.size < 2:
            return None
        order = np.argsort(ref_x)
        ref_x = ref_x[order]
        ref_y = ref_y[order]

        if ref_x.shape == current_x.shape and np.allclose(ref_x, current_x):
            roi_y = current_y.copy()
            if self._iad_baseline_alignment_enabled():
                roi_y, ref_y = _align_iad_baselines(roi_y, ref_y)
            normalized = _area_normalize_pair(ref_x, roi_y, ref_y)
            return normalized if normalized[0].size >= 2 else None

        min_x = max(float(np.nanmin(ref_x)), float(np.nanmin(current_x)))
        max_x = min(float(np.nanmax(ref_x)), float(np.nanmax(current_x)))
        mask = (ref_x >= min_x) & (ref_x <= max_x)
        if np.count_nonzero(mask) < 2:
            if warn:
                QtWidgets.QMessageBox.warning(
                    self,
                    "IAD",
                    f"{reference['label']} does not overlap the current peak-fit x range.",
                )
            return None
        common_x = ref_x[mask]
        common_ref = ref_y[mask]
        common_roi = np.interp(common_x, current_x, current_y)
        if self._iad_baseline_alignment_enabled():
            common_roi, common_ref = _align_iad_baselines(common_roi, common_ref)
        normalized = _area_normalize_pair(common_x, common_roi, common_ref)
        return normalized if normalized[0].size >= 2 else None

    def _aligned_iad_arrays(self, reference):
        current = self._current_peak_fit_xy()
        if current is None:
            return None
        current_x, current_y = current
        return self._iad_arrays_from_fit(current_x, current_y, reference)

    def _satellite_controls(self, spectrum_len, warn=True):
        try:
            cross_begin = int(self.cross_begin.text().strip())
            cross_end = int(self.cross_end.text().strip())
        except ValueError:
            if warn:
                QtWidgets.QMessageBox.warning(self, "Satellite IAD", "Spectra Crossing Range values must be integers.")
            return None

        max_index = max(int(spectrum_len) - 1, 0)
        cross_begin = max(0, min(cross_begin, max_index))
        cross_end = max(0, min(cross_end, max_index))
        if cross_end <= cross_begin:
            if warn:
                QtWidgets.QMessageBox.warning(self, "Satellite IAD", "Spectra Crossing Range end must be greater than the start.")
            return None

        eyeball_text = self.eye_ball_cross.text().strip()
        try:
            eyeball_point = int(eyeball_text) if eyeball_text else None
        except ValueError:
            eyeball_point = None
        if eyeball_point is not None:
            eyeball_point = max(0, min(eyeball_point, max_index))
        return cross_begin, cross_end, eyeball_point

    def _satellite_iad_arrays_from_aligned(self, aligned, warn=True):
        if aligned is None:
            return None
        _x_data, roi_y, ref_y = aligned
        try:
            roi_aligned, ref_aligned = _align_spectra_by_main_peak(roi_y, ref_y)
            ref_aligned, _tail_matched = _tail_area_match_to_target(ref_aligned, roi_aligned)
            if self._iad_baseline_alignment_enabled():
                roi_aligned, ref_aligned = _align_iad_baselines(roi_aligned, ref_aligned)
            x_aligned = np.arange(len(roi_aligned), dtype=float)
            x_aligned, roi_aligned, ref_aligned = _area_normalize_pair(x_aligned, roi_aligned, ref_aligned)
        except ValueError as exc:
            if warn:
                QtWidgets.QMessageBox.warning(self, "Satellite IAD", str(exc))
            return None
        controls = self._satellite_controls(len(roi_aligned), warn=warn)
        if controls is None:
            return None
        cross_begin, cross_end, eyeball_point = controls
        transition_point = _find_satellite_transition(
            roi_aligned,
            ref_aligned,
            cross_begin,
            cross_end,
            eyeball_point,
        )
        if transition_point is None:
            transition_point = min(len(roi_aligned), len(ref_aligned))
        transition_point = max(1, min(int(transition_point), min(len(roi_aligned), len(ref_aligned))))
        return x_aligned, roi_aligned, ref_aligned, transition_point

    def _draw_iad_comparison(self, x_data, roi_y, ref_y, reference, diff_label, view_label, shade_end_index=None):
        self._clear_spectrum()
        self.spectrum_plot.setLabel('bottom', 'Column Index', **{'font-size': '13pt', 'font-family': 'Arial'})
        self.spectrum_plot.setLabel('left', 'Normalized intensity', **{'font-size': '13pt', 'font-family': 'Arial'})
        self._legend = self._new_spectrum_legend()

        x_data = np.asarray(x_data, dtype=float)
        roi_y = np.asarray(roi_y, dtype=float)
        ref_y = np.asarray(ref_y, dtype=float)
        diff_y = roi_y - ref_y
        zero_y = np.zeros_like(diff_y)
        ref_color = reference.get('color', "#4daf4a")
        fill_color = QtGui.QColor(ref_color)
        fill_color.setAlpha(70)

        roi_line = pg.PlotDataItem(x_data, roi_y, pen=self._selected_line_pen())
        ref_line = pg.PlotDataItem(x_data, ref_y, pen=pg.mkPen(ref_color, width=1.3))
        zero_line = pg.PlotCurveItem(
            x_data,
            zero_y,
            pen=pg.mkPen("#666666", width=0.8, style=QtCore.Qt.PenStyle.DashLine),
        )
        diff_line = pg.PlotCurveItem(x_data, diff_y, pen=pg.mkPen(ref_color, width=1.0))

        shade_x = x_data
        shade_diff = diff_y
        if shade_end_index is not None:
            end_index = max(1, min(int(shade_end_index), diff_y.size))
            shade_x = x_data[:end_index]
            shade_diff = diff_y[:end_index]
        shade_zero = np.zeros_like(shade_diff)
        shade_curve = pg.PlotCurveItem(shade_x, shade_diff, pen=pg.mkPen(None))
        shade_baseline = pg.PlotCurveItem(shade_x, shade_zero, pen=pg.mkPen(None))
        diff_fill = pg.FillBetweenItem(shade_curve, shade_baseline, brush=pg.mkBrush(fill_color))
        diff_fill.setZValue(-5)

        self.spectrum_plot.addItem(diff_fill)
        self.spectrum_plot.addItem(shade_curve)
        self.spectrum_plot.addItem(shade_baseline)
        self.spectrum_plot.addItem(ref_line)
        self.spectrum_plot.addItem(roi_line)
        self.spectrum_plot.addItem(zero_line)
        self.spectrum_plot.addItem(diff_line)
        self._spectrum_items.extend([diff_fill, shade_curve, shade_baseline, ref_line, roi_line, zero_line, diff_line])
        self._legend.addItem(ref_line, f"Ref, {reference['label']}")
        self._legend.addItem(roi_line, f"ROI, {self._run_display_text()}")
        self._legend.addItem(diff_line, diff_label)
        self._set_spectrum_view_range(x_data, [roi_y, ref_y, diff_y, zero_y])
        self._set_spectrum_view_mode("iad")
        self._update_spectrum_header(view_label)
        self.last_iad_plot = (x_data, roi_y, ref_y, reference, diff_label, view_label, shade_end_index)

    def import_reference_spectrum(self):
        path, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import pkfit reference",
            "",
            "Peak fit files (*.txt *.csv);;All files (*.*)",
        )
        if not path:
            return
        try:
            x_data, y_data, reference_metadata = _read_pkfit_iad_reference(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Import Ref.", str(exc))
            return

        basename = os.path.basename(path)
        is_average_reference = bool(reference_metadata.get("is_average", False))
        run_label = "Ref. average" if is_average_reference else _run_label_from_path(path)
        palette = ["#4daf4a", "#377eb8", "#984ea3", "#ff7f00", "#a65628", "#f781bf"]
        reference = {
            'path': path,
            'x': x_data,
            'y': y_data,
            'label': run_label,
            'is_average': is_average_reference,
            'fit_error': reference_metadata.get("fit_error", np.nan),
            'average_reference_fit_error': reference_metadata.get("average_reference_fit_error", np.nan),
            'reference_mean_scatter_error': reference_metadata.get("reference_mean_scatter_error", np.nan),
            'total_reference_error': reference_metadata.get("total_reference_error", np.nan),
            'reference_scatter_x': reference_metadata.get("reference_scatter_x", np.array([], dtype=float)),
            'reference_scatter_mean': reference_metadata.get("reference_scatter_mean", np.array([], dtype=float)),
            'reference_scatter_std': reference_metadata.get("reference_scatter_std", np.array([], dtype=float)),
            'reference_scatter_valid_n': reference_metadata.get("reference_scatter_valid_n", np.array([], dtype=float)),
            'color': palette[len(self.reference_spectra) % len(palette)],
            'iad': None,
            'iad_error': None,
            'iad_error_method': "",
            'iad_sample_mc_error': None,
            'iad_sample_residual_score': None,
            'iad_reference_fit_score': None,
            'iad_reference_scatter_error': None,
            'iad_reference_total_score': None,
            'iad_mc_trials': None,
            'iad_mc_success': None,
            'iad_mc_failures': None,
            'iad_mc_mean': None,
            'iad_mc_median': None,
            'satellite_iad': None,
            'satellite_iad_error': None,
            'satellite_iad_error_method': "",
            'satellite_sample_mc_error': None,
            'satellite_reference_scatter_error': None,
            'satellite_fraction': None,
            'satellite_mc_trials': None,
            'satellite_mc_success': None,
            'satellite_mc_failures': None,
            'satellite_mc_mean': None,
            'satellite_mc_median': None,
            'satellite_cross_point': None,
        }
        self.reference_spectra.append(reference)
        self._insert_iad_reference_row(reference)
        legend_label = run_label if is_average_reference else f"Ref, {run_label}"
        self._add_imported_spectrum_curve(x_data, y_data, legend_label, color=reference['color'], clear_if_empty=True)
        self._update_spectrum_header("Reference")
        self.statusBar().showMessage(f"Imported pkfit reference: {basename}")

    def remove_iad_reference(self):
        if not hasattr(self, 'iad_line_table') or not self.reference_spectra:
            return
        rows = sorted({index.row() for index in self.iad_line_table.selectedIndexes()}, reverse=True)
        if not rows:
            current = self.iad_line_table.currentRow()
            if current >= 0:
                rows = [current]
        if not rows:
            return
        for row in rows:
            if 0 <= row < len(self.reference_spectra):
                self.reference_spectra.pop(row)
                self.iad_line_table.removeRow(row)
        self.statusBar().showMessage("Removed selected IAD reference")

    def plot_integrated_diff(self):
        indices = self._checked_iad_reference_indices()
        if not indices:
            QtWidgets.QMessageBox.information(self, "IAD", "Select or import a pkfit reference first.")
            return
        last_plot = None
        for row in indices:
            reference = self.reference_spectra[row]
            aligned = self._aligned_iad_arrays(reference)
            if aligned is None:
                continue
            x_data, roi_y, ref_y = aligned
            iad_value = _integrated_absolute_difference(x_data, roi_y, ref_y)
            reference['iad'] = iad_value
            reference['iad_error'] = None
            reference['iad_error_method'] = ""
            reference['iad_sample_mc_error'] = None
            reference['iad_sample_residual_score'] = None
            reference['iad_reference_fit_score'] = None
            reference['iad_reference_scatter_error'] = None
            reference['iad_reference_total_score'] = None
            reference['iad_mc_trials'] = None
            reference['iad_mc_success'] = None
            reference['iad_mc_failures'] = None
            reference['iad_mc_mean'] = None
            reference['iad_mc_median'] = None
            self._set_iad_table_value(row, 2, f"{iad_value:.6g}")
            self._set_iad_table_value(row, 3, "")
            last_plot = (x_data, roi_y, ref_y, reference)
        if last_plot is None:
            return
        self._draw_iad_comparison(*last_plot, diff_label="IAD", view_label="IAD difference")
        self.statusBar().showMessage(f"Calculated IAD for {len(indices)} reference(s); click IAD Error for MC uncertainty")

    def calculate_iad_error(self):
        indices = self._checked_iad_reference_indices()
        if not indices:
            QtWidgets.QMessageBox.information(self, "IAD Error", "Select or import a pkfit reference first.")
            return
        last_plot = None
        trial_count = self._iad_mc_trial_count()
        self.statusBar().showMessage(f"Calculating MC IAD uncertainty ({trial_count} trials)...")
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.setOverrideCursor(_wait_cursor())
        try:
            for row in indices:
                reference = self.reference_spectra[row]
                aligned = self._aligned_iad_arrays(reference)
                if aligned is None:
                    continue
                x_data, roi_y, ref_y = aligned
                iad_value = _integrated_absolute_difference(x_data, roi_y, ref_y)
                iad_uncertainty = self._iad_uncertainty(reference)
                iad_error = float(iad_uncertainty.get("iad_error", np.nan))
                reference['iad'] = iad_value
                reference['iad_error'] = None if not np.isfinite(iad_error) else iad_error
                reference['iad_error_method'] = "MC refit + reference scatter"
                reference['iad_sample_mc_error'] = iad_uncertainty.get("sample_mc_error")
                reference['iad_sample_residual_score'] = iad_uncertainty.get("sample_residual_score")
                reference['iad_reference_fit_score'] = iad_uncertainty.get("reference_fit_score")
                reference['iad_reference_scatter_error'] = iad_uncertainty.get("reference_scatter_error")
                reference['iad_reference_total_score'] = iad_uncertainty.get("reference_total_score")
                reference['iad_mc_trials'] = iad_uncertainty.get("mc_trials")
                reference['iad_mc_success'] = iad_uncertainty.get("mc_success")
                reference['iad_mc_failures'] = iad_uncertainty.get("mc_failures")
                reference['iad_mc_mean'] = iad_uncertainty.get("mc_mean")
                reference['iad_mc_median'] = iad_uncertainty.get("mc_median")
                self._set_iad_table_value(row, 2, f"{iad_value:.6g}")
                self._set_iad_table_value(row, 3, "" if not np.isfinite(iad_error) else f"{iad_error:.6g}")
                last_plot = (x_data, roi_y, ref_y, reference)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if last_plot is None:
            return
        self._draw_iad_comparison(*last_plot, diff_label="IAD", view_label="IAD difference")
        self.statusBar().showMessage(f"Calculated MC IAD uncertainty for {len(indices)} reference(s)")

    def calculate_satellite_iad(self):
        indices = self._checked_iad_reference_indices()
        if not indices:
            QtWidgets.QMessageBox.information(self, "Satellite IAD", "Select or import a pkfit reference first.")
            return
        last_plot = None
        for row in indices:
            reference = self.reference_spectra[row]
            aligned = self._aligned_iad_arrays(reference)
            prepared = self._satellite_iad_arrays_from_aligned(aligned)
            if prepared is None:
                continue
            x_aligned, roi_aligned, ref_aligned, transition_point = prepared
            satellite_value = _integrated_absolute_difference(
                x_aligned[:transition_point],
                roi_aligned[:transition_point],
                ref_aligned[:transition_point],
            )
            reference['satellite_iad'] = satellite_value
            reference['satellite_iad_error'] = None
            reference['satellite_iad_error_method'] = ""
            reference['satellite_sample_mc_error'] = None
            reference['satellite_reference_scatter_error'] = None
            reference['satellite_mc_trials'] = None
            reference['satellite_mc_success'] = None
            reference['satellite_mc_failures'] = None
            reference['satellite_mc_mean'] = None
            reference['satellite_mc_median'] = None
            reference['satellite_fraction'] = None
            reference['satellite_cross_point'] = transition_point
            self._set_iad_table_value(row, 4, f"{satellite_value:.6g}")
            self._set_iad_table_value(row, 5, "")
            last_plot = (
                x_aligned,
                roi_aligned,
                ref_aligned,
                reference,
            )
        if last_plot is None:
            return
        self._draw_iad_comparison(
            *last_plot,
            diff_label="Satellite IAD",
            view_label="Satellite IAD",
            shade_end_index=last_plot[3].get('satellite_cross_point'),
        )
        self.statusBar().showMessage(f"Calculated satellite IAD for {len(indices)} reference(s)")

    def calculate_satellite_iad_error(self):
        indices = self._checked_iad_reference_indices()
        if not indices:
            QtWidgets.QMessageBox.information(self, "Satellite IAD error", "Select or import a pkfit reference first.")
            return
        last_plot = None
        trial_count = self._iad_mc_trial_count()
        self.statusBar().showMessage(f"Calculating MC satellite IAD uncertainty ({trial_count} trials)...")
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.setOverrideCursor(_wait_cursor())
        try:
            for row in indices:
                reference = self.reference_spectra[row]
                aligned = self._aligned_iad_arrays(reference)
                prepared = self._satellite_iad_arrays_from_aligned(aligned)
                if prepared is None:
                    continue
                x_aligned, roi_aligned, ref_aligned, transition_point = prepared
                satellite_value = _integrated_absolute_difference(
                    x_aligned[:transition_point],
                    roi_aligned[:transition_point],
                    ref_aligned[:transition_point],
                )
                satellite_uncertainty = self._satellite_iad_uncertainty(reference)
                satellite_error = float(satellite_uncertainty.get("satellite_iad_error", np.nan))
                reference['satellite_iad'] = satellite_value
                reference['satellite_iad_error'] = None if not np.isfinite(satellite_error) else satellite_error
                reference['satellite_iad_error_method'] = "MC refit + satellite reference scatter"
                reference['satellite_sample_mc_error'] = satellite_uncertainty.get("satellite_sample_mc_error")
                reference['satellite_reference_scatter_error'] = satellite_uncertainty.get("satellite_reference_scatter_error")
                reference['satellite_fraction'] = satellite_uncertainty.get("satellite_fraction")
                reference['satellite_mc_trials'] = satellite_uncertainty.get("mc_trials")
                reference['satellite_mc_success'] = satellite_uncertainty.get("mc_success")
                reference['satellite_mc_failures'] = satellite_uncertainty.get("mc_failures")
                reference['satellite_mc_mean'] = satellite_uncertainty.get("mc_mean")
                reference['satellite_mc_median'] = satellite_uncertainty.get("mc_median")
                reference['satellite_cross_point'] = transition_point
                self._set_iad_table_value(row, 4, f"{satellite_value:.6g}")
                self._set_iad_table_value(row, 5, "" if not np.isfinite(satellite_error) else f"{satellite_error:.6g}")
                last_plot = (
                    x_aligned,
                    roi_aligned,
                    ref_aligned,
                    reference,
                )
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if last_plot is None:
            return
        self._draw_iad_comparison(
            *last_plot,
            diff_label="Satellite IAD",
            view_label="Satellite IAD",
            shade_end_index=last_plot[3].get('satellite_cross_point'),
        )
        self.statusBar().showMessage(f"Calculated MC satellite IAD uncertainty for {len(indices)} reference(s)")

    def save_iad_results(self):
        if not self.reference_spectra:
            QtWidgets.QMessageBox.information(self, "Export IAD Values", "Import a pkfit reference first.")
            return
        indices = self._checked_iad_reference_indices()
        if not indices:
            QtWidgets.QMessageBox.information(self, "Export IAD Values", "Select at least one reference row before exporting.")
            return
        references = [self.reference_spectra[index] for index in indices]
        if all(reference.get('iad') is None and reference.get('satellite_iad') is None for reference in references):
            QtWidgets.QMessageBox.information(self, "Export IAD Values", "Calculate IAD before exporting.")
            return
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export IAD values",
            self._default_save_path(f"{self._run_basename()}_iad.txt"),
            "Text data (*.txt);;CSV data (*.csv);;All files (*.*)",
        )
        if not path:
            return
        delimiter = "," if path.lower().endswith(".csv") else "\t"
        with open(path, mode="w", newline="") as handle:
            writer = csv.writer(handle, delimiter=delimiter)
            writer.writerow(["Current run", self._run_display_text()])
            writer.writerow(["Baseline aligned", self._iad_baseline_alignment_enabled()])
            writer.writerow(["Area normalized", True])
            writer.writerow(["Spectra Crossing Range", self.cross_begin.text().strip(), self.cross_end.text().strip()])
            writer.writerow(["Manual crossing", self.eye_ball_cross.text().strip()])
            writer.writerow(["IAD error method", "MC refit sample uncertainty + reference scatter"])
            writer.writerow(["Satellite IAD error method", "MC refit with recalculated crossing + satellite reference scatter"])
            writer.writerow(["IAD Monte Carlo trials setting", self._iad_mc_trial_count()])
            writer.writerow(
                [
                    "Reference run",
                    "IAD",
                    "IAD error",
                    "Sample MC error",
                    "Sample residual score",
                    "Reference scatter error",
                    "Reference fit score",
                    "Reference total score",
                    "Monte Carlo trials",
                    "MC success",
                    "MC mean IAD",
                    "MC median IAD",
                    "Satellite IAD",
                    "Satellite IAD error",
                    "Satellite sample MC error",
                    "Satellite reference scatter error",
                    "Satellite fraction",
                    "Satellite Monte Carlo trials",
                    "Satellite MC success",
                    "Satellite MC mean IAD",
                    "Satellite MC median IAD",
                    "Satellite cross point",
                    "Reference file",
                ]
            )
            for reference in references:
                iad_value = _format_optional_float(reference.get('iad'))
                iad_error = _format_optional_float(reference.get('iad_error'))
                sample_mc_error = _format_optional_float(reference.get('iad_sample_mc_error'))
                sample_residual_score = _format_optional_float(reference.get('iad_sample_residual_score'))
                ref_scatter_error = _format_optional_float(reference.get('iad_reference_scatter_error'))
                ref_fit_score = _format_optional_float(reference.get('iad_reference_fit_score'))
                ref_total_score = _format_optional_float(reference.get('iad_reference_total_score'))
                try:
                    mc_trials = str(int(reference.get('iad_mc_trials')))
                except (TypeError, ValueError):
                    mc_trials = ""
                try:
                    mc_success = str(int(reference.get('iad_mc_success')))
                except (TypeError, ValueError):
                    mc_success = ""
                mc_mean = _format_optional_float(reference.get('iad_mc_mean'))
                mc_median = _format_optional_float(reference.get('iad_mc_median'))
                satellite_value = _format_optional_float(reference.get('satellite_iad'))
                satellite_error = _format_optional_float(reference.get('satellite_iad_error'))
                satellite_sample_mc_error = _format_optional_float(reference.get('satellite_sample_mc_error'))
                satellite_reference_scatter_error = _format_optional_float(reference.get('satellite_reference_scatter_error'))
                satellite_fraction = _format_optional_float(reference.get('satellite_fraction'))
                try:
                    satellite_mc_trials = str(int(reference.get('satellite_mc_trials')))
                except (TypeError, ValueError):
                    satellite_mc_trials = ""
                try:
                    satellite_mc_success = str(int(reference.get('satellite_mc_success')))
                except (TypeError, ValueError):
                    satellite_mc_success = ""
                satellite_mc_mean = _format_optional_float(reference.get('satellite_mc_mean'))
                satellite_mc_median = _format_optional_float(reference.get('satellite_mc_median'))
                cross_point = "" if reference.get('satellite_cross_point') is None else reference['satellite_cross_point']
                writer.writerow(
                    [
                        reference['label'],
                        iad_value,
                        iad_error,
                        sample_mc_error,
                        sample_residual_score,
                        ref_scatter_error,
                        ref_fit_score,
                        ref_total_score,
                        mc_trials,
                        mc_success,
                        mc_mean,
                        mc_median,
                        satellite_value,
                        satellite_error,
                        satellite_sample_mc_error,
                        satellite_reference_scatter_error,
                        satellite_fraction,
                        satellite_mc_trials,
                        satellite_mc_success,
                        satellite_mc_mean,
                        satellite_mc_median,
                        cross_point,
                        reference['path'],
                    ]
                )
        self.statusBar().showMessage(f"Saved IAD results: {path}")

    def _calibration_fit_config(self):
        try:
            x_min_text = self.cal_fit_start.text().strip()
            x_max_text = self.cal_fit_end.text().strip()
            x_min = float(x_min_text) if x_min_text else None
            x_max = float(x_max_text) if x_max_text else None
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Calibration", "Calibration fit range must be numeric.")
            return None
        if x_min is not None and x_max is not None and x_min > x_max:
            x_min, x_max = x_max, x_min
        try:
            return PeakFitConfig(
                n_peaks=3,
                x_min=x_min,
                x_max=x_max,
                peak_shape=self.selected_peak_fit_model(),
                physical_fit=self.physical_fit_enabled(),
                tail_baseline=self.tail_baseline_enabled(),
            )
        except PeakFitError as exc:
            QtWidgets.QMessageBox.warning(self, "Calibration", str(exc))
            return None

    def _calibration_centers3(self, fit_result):
        centers = np.asarray(getattr(fit_result, 'centers', []), dtype=float)
        if centers.size < 3 or not np.all(np.isfinite(centers[:3])):
            raise ValueError("The 3-peak fit did not return valid Kβ′, Kβᵣₑₛ, and Kβ₁,₃ centers.")
        return float(centers[0]), float(centers[1]), float(centers[2])

    def _set_calibration_output(self, label, value, precision=6):
        entry = self.calibration_outputs.get(label) if hasattr(self, 'calibration_outputs') else None
        if entry is None:
            return
        entry.setText("" if value is None or not np.isfinite(value) else f"{float(value):.{precision}g}")

    def _set_calibration_scalar_output(self, entry, value, precision=8):
        entry.setText("" if value is None or not np.isfinite(value) else f"{float(value):.{precision}g}")

    def _update_calibration_center_outputs(self, prefix, centers):
        sat, res, main = centers
        if prefix == "cal":
            labels = ("Cal E Kβ′", "Cal E Kβᵣₑₛ", "Cal E Kβ₁,₃")
        else:
            labels = ("Current px Kβ′", "Current px Kβᵣₑₛ", "Current px Kβ₁,₃")
        for label, value in zip(labels, (sat, res, main)):
            self._set_calibration_output(label, value)

    def _clear_calibration_outputs(self, clear_current=False):
        labels = ["Cal E Kβ′", "Cal E Kβᵣₑₛ", "Cal E Kβ₁,₃"]
        if clear_current:
            labels.extend(["Current px Kβ′", "Current px Kβᵣₑₛ", "Current px Kβ₁,₃"])
        for label in labels:
            entry = self.calibration_outputs.get(label) if hasattr(self, 'calibration_outputs') else None
            if entry is not None:
                entry.clear()
        self.cal_slope_entry.clear()
        self.cal_intercept_entry.clear()
        self.cal_residual_entry.clear()

    def import_calibration_spectrum(self):
        path, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import calibration spectrum",
            "",
            "Spectrum files (*.txt *.chi *.csv);;All files (*.*)",
        )
        if not path:
            return
        try:
            x_data, y_data, source_column = _read_calibration_spectrum(path)
            y_display = self._normalize_imported_spectrum(y_data)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Import Cal.", str(exc))
            return

        basename = os.path.basename(path)
        self.calibration_filepath = path
        self.calibration_x = x_data
        self.calibration_y = y_data
        self.calibration_source_column = source_column
        self.calibration_file_label.setText(f"{basename} ({source_column})")
        for attr_name in ("calibration_fit", "energy_calibration"):
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        self._clear_calibration_outputs(clear_current=False)
        self._clear_spectrum()
        self.last_x = np.array([], dtype=float)
        self.last_y = np.array([], dtype=float)
        self.last_raw_x = np.array([], dtype=float)
        self.last_raw_y = np.array([], dtype=float)
        self._add_imported_spectrum_curve(x_data, y_display, f"Cal, {basename}", color="#8db7ff", clear_if_empty=False)
        self._update_spectrum_header("Calibration")
        self.statusBar().showMessage(f"Imported calibration spectrum: {basename}")

    def fit_calibration_spectrum(self):
        if not hasattr(self, 'calibration_x') or not hasattr(self, 'calibration_y'):
            QtWidgets.QMessageBox.warning(self, "Calibration", "Import a calibration spectrum first.")
            return None
        config = self._calibration_fit_config()
        if config is None:
            return None
        try:
            fit_result = fit_spectrum(self.calibration_x, self.calibration_y, config)
        except PeakFitError as exc:
            QtWidgets.QMessageBox.warning(self, "Calibration", str(exc))
            return None
        self.calibration_fit = fit_result
        try:
            centers = self._calibration_centers3(fit_result)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Calibration", str(exc))
            return None
        self._update_calibration_center_outputs("cal", centers)
        self.statusBar().showMessage("Calibration spectrum fitted")
        return fit_result

    def calculate_energy_calibration(self):
        if not self._ensure_current_peak_fit():
            return None
        cal_fit = getattr(self, 'calibration_fit', None)
        if cal_fit is None:
            cal_fit = self.fit_calibration_spectrum()
        if cal_fit is None:
            return None
        try:
            pix_sat, pix_res, pix_main = self._calibration_centers3(self.last_peak_fit_profile)
            ene_sat, ene_res, ene_main = self._calibration_centers3(cal_fit)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Calibration", str(exc))
            return None

        denom = pix_main - pix_sat
        if abs(denom) <= np.finfo(float).eps:
            QtWidgets.QMessageBox.warning(self, "Calibration", "Current Kβ′ and Kβ₁,₃ pixel centers are identical.")
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
        self._update_calibration_center_outputs("current", (pix_sat, pix_res, pix_main))
        self._update_calibration_center_outputs("cal", (ene_sat, ene_res, ene_main))
        self._set_calibration_scalar_output(self.cal_slope_entry, slope, precision=8)
        self._set_calibration_scalar_output(self.cal_intercept_entry, intercept, precision=8)
        self._set_calibration_scalar_output(self.cal_residual_entry, residual, precision=5)
        self.statusBar().showMessage("Energy calibration calculated")
        return self.energy_calibration

    def show_calibration_fit(self):
        fit_result = getattr(self, 'calibration_fit', None)
        if fit_result is None:
            fit_result = self.fit_calibration_spectrum()
        if fit_result is None:
            return

        self._clear_spectrum()
        self._legend = self._new_spectrum_legend()

        cal_line = pg.PlotDataItem(self.calibration_x, self.calibration_y, pen=pg.mkPen("#9a9a9a", width=1.0))
        fit_line = pg.PlotDataItem(fit_result.x, fit_result.normalized_fit, pen=pg.mkPen("#202020", width=1.4))
        self.spectrum_plot.addItem(cal_line)
        self.spectrum_plot.addItem(fit_line)
        self._spectrum_items.extend([cal_line, fit_line])
        self._legend.addItem(cal_line, "Calibrant")
        self._legend.addItem(fit_line, "Cal. fit")

        scale = _component_scale(fit_result)
        component_colors = ['#2b8cbe', '#e34a33', '#31a354', '#756bb1', '#fdae6b', '#636363']
        y_arrays = [self.calibration_y, fit_result.normalized_fit]
        if scale is not None:
            for index, peak_curve in enumerate(fit_result.peak_curves):
                component = np.asarray(peak_curve, dtype=float) / scale
                label = fit_result.peak_labels[index] if index < len(fit_result.peak_labels) else f"Peak {index + 1}"
                component_line = pg.PlotDataItem(
                    fit_result.x,
                    component,
                    pen=pg.mkPen(component_colors[index % len(component_colors)], width=1.0, style=QtCore.Qt.PenStyle.DashLine),
                )
                self.spectrum_plot.addItem(component_line)
                self._spectrum_items.append(component_line)
                self._legend.addItem(component_line, _display_peak_label(label))
                y_arrays.append(component)

        self.spectrum_plot.setLabel('bottom', 'Energy', **{'font-size': '13pt', 'font-family': 'Arial'})
        self.spectrum_plot.setLabel('left', 'Intensity', **{'font-size': '13pt', 'font-family': 'Arial'})
        self._set_spectrum_view_range(self.calibration_x, y_arrays)
        self._set_spectrum_view_mode("calibration")
        self._update_spectrum_header("Calibration fit")

    def apply_energy_calibration(self):
        calibration = getattr(self, 'energy_calibration', None)
        if calibration is None:
            calibration = self.calculate_energy_calibration()
        if calibration is None:
            return
        if self.last_x.size == 0 or self.last_y.size == 0:
            QtWidgets.QMessageBox.warning(self, "Calibration", "Plot the current spectrum first.")
            return

        energy = _energy_axis(calibration, self.last_x)
        self.calibration_energy_axis = energy
        self.calibrated_spectrum_y = self.last_y.copy()
        self._clear_spectrum()
        self._legend = self._new_spectrum_legend()

        current_line = pg.PlotDataItem(energy, self.last_y, pen=pg.mkPen("#777777", width=1.0))
        self.spectrum_plot.addItem(current_line)
        self._spectrum_items.append(current_line)
        self._legend.addItem(current_line, f"Current, {self._run_display_text()}")
        x_values = energy
        y_arrays = [self.last_y]

        fit_result = getattr(self, 'last_peak_fit_profile', None)
        if fit_result is not None:
            fit_energy = _energy_axis(calibration, fit_result.x)
            self.calibrated_fit_energy = fit_energy
            self.calibrated_fit_y = np.asarray(fit_result.normalized_fit, dtype=float).copy()
            fit_line = pg.PlotDataItem(fit_energy, fit_result.normalized_fit, pen=self._selected_line_pen())
            self.spectrum_plot.addItem(fit_line)
            self._spectrum_items.append(fit_line)
            self._legend.addItem(fit_line, "Peak fit")
            x_values = np.concatenate([x_values, fit_energy])
            y_arrays.append(fit_result.normalized_fit)

            scale = _component_scale(fit_result)
            if scale is not None:
                self.calibrated_component_labels = list(getattr(fit_result, "peak_labels", []))
                self.calibrated_components = []
                component_colors = ['#2b8cbe', '#e34a33', '#31a354', '#756bb1', '#fdae6b', '#636363']
                for index, peak_curve in enumerate(fit_result.peak_curves):
                    component = np.asarray(peak_curve, dtype=float) / scale
                    self.calibrated_components.append(component)
                    label = fit_result.peak_labels[index] if index < len(fit_result.peak_labels) else f"Peak {index + 1}"
                    component_line = pg.PlotDataItem(
                        fit_energy,
                        component,
                        pen=pg.mkPen(component_colors[index % len(component_colors)], width=1.0, style=QtCore.Qt.PenStyle.DashLine),
                    )
                    self.spectrum_plot.addItem(component_line)
                    self._spectrum_items.append(component_line)
                    self._legend.addItem(component_line, _display_peak_label(label))
                    y_arrays.append(component)

        self.spectrum_plot.setLabel('bottom', 'Energy', **{'font-size': '13pt', 'font-family': 'Arial'})
        self.spectrum_plot.setLabel('left', 'Normalized intensity', **{'font-size': '13pt', 'font-family': 'Arial'})
        self._set_spectrum_view_range(x_values, y_arrays)
        self._set_spectrum_view_mode("calibration")
        self._update_spectrum_header("Calibrated spectrum")
        self.statusBar().showMessage("Current spectrum plotted on calibrated energy axis")

    def compare_calibration_overlay(self):
        if not hasattr(self, 'last_peak_fit_profile'):
            QtWidgets.QMessageBox.warning(self, "Calibration", "Run Peak Fit on the current spectrum first.")
            return
        cal_fit = getattr(self, 'calibration_fit', None)
        if cal_fit is None:
            cal_fit = self.fit_calibration_spectrum()
        if cal_fit is None:
            return
        calibration = getattr(self, 'energy_calibration', None)
        if calibration is None:
            calibration = self.calculate_energy_calibration()
        if calibration is None:
            return

        current_fit = self.last_peak_fit_profile
        current_energy = _energy_axis(calibration, current_fit.x)
        current_y = _normalize_overlay(current_fit.normalized_fit)
        cal_y = _normalize_overlay(cal_fit.normalized_fit)
        self._clear_spectrum()
        self._legend = self._new_spectrum_legend()

        cal_line = pg.PlotDataItem(cal_fit.x, cal_y, pen=pg.mkPen("#d95f02", width=1.3))
        current_line = pg.PlotDataItem(current_energy, current_y, pen=pg.mkPen("#202020", width=1.6))
        self.spectrum_plot.addItem(cal_line)
        self.spectrum_plot.addItem(current_line)
        self._spectrum_items.extend([cal_line, current_line])
        self._legend.addItem(cal_line, "Calibrant fit")
        self._legend.addItem(current_line, "Current fit")
        self.spectrum_plot.setLabel('bottom', 'Energy', **{'font-size': '13pt', 'font-family': 'Arial'})
        self.spectrum_plot.setLabel('left', 'Normalized intensity', **{'font-size': '13pt', 'font-family': 'Arial'})
        self._set_spectrum_view_range(np.concatenate([cal_fit.x, current_energy]), [cal_y, current_y])
        self._set_spectrum_view_mode("calibration")
        self._update_spectrum_header("Calibration comparison")
        self.statusBar().showMessage("Calibration comparison plotted")

    def save_calibrated_spectrum_data(self):
        calibration = getattr(self, 'energy_calibration', None)
        if calibration is None:
            calibration = self.calculate_energy_calibration()
        if calibration is None:
            return
        if self.last_x.size == 0 or self.last_y.size == 0:
            QtWidgets.QMessageBox.warning(self, "Export Cal.", "Plot the current spectrum first.")
            return
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export calibrated spectrum",
            self._default_save_path(f"{self._run_basename()}_calibrated_spectrum.txt"),
            "Text data (*.txt);;CSV data (*.csv);;All files (*.*)",
        )
        if not path:
            return
        delimiter = "," if path.lower().endswith(".csv") else "\t"
        energy = _energy_axis(calibration, self.last_x)
        fit_result = getattr(self, 'last_peak_fit_profile', None)
        with open(path, mode="w", newline="") as handle:
            writer = csv.writer(handle, delimiter=delimiter)
            writer.writerow(["Calibrated spectrum"])
            writer.writerow(["slope", calibration["slope"]])
            writer.writerow(["intercept", calibration["intercept"]])
            writer.writerow(["Kbeta_res residual", calibration.get("residual_res_energy", "")])
            writer.writerow([])
            if fit_result is None:
                writer.writerow(["Energy", "Current intensity"])
                for row_index, energy_value in enumerate(energy):
                    writer.writerow([energy_value, self.last_y[row_index]])
            else:
                fit_energy = _energy_axis(calibration, fit_result.x)
                fit_y = np.asarray(fit_result.normalized_fit, dtype=float)
                current_interp = np.interp(fit_energy, energy, self.last_y)
                scale = _component_scale(fit_result)
                component_labels = list(getattr(fit_result, "peak_labels", []))
                components = []
                if scale is not None:
                    components = [np.asarray(curve, dtype=float) / scale for curve in fit_result.peak_curves]
                writer.writerow(["Energy", "Current intensity", "Peak fit", *[_display_peak_label(label) for label in component_labels[:len(components)]]])
                for row_index, energy_value in enumerate(fit_energy):
                    writer.writerow(
                        [
                            energy_value,
                            current_interp[row_index],
                            fit_y[row_index],
                            *[component[row_index] for component in components],
                        ]
                    )
        self.statusBar().showMessage(f"Saved calibrated spectrum: {path}")

    def _project_manifest(self):
        peak_fit_meta = _fit_metadata(self.last_peak_fit_profile) if hasattr(self, 'last_peak_fit_profile') else None
        calibration_meta = {
            "filepath": getattr(self, "calibration_filepath", ""),
            "source_column": getattr(self, "calibration_source_column", ""),
            "fit_start": self.cal_fit_start.text() if hasattr(self, 'cal_fit_start') else "",
            "fit_end": self.cal_fit_end.text() if hasattr(self, 'cal_fit_end') else "",
            "energy_calibration": _json_safe(getattr(self, "energy_calibration", None)),
            "fit": _fit_metadata(self.calibration_fit) if hasattr(self, 'calibration_fit') else None,
        }
        references = []
        for reference in self.reference_spectra:
            references.append(
                {
                    "path": reference.get("path", ""),
                    "label": reference.get("label", ""),
                    "is_average": bool(reference.get("is_average", False)),
                    "color": reference.get("color", "#4daf4a"),
                    "fit_error": reference.get("fit_error"),
                    "average_reference_fit_error": reference.get("average_reference_fit_error"),
                    "reference_mean_scatter_error": reference.get("reference_mean_scatter_error"),
                    "total_reference_error": reference.get("total_reference_error"),
                    "iad": reference.get("iad"),
                    "iad_error": reference.get("iad_error"),
                    "iad_error_method": reference.get("iad_error_method"),
                    "iad_sample_mc_error": reference.get("iad_sample_mc_error"),
                    "iad_sample_residual_score": reference.get("iad_sample_residual_score"),
                    "iad_reference_fit_score": reference.get("iad_reference_fit_score"),
                    "iad_reference_scatter_error": reference.get("iad_reference_scatter_error"),
                    "iad_reference_total_score": reference.get("iad_reference_total_score"),
                    "iad_mc_trials": reference.get("iad_mc_trials"),
                    "iad_mc_success": reference.get("iad_mc_success"),
                    "iad_mc_failures": reference.get("iad_mc_failures"),
                    "iad_mc_mean": reference.get("iad_mc_mean"),
                    "iad_mc_median": reference.get("iad_mc_median"),
                    "satellite_iad": reference.get("satellite_iad"),
                    "satellite_iad_error": reference.get("satellite_iad_error"),
                    "satellite_iad_error_method": reference.get("satellite_iad_error_method"),
                    "satellite_sample_mc_error": reference.get("satellite_sample_mc_error"),
                    "satellite_reference_scatter_error": reference.get("satellite_reference_scatter_error"),
                    "satellite_fraction": reference.get("satellite_fraction"),
                    "satellite_mc_trials": reference.get("satellite_mc_trials"),
                    "satellite_mc_success": reference.get("satellite_mc_success"),
                    "satellite_mc_failures": reference.get("satellite_mc_failures"),
                    "satellite_mc_mean": reference.get("satellite_mc_mean"),
                    "satellite_mc_median": reference.get("satellite_mc_median"),
                    "satellite_cross_point": reference.get("satellite_cross_point"),
                }
            )
        return {
            "format": PROJECT_FORMAT,
            "version": PROJECT_VERSION,
            "created": datetime.now().isoformat(timespec="seconds"),
            "filepaths": _json_safe(self.filepaths),
            "run_label": self.run_label.text() if hasattr(self, 'run_label') else "Run: Not Loaded",
            "parameters": _json_safe(self.parm),
            "ui": {
                "vmin": self.vmin_spin.value() if hasattr(self, 'vmin_spin') else 0,
                "vmax": self.vmax_spin.value() if hasattr(self, 'vmax_spin') else 100,
                "cmap": self.cmap_combo.currentText() if hasattr(self, 'cmap_combo') else "viridis",
                "roi_rows": self.roi_ranges(),
                "roi2_enabled": self.roi2_check.isChecked() if hasattr(self, 'roi2_check') else False,
                "roi_columns": self.col_range(),
                "bg_rows": self.bg_ranges(),
                "gap_enabled": self._gap_correction_enabled() if hasattr(self, 'gap_check') else True,
                "bg_enabled": self._background_removal_enabled() if hasattr(self, 'bg_check') else False,
                "line_color": self.line_color.name() if hasattr(self, 'line_color') else "#202020",
                "line_style": self.line_style_combo.currentText() if hasattr(self, 'line_style_combo') else "-",
                "line_width": self.line_width_spin.value() if hasattr(self, 'line_width_spin') else 1.6,
                "cross_begin": self.cross_begin.text() if hasattr(self, 'cross_begin') else "",
                "cross_end": self.cross_end.text() if hasattr(self, 'cross_end') else "",
                "eye_ball_cross": self.eye_ball_cross.text() if hasattr(self, 'eye_ball_cross') else "",
                "iad_align_baseline": self.iad_baseline_check.isChecked() if hasattr(self, 'iad_baseline_check') else True,
                "current_tilt_text": getattr(self, 'current_tilt_text', "Tilt: --"),
                "manual_tilt": self.manual_tilt_spin.value() if hasattr(self, 'manual_tilt_spin') else 0.0,
            },
            "peak_fit": peak_fit_meta,
            "calibration": calibration_meta,
            "references": references,
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
        _write_array(spectra, "last_x", getattr(self, "last_x", None))
        _write_array(spectra, "last_y", getattr(self, "last_y", None))
        _write_array(spectra, "last_raw_x", getattr(self, "last_raw_x", None))
        _write_array(spectra, "last_raw_y", getattr(self, "last_raw_y", None))

        if hasattr(self, 'last_peak_fit_profile'):
            _write_fit_group(h5_file, "peak_fit", self.last_peak_fit_profile)

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
                _write_fit_group(cal_group, "fit", self.calibration_fit)

        refs_group = h5_file.create_group("references")
        for index, reference in enumerate(self.reference_spectra):
            ref_group = refs_group.create_group(f"ref_{index}")
            _write_array(ref_group, "x", reference.get("x", []))
            _write_array(ref_group, "y", reference.get("y", []))
            _write_array(ref_group, "reference_scatter_x", reference.get("reference_scatter_x", []))
            _write_array(ref_group, "reference_scatter_mean", reference.get("reference_scatter_mean", []))
            _write_array(ref_group, "reference_scatter_std", reference.get("reference_scatter_std", []))
            _write_array(ref_group, "reference_scatter_valid_n", reference.get("reference_scatter_valid_n", []))

    def save_project(self):
        if h5py is None:
            QtWidgets.QMessageBox.warning(self, "Export Project", "h5py is required to save IXE project files.")
            return
        path = getattr(self, "current_project_path", "")
        if not path:
            path, _selected = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save IXE project",
                self._default_save_path(f"{self._run_basename()}.ixeproj"),
                "IXE project (*.ixeproj);;HDF5 files (*.h5 *.hdf5);;All files (*.*)",
            )
            if not path:
                return
        try:
            manifest = self._project_manifest()
            with h5py.File(path, "w") as h5_file:
                text_dtype = h5py.string_dtype(encoding="utf-8")
                h5_file.create_dataset("manifest", data=json.dumps(manifest, indent=2), dtype=text_dtype)
                h5_file.attrs["format"] = PROJECT_FORMAT
                h5_file.attrs["version"] = PROJECT_VERSION
                self._write_project_arrays(h5_file)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export Project", str(exc))
            return
        self.current_project_path = path
        self.statusBar().showMessage(f"Project saved: {path}")

    def _restore_project_ui(self, manifest):
        ui = manifest.get("ui", {}) or {}
        self.filepaths = list(manifest.get("filepaths", []))
        self.run_label.setText(manifest.get("run_label", "Run: Not Loaded"))
        if hasattr(self, 'file_path_edit'):
            self.file_path_edit.setText("; ".join(self.filepaths))
        self.parm.update(manifest.get("parameters", {}) or {})

        self.vmin_spin.blockSignals(True)
        self.vmax_spin.blockSignals(True)
        self.vmin_spin.setValue(float(ui.get("vmin", self.vmin_spin.value())))
        self.vmax_spin.setValue(float(ui.get("vmax", self.vmax_spin.value())))
        self.vmin_spin.blockSignals(False)
        self.vmax_spin.blockSignals(False)
        cmap = ui.get("cmap")
        if cmap:
            index = self.cmap_combo.findText(cmap)
            if index >= 0:
                self.cmap_combo.setCurrentIndex(index)

        if hasattr(self, 'gap_enabled_check'):
            self.gap_enabled_check.setChecked(bool(ui.get("gap_enabled", self._gap_correction_enabled())))
        if hasattr(self, 'bg_enabled_check'):
            self.bg_enabled_check.setChecked(bool(ui.get("bg_enabled", self._background_removal_enabled())))
        peak_defaults = self.parm.get("peak_fit", {}) or {}
        if hasattr(self, 'physical_fit_check'):
            self.physical_fit_check.setChecked(bool(peak_defaults.get("physical_fit", self.physical_fit_check.isChecked())))
        if hasattr(self, 'tail_baseline_check'):
            self.tail_baseline_check.setChecked(bool(peak_defaults.get("tail_baseline", self.tail_baseline_check.isChecked())))
        self.current_tilt_text = ui.get("current_tilt_text", "Tilt: --")
        if hasattr(self, 'manual_tilt_spin'):
            self.manual_tilt_spin.setValue(float(ui.get("manual_tilt", self.manual_tilt_spin.value())))
        self.line_color = QtGui.QColor(ui.get("line_color", "#202020"))
        line_style = ui.get("line_style", "-")
        style_index = self.line_style_combo.findText(line_style)
        if style_index >= 0:
            self.line_style_combo.setCurrentIndex(style_index)
        self.line_width_spin.setValue(float(ui.get("line_width", self.line_width_spin.value())))
        self.cross_begin.setText(str(ui.get("cross_begin", self.cross_begin.text())))
        self.cross_end.setText(str(ui.get("cross_end", self.cross_end.text())))
        self.eye_ball_cross.setText(str(ui.get("eye_ball_cross", "")))
        if hasattr(self, 'iad_baseline_check'):
            self.iad_baseline_check.setChecked(bool(ui.get("iad_align_baseline", True)))

    def _restore_project_regions(self, manifest):
        ui = manifest.get("ui", {}) or {}
        display_data = getattr(self, "immm", getattr(self, "imm", None))
        if display_data is None:
            return
        self._set_spin_limits(display_data.shape)
        roi_rows = ui.get("roi_rows") or []
        bg_rows = ui.get("bg_rows") or []
        roi_columns = ui.get("roi_columns") or [0, max(display_data.shape[1] - 1, 0)]
        first_roi = roi_rows[0] if roi_rows else [0, min(1, display_data.shape[0] - 1)]
        second_roi = roi_rows[1] if len(roi_rows) > 1 else first_roi
        first_bg = bg_rows[0] if bg_rows else [0, min(1, display_data.shape[0] - 1)]
        second_bg = bg_rows[1] if len(bg_rows) > 1 else first_bg
        self.roi2_check.setChecked(bool(ui.get("roi2_enabled", len(roi_rows) > 1)))
        self._update_roi2_widgets()
        self._set_spin_values(
            {
                self.roi_row_start: first_roi[0],
                self.roi_row_end: first_roi[1],
                self.roi2_row_start: second_roi[0],
                self.roi2_row_end: second_roi[1],
                self.roi_col_start: roi_columns[0],
                self.roi_col_end: roi_columns[1],
                self.bg_row_start: first_bg[0],
                self.bg_row_end: first_bg[1],
                self.bg2_row_start: second_bg[0],
                self.bg2_row_end: second_bg[1],
            }
        )

    def _restore_project_fits(self, h5_file, manifest):
        peak_meta = manifest.get("peak_fit")
        if peak_meta and "peak_fit" in h5_file:
            self.last_peak_fit_profile = _read_fit_group(h5_file["peak_fit"], peak_meta)
            if hasattr(self, 'last_iad_mc_sample'):
                del self.last_iad_mc_sample
            if self.last_peak_fit_profile is not None:
                self._update_peak_fit_results(self.last_peak_fit_profile)

        calibration_meta = manifest.get("calibration", {}) or {}
        cal_group = h5_file.get("calibration")
        if cal_group is not None:
            cal_x = _read_array(cal_group, "x")
            cal_y = _read_array(cal_group, "y")
            if cal_x is not None and cal_y is not None:
                self.calibration_x = cal_x
                self.calibration_y = cal_y
            for attr_name, dataset_name in (
                ("calibration_energy_axis", "calibrated_energy_axis"),
                ("calibrated_spectrum_y", "calibrated_spectrum_y"),
                ("calibrated_fit_energy", "calibrated_fit_energy"),
                ("calibrated_fit_y", "calibrated_fit_y"),
            ):
                values = _read_array(cal_group, dataset_name)
                if values is not None:
                    setattr(self, attr_name, values)
            components = _read_array(cal_group, "calibrated_components")
            if components is not None and components.size:
                self.calibrated_components = [row.copy() for row in components]
            fit_meta = calibration_meta.get("fit")
            if fit_meta and "fit" in cal_group:
                self.calibration_fit = _read_fit_group(cal_group["fit"], fit_meta)

        self.calibration_filepath = calibration_meta.get("filepath", "")
        self.calibration_source_column = calibration_meta.get("source_column", "")
        self.calibration_file_label.setText(
            os.path.basename(self.calibration_filepath) if self.calibration_filepath else "No calibration file loaded"
        )
        self.cal_fit_start.setText(str(calibration_meta.get("fit_start", "") or ""))
        self.cal_fit_end.setText(str(calibration_meta.get("fit_end", "") or ""))
        if hasattr(self, 'calibration_fit') and self.calibration_fit is not None:
            try:
                self._update_calibration_center_outputs("cal", self._calibration_centers3(self.calibration_fit))
            except ValueError:
                pass
        calibration = calibration_meta.get("energy_calibration")
        if calibration:
            self.energy_calibration = calibration
            self._update_calibration_center_outputs(
                "current",
                (
                    calibration.get("pixel_satellite", np.nan),
                    calibration.get("pixel_res", np.nan),
                    calibration.get("pixel_main", np.nan),
                ),
            )
            self._update_calibration_center_outputs(
                "cal",
                (
                    calibration.get("energy_satellite", np.nan),
                    calibration.get("energy_res", np.nan),
                    calibration.get("energy_main", np.nan),
                ),
            )
            self._set_calibration_scalar_output(self.cal_slope_entry, calibration.get("slope", np.nan))
            self._set_calibration_scalar_output(self.cal_intercept_entry, calibration.get("intercept", np.nan))
            self._set_calibration_scalar_output(self.cal_residual_entry, calibration.get("residual_res_energy", np.nan), precision=5)

    def _restore_project_references(self, h5_file, manifest):
        self.reference_spectra = []
        if hasattr(self, 'iad_line_table'):
            self.iad_line_table.setRowCount(0)
        refs_group = h5_file.get("references")
        for index, meta in enumerate(manifest.get("references", []) or []):
            ref_group = refs_group.get(f"ref_{index}") if refs_group is not None else None
            if ref_group is None:
                continue
            reference = {
                "path": meta.get("path", ""),
                "label": meta.get("label", f"Ref {index + 1}"),
                "is_average": bool(meta.get("is_average", False)),
                "color": meta.get("color", "#4daf4a"),
                "fit_error": meta.get("fit_error"),
                "average_reference_fit_error": meta.get("average_reference_fit_error"),
                "reference_mean_scatter_error": meta.get("reference_mean_scatter_error"),
                "total_reference_error": meta.get("total_reference_error"),
                "iad": meta.get("iad"),
                "iad_error": meta.get("iad_error"),
                "iad_error_method": meta.get("iad_error_method"),
                "iad_sample_mc_error": meta.get("iad_sample_mc_error"),
                "iad_sample_residual_score": meta.get("iad_sample_residual_score"),
                "iad_reference_fit_score": meta.get("iad_reference_fit_score"),
                "iad_reference_scatter_error": meta.get("iad_reference_scatter_error"),
                "iad_reference_total_score": meta.get("iad_reference_total_score"),
                "iad_mc_trials": meta.get("iad_mc_trials"),
                "iad_mc_success": meta.get("iad_mc_success"),
                "iad_mc_failures": meta.get("iad_mc_failures"),
                "iad_mc_mean": meta.get("iad_mc_mean"),
                "iad_mc_median": meta.get("iad_mc_median"),
                "satellite_iad": meta.get("satellite_iad"),
                "satellite_iad_error": meta.get("satellite_iad_error"),
                "satellite_iad_error_method": meta.get("satellite_iad_error_method"),
                "satellite_sample_mc_error": meta.get("satellite_sample_mc_error"),
                "satellite_reference_scatter_error": meta.get("satellite_reference_scatter_error"),
                "satellite_fraction": meta.get("satellite_fraction"),
                "satellite_mc_trials": meta.get("satellite_mc_trials"),
                "satellite_mc_success": meta.get("satellite_mc_success"),
                "satellite_mc_failures": meta.get("satellite_mc_failures"),
                "satellite_mc_mean": meta.get("satellite_mc_mean"),
                "satellite_mc_median": meta.get("satellite_mc_median"),
                "satellite_cross_point": meta.get("satellite_cross_point"),
                "x": _read_array(ref_group, "x", np.array([], dtype=float)),
                "y": _read_array(ref_group, "y", np.array([], dtype=float)),
                "reference_scatter_x": _read_array(ref_group, "reference_scatter_x", np.array([], dtype=float)),
                "reference_scatter_mean": _read_array(ref_group, "reference_scatter_mean", np.array([], dtype=float)),
                "reference_scatter_std": _read_array(ref_group, "reference_scatter_std", np.array([], dtype=float)),
                "reference_scatter_valid_n": _read_array(ref_group, "reference_scatter_valid_n", np.array([], dtype=float)),
            }
            self.reference_spectra.append(reference)
            self._insert_iad_reference_row(reference)
            row = self.iad_line_table.rowCount() - 1
            if reference.get("iad") is not None:
                self._set_iad_table_value(row, 2, _format_fit_value(reference.get("iad")))
            if reference.get("iad_error") is not None:
                self._set_iad_table_value(row, 3, _format_fit_value(reference.get("iad_error")))
            if reference.get("satellite_iad") is not None:
                self._set_iad_table_value(row, 4, _format_fit_value(reference.get("satellite_iad")))
            if reference.get("satellite_iad_error") is not None:
                self._set_iad_table_value(row, 5, _format_fit_value(reference.get("satellite_iad_error")))

    def open_project(self):
        if h5py is None:
            QtWidgets.QMessageBox.warning(self, "Open Project", "h5py is required to open IXE project files.")
            return
        path, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open IXE project",
            "",
            "IXE project (*.ixeproj);;HDF5 files (*.h5 *.hdf5);;All files (*.*)",
        )
        if not path:
            return
        try:
            with h5py.File(path, "r") as h5_file:
                manifest = _read_manifest(h5_file)
                if manifest.get("format") != PROJECT_FORMAT:
                    raise ValueError("This file is not an IXE project.")
                self._restore_project_ui(manifest)

                images = h5_file.get("images")
                raw = _read_array(images, "raw")
                processed = _read_array(images, "processed")
                self.imm = raw.astype(np.float32, copy=False) if raw is not None else None
                self.immm = processed.astype(np.float32, copy=False) if processed is not None else None

                masks = h5_file.get("masks")
                raw_gap = _read_array(masks, "raw_gap")
                processed_gap = _read_array(masks, "processed_gap")
                detector_background = _read_array(masks, "detector_background")
                self.raw_gap_mask = raw_gap.astype(bool, copy=False) if raw_gap is not None else None
                self.processed_gap_mask = processed_gap.astype(bool, copy=False) if processed_gap is not None else None
                self.processed_detector_background_mask = detector_background.astype(bool, copy=False) if detector_background is not None else None

                spectra = h5_file.get("spectra")
                self.last_x = _read_array(spectra, "last_x", np.array([], dtype=float))
                self.last_y = _read_array(spectra, "last_y", np.array([], dtype=float))
                self.last_raw_x = _read_array(spectra, "last_raw_x", self.last_x.copy())
                self.last_raw_y = _read_array(spectra, "last_raw_y", self.last_y.copy())
                self._clear_peak_fit_results()
                self._restore_project_fits(h5_file, manifest)
                self._restore_project_references(h5_file, manifest)

            display_data = self.immm if self.immm is not None else self.imm
            if display_data is not None:
                self._restore_project_regions(manifest)
                self.spect_processor = SpectrumProcessor(
                    display_data,
                    int(self.parm.get('n_moveavg', 5)),
                    invalid_pixel_mask=self.processed_gap_mask,
                    detector_background_mask=self.processed_detector_background_mask,
                    edge_valid_fraction_threshold=self.parm.get('ccd_gap', {}).get('edge_valid_fraction', 0.80),
                )
                self.configure_processor()
                title = "Processed image" if self.immm is not None else "Raw image"
                self.display_image(display_data, title)
            else:
                self.spect_processor = None

            self._clear_spectrum()
            if hasattr(self, 'last_peak_fit_profile'):
                self._draw_peak_fit(self.last_peak_fit_profile)
            elif self.last_x.size and self.last_y.size:
                self._draw_spectrum(self.last_x, self.last_y, label="Restored", show_gap_markers=False, view_label="Restored")
            else:
                self._update_spectrum_header()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Open Project", str(exc))
            return
        self.current_project_path = path
        self.statusBar().showMessage(f"Project opened: {path}")

    def _spin(self):
        spin = QtWidgets.QSpinBox()
        spin.setRange(0, 99999)
        spin.setMinimumWidth(76)
        return spin

    def _sync_range_bar_limits(self):
        if hasattr(self, 'roi_row_bar'):
            row_min = self.roi_row_start.minimum()
            row_max = self.roi_row_start.maximum()
            self.roi_row_bar.setLimits(
                row_min,
                row_max,
                self.roi_row_start.value(),
                self.roi_row_end.value(),
                self.roi2_row_start.value(),
                self.roi2_row_end.value(),
                emit=False,
            )
        if hasattr(self, 'roi_col_bar'):
            self.roi_col_bar.setLimits(
                self.roi_col_start.minimum(),
                self.roi_col_start.maximum(),
                self.roi_col_start.value(),
                self.roi_col_end.value(),
                emit=False,
            )
        if hasattr(self, 'bg_row_bar'):
            row_min = self.bg_row_start.minimum()
            row_max = self.bg_row_start.maximum()
            self.bg_row_bar.setLimits(
                row_min,
                row_max,
                self.bg_row_start.value(),
                self.bg_row_end.value(),
                self.bg2_row_start.value(),
                self.bg2_row_end.value(),
                emit=False,
            )

    def _sync_range_bars_from_spins(self):
        if hasattr(self, 'roi_row_bar'):
            self.roi_row_bar.setSecondEnabled(self.roi2_check.isChecked())
            self.roi_row_bar.setValues(
                self.roi_row_start.value(),
                self.roi_row_end.value(),
                self.roi2_row_start.value(),
                self.roi2_row_end.value(),
                emit=False,
            )
        if hasattr(self, 'roi_col_bar'):
            self.roi_col_bar.setValues(self.roi_col_start.value(), self.roi_col_end.value(), emit=False)
        if hasattr(self, 'bg_row_bar'):
            self.bg_row_bar.setValues(
                self.bg_row_start.value(),
                self.bg_row_end.value(),
                self.bg2_row_start.value(),
                self.bg2_row_end.value(),
                emit=False,
            )

    def _on_roi_row_bar_changed(self, first_left, first_right, second_left, second_right):
        self._set_spin_values(
            {
                self.roi_row_start: first_left,
                self.roi_row_end: first_right,
                self.roi2_row_start: second_left,
                self.roi2_row_end: second_right,
            }
        )
        if self.last_x.size:
            self.plot_spectrum()

    def _on_roi_col_bar_changed(self, left, right):
        self._set_spin_values({self.roi_col_start: left, self.roi_col_end: right})
        if self.last_x.size:
            self.plot_spectrum()

    def _on_bg_row_bar_changed(self, first_left, first_right, second_left, second_right):
        self._set_spin_values(
            {
                self.bg_row_start: first_left,
                self.bg_row_end: first_right,
                self.bg2_row_start: second_left,
                self.bg2_row_end: second_right,
            }
        )
        if self.last_x.size and self._background_removal_enabled():
            self.plot_spectrum()

    def _update_roi2_widgets(self):
        enabled = bool(self.roi2_check.isChecked()) if hasattr(self, 'roi2_check') else False
        for widget_name in ('roi2_row_start', 'roi2_row_end'):
            if hasattr(self, widget_name):
                getattr(self, widget_name).setEnabled(enabled)
        if hasattr(self, 'roi_row_bar'):
            self.roi_row_bar.setSecondEnabled(enabled)
        if hasattr(self, 'roi_add_button'):
            self.roi_add_button.setEnabled(not enabled)
        if hasattr(self, 'roi_remove_button'):
            self.roi_remove_button.setEnabled(enabled)

    def add_roi_row_range(self):
        if self.roi2_check.isChecked():
            return
        row_begin, row_end = self.roi_ranges()[0] if self.roi_ranges() else (0, 1)
        max_row = self.roi_row_start.maximum()
        height = max(row_end - row_begin, 1)
        gap = max(2, int(round(height * 0.35)))
        second_begin = min(row_end + gap + 1, max(max_row - height, 0))
        second_end = min(second_begin + height, max_row)
        if second_end <= second_begin:
            second_begin, second_end = row_begin, row_end
        self._set_spin_values(
            {
                self.roi2_row_start: second_begin,
                self.roi2_row_end: second_end,
            }
        )
        self.roi2_check.setChecked(True)
        self._update_roi2_widgets()
        if self.last_x.size:
            self.plot_spectrum()

    def remove_roi_row_range(self):
        if not self.roi2_check.isChecked():
            return
        self.roi2_check.setChecked(False)
        self._update_roi2_widgets()
        self.refresh_regions()
        if self.last_x.size:
            self.plot_spectrum()

    def _current_colormap(self):
        name = self.cmap_combo.currentText() if hasattr(self, 'cmap_combo') else 'viridis'
        try:
            return pg.colormap.get(name)
        except Exception:
            return pg.colormap.get('viridis')

    def update_cmap(self):
        if not hasattr(self, 'image_item'):
            return
        cmap = self._current_colormap()
        try:
            self.colorbar.setColorMap(cmap)
        except Exception:
            pass
        try:
            self.image_item.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        except Exception:
            pass
        self.refresh_image()

    def show_gap_mask(self):
        if self.processed_gap_mask is None:
            QtWidgets.QMessageBox.information(self, "Gap Mask", "Process an image before showing the gap mask.")
            return
        self.display_image(self.immm, "Processed image + CCD gap mask")
        mask = self.processed_gap_mask
        if mask is not None and self.immm is not None and mask.shape == self.immm.shape:
            overlay = np.zeros(mask.shape + (4,), dtype=np.ubyte)
            overlay[mask, 0] = 155
            overlay[mask, 3] = 120
            item = pg.ImageItem(overlay)
            item.setZValue(20)
            self.image_plot.addItem(item)
            self._image_region_items.append(item)

    def save_image_png(self):
        data = self.immm if self.immm is not None else self.imm
        if data is None:
            QtWidgets.QMessageBox.information(self, "Export Image", "Import an image first.")
            return
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export displayed CCD view",
            self._default_save_path(f"{self._run_basename()}_ccd.png"),
            "PNG image (*.png)",
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path = f"{path}.png"
        exporter = ImageExporter(self.image_plot)
        exporter.parameters()['width'] = max(int(self.image_plot.width()), 1800)
        exporter.export(path)
        self.statusBar().showMessage(f"Saved CCD image: {path}")

    def _gap_correction_enabled(self):
        if hasattr(self, 'gap_enabled_check'):
            return self.gap_enabled_check.isChecked()
        return self.gap_check.isChecked() if hasattr(self, 'gap_check') else True

    def _background_removal_enabled(self):
        if hasattr(self, 'bg_enabled_check'):
            return self.bg_enabled_check.isChecked()
        return self.bg_check.isChecked() if hasattr(self, 'bg_check') else False

    def toggle_gap_correction(self, checked=None):
        sender = self.sender()
        if hasattr(self, 'gap_enabled_check'):
            if sender is self.gap_check:
                checked = not self.gap_enabled_check.isChecked()
                self.gap_enabled_check.blockSignals(True)
                self.gap_enabled_check.setChecked(checked)
                self.gap_enabled_check.blockSignals(False)
            else:
                checked = self.gap_enabled_check.isChecked() if checked is None else self.gap_enabled_check.isChecked()
        else:
            checked = self.gap_check.isChecked() if checked is None else bool(checked)
        self.parm.setdefault('ccd_gap', {})['enabled'] = checked
        if self.spect_processor is not None:
            self.configure_processor()
        if self.last_x.size:
            self.plot_spectrum()
        else:
            self._update_spectrum_header()

    def toggle_background_removal(self, checked=None):
        sender = self.sender()
        if hasattr(self, 'bg_enabled_check'):
            if sender is self.bg_check:
                checked = not self.bg_enabled_check.isChecked()
                self.bg_enabled_check.blockSignals(True)
                self.bg_enabled_check.setChecked(checked)
                self.bg_enabled_check.blockSignals(False)
            else:
                checked = self.bg_enabled_check.isChecked() if checked is None else self.bg_enabled_check.isChecked()
        else:
            checked = self.bg_check.isChecked() if checked is None else bool(checked)
        if self.spect_processor is not None:
            self.configure_processor()
        if self.last_x.size:
            self.plot_spectrum()
        else:
            self._update_spectrum_header()

    def pick_line_color(self):
        color = QtWidgets.QColorDialog.getColor(self.line_color, self, "Pick spectrum color")
        if not color.isValid():
            return
        self.line_color = color
        self.update_plot_style()

    def _selected_line_pen(self):
        color = self.line_color if hasattr(self, 'line_color') else QtGui.QColor("#202020")
        width = self.line_width_spin.value() if hasattr(self, 'line_width_spin') else 1.6
        pen = pg.mkPen(color, width=width)
        style_text = self.line_style_combo.currentText() if hasattr(self, 'line_style_combo') else "-"
        style_map = {
            "-": QtCore.Qt.PenStyle.SolidLine,
            "--": QtCore.Qt.PenStyle.DashLine,
            "-.": QtCore.Qt.PenStyle.DashDotLine,
            ":": QtCore.Qt.PenStyle.DotLine,
        }
        pen.setStyle(style_map.get(style_text, QtCore.Qt.PenStyle.SolidLine))
        return pen

    def update_plot_style(self):
        if self.last_x.size:
            mode = getattr(self, 'current_spectrum_view', 'gap')
            if mode == 'raw':
                self.show_raw_spectrum()
            elif mode == 'fit' and hasattr(self, 'last_peak_fit_profile'):
                self.show_peak_fit_view()
            else:
                self.show_gap_corrected_spectrum()

    def _update_peak_fit_model(self):
        peak_defaults = self.parm.setdefault('peak_fit', {})
        peak_defaults['peak_shape'] = self.selected_peak_fit_model()
        peak_defaults['physical_fit'] = self.physical_fit_enabled()
        peak_defaults['tail_baseline'] = self.tail_baseline_enabled()
        if self.physical_fit_enabled() and hasattr(self, 'peak_fit_count') and self.peak_fit_count.value() != 3:
            self.peak_fit_count.setValue(3)

    def selected_peak_fit_model(self):
        if hasattr(self, 'lorentzian_radio') and self.lorentzian_radio.isChecked():
            return "lorentzian"
        return "pseudo_voigt"

    def physical_fit_enabled(self):
        return hasattr(self, 'physical_fit_check') and self.physical_fit_check.isChecked()

    def tail_baseline_enabled(self):
        return hasattr(self, 'tail_baseline_check') and self.tail_baseline_check.isChecked()

    def _peak_fit_config(self):
        try:
            x_min_text = self.peak_fit_start.text().strip()
            x_max_text = self.peak_fit_end.text().strip()
            x_min = float(x_min_text) if x_min_text else None
            x_max = float(x_max_text) if x_max_text else None
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Peak Fit", "Fit range values must be numeric.")
            return None
        if x_min is not None and x_max is not None and x_min > x_max:
            x_min, x_max = x_max, x_min
        peak_defaults = self.parm.setdefault('peak_fit', {})
        peak_defaults['n_peaks'] = int(self.peak_fit_count.value())
        peak_defaults['x_min'] = x_min
        peak_defaults['x_max'] = x_max
        peak_defaults['peak_shape'] = self.selected_peak_fit_model()
        peak_defaults['physical_fit'] = self.physical_fit_enabled()
        peak_defaults['tail_baseline'] = self.tail_baseline_enabled()
        if peak_defaults['physical_fit'] and peak_defaults['n_peaks'] != 3:
            QtWidgets.QMessageBox.warning(self, "Peak Fit", "Physical Kβ fit uses exactly 3 components.")
            self.peak_fit_count.setValue(3)
            peak_defaults['n_peaks'] = 3
        try:
            return PeakFitConfig(
                n_peaks=peak_defaults['n_peaks'],
                x_min=x_min,
                x_max=x_max,
                peak_shape=peak_defaults['peak_shape'],
                physical_fit=peak_defaults['physical_fit'],
                tail_baseline=peak_defaults['tail_baseline'],
            )
        except PeakFitError as exc:
            QtWidgets.QMessageBox.warning(self, "Peak Fit", str(exc))
            return None

    def _clear_peak_fit_results(self):
        if hasattr(self, 'last_iad_mc_sample'):
            del self.last_iad_mc_sample
        if hasattr(self, 'peak_fit_summary_label'):
            self.peak_fit_summary_label.setText("Peak fit parameters: not fitted")
        if hasattr(self, 'peak_fit_error_entry'):
            self.peak_fit_error_entry.clear()
        if hasattr(self, 'peak_fit_table'):
            self.peak_fit_table.setRowCount(0)

    def _update_peak_fit_results(self, fit_result):
        rows = _peak_fit_parameter_rows(fit_result)
        model = getattr(fit_result, 'peak_shape', 'pseudo_voigt')
        physical_text = "Physical " if getattr(fit_result, 'physical_fit', False) else ""
        baseline_text = "; tail baseline ON" if getattr(fit_result, 'tail_baseline_enabled', False) else ""
        if model == "pseudo_voigt":
            self.peak_fit_summary_label.setText(f"{physical_text}Pseudo-Voigt fit{baseline_text}; Lorentzian components: {len(rows)}")
        else:
            self.peak_fit_summary_label.setText(f"{physical_text}Lorentzian fit{baseline_text}; peaks: {len(rows)}")
        if hasattr(self, 'peak_fit_error_entry'):
            self.peak_fit_error_entry.setText(_format_fit_error(getattr(fit_result, 'relative_fit_error', np.nan)))
        self.peak_fit_table.setRowCount(len(rows))
        for row_index, row_values in enumerate(rows):
            for column_index, value in enumerate(row_values):
                text = str(value) if column_index < 2 else _format_fit_value(value)
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.peak_fit_table.setItem(row_index, column_index, item)

    def show_peak_fit_profile(self):
        if self.last_x.size == 0:
            QtWidgets.QMessageBox.information(self, "Peak Fit", "Plot a spectrum first.")
            return
        fit_config = self._peak_fit_config()
        if fit_config is None:
            return
        try:
            fit_result = fit_spectrum(self.last_x, self.last_y, fit_config)
        except PeakFitError as exc:
            QtWidgets.QMessageBox.warning(self, "Peak Fit", str(exc))
            return
        self.last_peak_fit_profile = fit_result
        if hasattr(self, 'last_iad_mc_sample'):
            del self.last_iad_mc_sample
        self._update_peak_fit_results(fit_result)
        self._set_spectrum_view_mode("fit")
        self._draw_peak_fit(fit_result)
        centers = ", ".join(
            f"{_display_peak_label(label)}={center:.2f}"
            for label, center in zip(fit_result.peak_labels, fit_result.centers)
        )
        self.statusBar().showMessage(f"Peak fit centers: {centers}")

    def show_raw_spectrum(self):
        if self.last_x.size == 0:
            QtWidgets.QMessageBox.information(self, "Raw", "Plot a spectrum first.")
            return
        x_data = getattr(self, 'last_raw_x', self.last_x)
        y_data = getattr(self, 'last_raw_y', self.last_y)
        self._set_spectrum_view_mode("raw")
        self._draw_spectrum(x_data, y_data, label="Raw", show_gap_markers=False, view_label="Raw")

    def show_gap_corrected_spectrum(self):
        if self.last_x.size == 0:
            QtWidgets.QMessageBox.information(self, "Corrected", "Plot a spectrum first.")
            return
        self._set_spectrum_view_mode("gap")
        self._draw_spectrum(
            self.last_x,
            self.last_y,
            label="Corrected",
            show_gap_markers=False,
            view_label="Corrected",
        )

    def show_peak_fit_view(self):
        if not hasattr(self, 'last_peak_fit_profile'):
            QtWidgets.QMessageBox.information(self, "Peak Fit", "Run Peak Fit first.")
            return
        self._set_spectrum_view_mode("fit")
        self._draw_peak_fit(self.last_peak_fit_profile)

    def show_calibration_view(self):
        if hasattr(self, 'calibration_energy_axis') and hasattr(self, 'calibrated_spectrum_y'):
            self.apply_energy_calibration()
            return
        if hasattr(self, 'calibration_fit'):
            self.show_calibration_fit()
            return
        self._set_spectrum_view_mode("calibration")
        self._update_spectrum_header("Calibration")
        QtWidgets.QMessageBox.information(self, "Calibration", "Import/Fit calibration data or apply a calibration first.")

    def _draw_peak_fit(self, fit_result):
        self._clear_spectrum()
        self.spectrum_plot.setLabel('bottom', 'Column Index', **{'font-size': '13pt', 'font-family': 'Arial'})
        self.spectrum_plot.setLabel('left', 'Normalized intensity', **{'font-size': '13pt', 'font-family': 'Arial'})
        self._legend = self._new_spectrum_legend()

        x_fit = np.asarray(fit_result.x, dtype=float)
        raw_y = np.asarray(getattr(fit_result, 'raw_y', np.array([], dtype=float)), dtype=float)
        fit_y = np.asarray(getattr(fit_result, 'fit_y', raw_y), dtype=float)
        tail_baseline = np.asarray(getattr(fit_result, 'tail_baseline', np.zeros_like(raw_y)), dtype=float)
        tail_enabled = bool(getattr(fit_result, 'tail_baseline_enabled', False))

        y_arrays = []
        if tail_enabled:
            raw_line = pg.PlotDataItem(x_fit, raw_y, pen=pg.mkPen('#b7b7b7', width=0.9))
            tail_line = pg.PlotDataItem(
                x_fit,
                tail_baseline,
                pen=pg.mkPen('#8c8c8c', width=1.0, style=QtCore.Qt.PenStyle.DashLine),
            )
            corrected_line = pg.PlotDataItem(x_fit, fit_y, pen=pg.mkPen('#5f5f5f', width=1.0))
            self.spectrum_plot.addItem(raw_line)
            self.spectrum_plot.addItem(tail_line)
            self.spectrum_plot.addItem(corrected_line)
            self._spectrum_items.extend([raw_line, tail_line, corrected_line])
            self._legend.addItem(raw_line, "Raw")
            self._legend.addItem(tail_line, "Tail baseline")
            self._legend.addItem(corrected_line, "Corrected")
            y_arrays.extend([raw_y, tail_baseline, fit_y])
        else:
            raw_line = pg.PlotDataItem(x_fit, raw_y, pen=pg.mkPen('#9a9a9a', width=1.0))
            self.spectrum_plot.addItem(raw_line)
            self._spectrum_items.append(raw_line)
            self._legend.addItem(raw_line, "Spectrum")
            y_arrays.append(raw_y)

        fit_line = pg.PlotDataItem(fit_result.x, fit_result.normalized_fit, pen=self._selected_line_pen())
        self.spectrum_plot.addItem(fit_line)
        self._spectrum_items.append(fit_line)
        self._legend.addItem(fit_line, "Peak fit")

        scale = _component_scale(fit_result)
        component_y_arrays = []
        component_colors = ['#2b8cbe', '#e34a33', '#31a354', '#756bb1', '#fdae6b', '#636363']
        if scale is not None:
            for index, peak_curve in enumerate(fit_result.peak_curves):
                component = np.asarray(peak_curve, dtype=float) / scale
                component_y_arrays.append(component)
                label = fit_result.peak_labels[index] if index < len(fit_result.peak_labels) else f"Peak {index + 1}"
                component_line = pg.PlotDataItem(
                    fit_result.x,
                    component,
                    pen=pg.mkPen(component_colors[index % len(component_colors)], width=1.0, style=QtCore.Qt.PenStyle.DashLine),
                )
                self.spectrum_plot.addItem(component_line)
                self._spectrum_items.append(component_line)
                self._legend.addItem(component_line, _display_peak_label(label))

        y_arrays.append(fit_result.normalized_fit)
        y_arrays.extend(component_y_arrays)

        residual = np.asarray(
            getattr(fit_result, 'fit_residual', fit_y - fit_result.normalized_fit),
            dtype=float,
        )
        if residual.shape == x_fit.shape and np.any(np.isfinite(residual)):
            residual_abs = np.abs(residual[np.isfinite(residual)])
            residual_max = float(np.nanmax(residual_abs)) if residual_abs.size else 0.0
            if np.isfinite(residual_max) and residual_max > np.finfo(float).eps:
                y_min, y_max = _finite_min_max(y_arrays)
                y_span = max(y_max - y_min, np.finfo(float).eps)
                residual_display_height = 0.12 * y_span
                residual_scale = residual_display_height / residual_max
                residual_zero = y_min - 0.12 * y_span
                zero_y = np.full_like(x_fit, residual_zero, dtype=float)
                residual_y = residual_zero + residual * residual_scale
                positive_y = residual_zero + np.where(residual > 0, residual * residual_scale, 0.0)
                negative_y = residual_zero + np.where(residual < 0, residual * residual_scale, 0.0)

                zero_curve = pg.PlotDataItem(
                    x_fit,
                    zero_y,
                    pen=pg.mkPen(QtGui.QColor(120, 120, 120, 150), width=0.8, style=QtCore.Qt.PenStyle.DotLine),
                )
                positive_curve = pg.PlotDataItem(x_fit, positive_y, pen=None)
                negative_curve = pg.PlotDataItem(x_fit, negative_y, pen=None)
                residual_line = pg.PlotDataItem(
                    x_fit,
                    residual_y,
                    pen=pg.mkPen(QtGui.QColor(118, 72, 185, 210), width=0.9),
                )
                positive_fill = pg.FillBetweenItem(
                    positive_curve,
                    zero_curve,
                    brush=pg.mkBrush(QtGui.QColor(150, 122, 235, 58)),
                )
                negative_fill = pg.FillBetweenItem(
                    negative_curve,
                    zero_curve,
                    brush=pg.mkBrush(QtGui.QColor(236, 116, 136, 58)),
                )
                for item in (positive_curve, negative_curve, zero_curve, positive_fill, negative_fill, residual_line):
                    self.spectrum_plot.addItem(item)
                    self._spectrum_items.append(item)
                self._legend.addItem(residual_line, "Fit residual (scaled)")
                y_arrays.extend([positive_y, negative_y, zero_y])

        self._set_spectrum_view_range(x_fit, y_arrays)
        self._update_spectrum_header("Peak Fit")

    def save_spectrum_data(self):
        if self.last_x.size == 0:
            QtWidgets.QMessageBox.information(self, "Export Spectrum", "Plot a spectrum first.")
            return
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export spectrum data",
            self._default_save_path(f"{self._run_basename()}_spectrum.txt"),
            "Text data (*.txt);;CSV data (*.csv);;All files (*.*)",
        )
        if not path:
            return
        delimiter = "," if path.lower().endswith(".csv") else "\t"
        with open(path, mode="w", newline="") as handle:
            writer = csv.writer(handle, delimiter=delimiter)
            writer.writerow([f"Run", self.run_label.text()])
            writer.writerow(["Background removed", self._background_removal_enabled()])
            writer.writerow(["CCD gap corrected", self._gap_correction_enabled()])
            writer.writerow(["ROI Rows", "; ".join(f"{begin}-{end}" for begin, end in self.roi_ranges())])
            writer.writerow(["ROI Columns", f"{self.col_range()[0]}-{self.col_range()[1]}"])
            writer.writerow(["BG Rows", "; ".join(f"{begin}-{end}" for begin, end in self.bg_ranges())])
            writer.writerow(["X", "Y"])
            for x_value, y_value in zip(self.last_x, self.last_y):
                writer.writerow([x_value, y_value])
        self.statusBar().showMessage(f"Saved spectrum data: {path}")

    def save_peak_fit_parameters(self):
        if not hasattr(self, 'last_peak_fit_profile'):
            QtWidgets.QMessageBox.information(self, "Export Params", "Run Peak Fit before exporting peak parameters.")
            return
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export peak fit parameters",
            self._default_save_path(f"{self._run_basename()}_peak_params.txt"),
            "Text data (*.txt);;CSV data (*.csv);;All files (*.*)",
        )
        if not path:
            return
        delimiter = "," if path.lower().endswith(".csv") else "\t"
        fit_result = self.last_peak_fit_profile
        rows = _peak_fit_parameter_rows(fit_result)
        with open(path, mode="w", newline="") as handle:
            writer = csv.writer(handle, delimiter=delimiter)
            model = getattr(fit_result, 'peak_shape', 'pseudo_voigt')
            writer.writerow(["Peak fit model", _peak_fit_model_label(model)])
            writer.writerow(["Physical fit", bool(getattr(fit_result, 'physical_fit', False))])
            writer.writerow(["Tail baseline", bool(getattr(fit_result, 'tail_baseline_enabled', False))])
            writer.writerow(["Fit error", getattr(fit_result, 'relative_fit_error', np.nan)])
            writer.writerow(["Fit residual area", getattr(fit_result, 'fit_residual_area', np.nan)])
            writer.writerow(["Fit total corrected intensity", getattr(fit_result, 'fit_total_intensity', np.nan)])
            if model == "pseudo_voigt":
                writer.writerow(["Lorentzian components", len(rows)])
            writer.writerow(["Peak", "Model", "Lorentzian fraction", "Position", "Width", "Area"])
            writer.writerows(rows)
        self.statusBar().showMessage(f"Saved peak parameters: {path}")

    def save_peak_fit_data(self):
        if not hasattr(self, 'last_peak_fit_profile'):
            QtWidgets.QMessageBox.information(self, "Export PKfit", "Run Peak Fit before exporting the fitted profile.")
            return
        fit_result = self.last_peak_fit_profile
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export peak fit profile",
            self._default_save_path(f"{self._run_basename()}_pkfit.txt"),
            "Text data (*.txt);;CSV data (*.csv);;All files (*.*)",
        )
        if not path:
            return
        delimiter = "," if path.lower().endswith(".csv") else "\t"
        scale = _component_scale(fit_result)
        components = []
        for curve in fit_result.peak_curves:
            if scale is None:
                components.append(np.full_like(fit_result.normalized_fit, np.nan, dtype=float))
            else:
                components.append(np.asarray(curve, dtype=float) / scale)
        labels = [_display_peak_label(label) for label in fit_result.peak_labels]
        with open(path, mode="w", newline="") as handle:
            writer = csv.writer(handle, delimiter=delimiter)
            writer.writerow(["Peak fit model", _peak_fit_model_label(getattr(fit_result, 'peak_shape', 'pseudo_voigt'))])
            writer.writerow(["Physical fit", bool(getattr(fit_result, 'physical_fit', False))])
            writer.writerow(["Tail baseline", bool(getattr(fit_result, 'tail_baseline_enabled', False))])
            writer.writerow(["Fit error", getattr(fit_result, 'relative_fit_error', np.nan)])
            writer.writerow(["Fit residual area", getattr(fit_result, 'fit_residual_area', np.nan)])
            writer.writerow(["Fit total corrected intensity", getattr(fit_result, 'fit_total_intensity', np.nan)])
            writer.writerow(["Peak parameters"])
            writer.writerow(["Peak", "Lorentzian fraction", "Position", "Width", "Area"])
            for label, _model_label, fraction, center, width, area in _peak_fit_parameter_rows(fit_result):
                writer.writerow([label, fraction, center, width, area])
            tail_baseline = np.asarray(getattr(fit_result, 'tail_baseline', np.zeros_like(fit_result.raw_y)), dtype=float)
            corrected_y = np.asarray(getattr(fit_result, 'fit_y', fit_result.raw_y), dtype=float)
            fit_residual = np.asarray(getattr(fit_result, 'fit_residual', corrected_y - fit_result.normalized_fit), dtype=float)
            normalized_residual = np.asarray(
                getattr(fit_result, 'normalized_fit_residual', np.full_like(fit_residual, np.nan, dtype=float)),
                dtype=float,
            )
            writer.writerow(["X", "Raw Y", "Tail baseline", "Corrected Y", "Peak fit", "Fit residual", "Normalized residual", *labels])
            for row_index, x_value in enumerate(fit_result.x):
                writer.writerow(
                    [
                        x_value,
                        fit_result.raw_y[row_index],
                        tail_baseline[row_index],
                        corrected_y[row_index],
                        fit_result.normalized_fit[row_index],
                        fit_residual[row_index],
                        normalized_residual[row_index],
                        *[component[row_index] for component in components],
                    ]
                )
        self.statusBar().showMessage(f"Saved peak fit profile: {path}")

    def _run_basename(self):
        label = self.run_label.text() if hasattr(self, 'run_label') else ""
        match = re.search(r"Run\s+(\d+)", label)
        if match:
            return f"Run_{match.group(1)}"
        return "qt_spectrum"

    def _default_save_path(self, filename):
        if os.path.isabs(filename):
            return filename
        for filepath in getattr(self, 'filepaths', []):
            folder = os.path.dirname(os.path.abspath(filepath))
            if folder and os.path.isdir(folder):
                return os.path.join(folder, filename)
        return filename

    def _run_display_text(self):
        if not hasattr(self, 'run_label'):
            return "Run: Not Loaded"
        text = self.run_label.text().strip()
        return text if text and text != "No image loaded" else "Run: Not Loaded"

    def _set_image_title(self, title):
        if hasattr(self, 'image_run_title'):
            self.image_run_title.setText(self._run_display_text())
        if hasattr(self, 'image_title'):
            self.image_title.setText(title)

    def _spectrum_status_parts(self, view_label=None):
        if self.last_x.size == 0:
            return ["No spectrum plotted"]
        parts = []
        if view_label:
            parts.append(f"View: {view_label}")
        parts.append("Gap corrected ON" if self._gap_correction_enabled() else "Gap corrected OFF")
        parts.append("BG remove ON" if self._background_removal_enabled() else "BG remove OFF")
        roi_text = "; ".join(f"{begin}-{end}" for begin, end in self.roi_ranges())
        col_begin, col_end = self.col_range()
        if roi_text:
            parts.append(f"ROI rows {roi_text}")
        parts.append(f"Cols {col_begin}-{col_end}")
        return parts

    def _update_spectrum_header(self, view_label=None):
        if not hasattr(self, 'spectrum_title'):
            return
        run_text = self._run_display_text()
        if run_text == "Run: Not Loaded":
            self.spectrum_title.setText("XES Spectrum")
        else:
            self.spectrum_title.setText(f"{run_text} | XES Spectrum")
        self.spectrum_status.setText(" | ".join(self._spectrum_status_parts(view_label)))

    def _set_spectrum_view_mode(self, mode):
        self.current_spectrum_view = mode
        buttons = getattr(self, 'spectrum_view_buttons', {})
        for button_mode, button in buttons.items():
            button.blockSignals(True)
            button.setChecked(button_mode == mode)
            button.blockSignals(False)

    def _build_image_panel(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self.image_run_title = QtWidgets.QLabel("Run: Not Loaded")
        self.image_run_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_run_title.setFont(QtGui.QFont("Arial", 13))
        layout.addWidget(self.image_run_title)
        self.image_title = QtWidgets.QLabel("CCD image")
        self.image_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_title.setFont(QtGui.QFont("Arial", 11))
        self.image_title.setStyleSheet("color: #555;")
        layout.addWidget(self.image_title)
        self.image_widget = pg.GraphicsLayoutWidget()
        self.image_widget.setBackground('w')
        self.image_plot = self.image_widget.addPlot(row=0, col=0)
        self.image_plot.setLabel('bottom', 'Columns')
        self.image_plot.setLabel('left', 'Rows')
        self.image_plot.invertY(True)
        self.image_plot.setAspectLocked(False)
        self.image_plot.showGrid(x=False, y=False)
        self.image_item = pg.ImageItem()
        self.image_plot.addItem(self.image_item)
        self.colorbar = pg.ColorBarItem(values=(0, 100), colorMap=pg.colormap.get('viridis'))
        self.colorbar.setImageItem(self.image_item, insert_in=self.image_plot)
        layout.addWidget(self.image_widget, stretch=1)

        display_controls = QtWidgets.QHBoxLayout()
        display_controls.setContentsMargins(4, 4, 0, 0)
        display_controls.setSpacing(4)
        display_controls.addWidget(QtWidgets.QLabel("vmin:"))
        display_controls.addWidget(self.vmin_spin)
        display_controls.addSpacing(12)
        display_controls.addWidget(QtWidgets.QLabel("vmax:"))
        display_controls.addWidget(self.vmax_spin)
        display_controls.addSpacing(12)
        display_controls.addWidget(QtWidgets.QLabel("Cmap:"))
        display_controls.addWidget(self.cmap_combo)
        display_controls.addStretch(1)
        layout.addLayout(display_controls)
        return panel

    def _build_spectrum_panel(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self.spectrum_title = QtWidgets.QLabel("XES Spectrum")
        self.spectrum_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.spectrum_title.setFont(QtGui.QFont("Arial", 13))
        layout.addWidget(self.spectrum_title)

        self.spectrum_status = QtWidgets.QLabel("No spectrum plotted")
        self.spectrum_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.spectrum_status.setFont(QtGui.QFont("Arial", 10))
        self.spectrum_status.setStyleSheet("color: #555;")
        layout.addWidget(self.spectrum_status)

        self.spectrum_widget = pg.GraphicsLayoutWidget()
        self.spectrum_widget.setBackground('w')
        self.spectrum_plot = self.spectrum_widget.addPlot(row=0, col=0)
        self.spectrum_plot.setLabel('bottom', 'Column Index')
        self.spectrum_plot.setLabel('left', 'Normalized intensity')
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.12)
        self.spectrum_plot.setMouseEnabled(x=True, y=True)
        self.spectrum_plot.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self._style_spectrum_axes()
        layout.addWidget(self.spectrum_widget, stretch=1)

        view_bar = QtWidgets.QHBoxLayout()
        view_bar.setContentsMargins(0, 2, 0, 0)
        view_bar.setSpacing(4)
        view_bar.addStretch(1)
        self.spectrum_view_buttons = {}
        for mode, label, callback in (
            ("raw", "Raw", self.show_raw_spectrum),
            ("gap", "Corrected", self.show_gap_corrected_spectrum),
            ("fit", "Peak Fit", self.show_peak_fit_view),
            ("calibration", "Calibration", self.show_calibration_view),
        ):
            button = QtWidgets.QPushButton(label)
            button.setCheckable(True)
            button.setFixedHeight(24)
            button.clicked.connect(lambda _checked=False, slot=callback: slot())
            self.spectrum_view_buttons[mode] = button
            view_bar.addWidget(button)
        layout.addLayout(view_bar)
        return panel

    def _style_spectrum_axes(self):
        font = QtGui.QFont("Arial", 11)
        for axis_name in ('left', 'bottom'):
            axis = self.spectrum_plot.getAxis(axis_name)
            axis.setTickFont(font)
            axis.setStyle(tickLength=5, tickTextOffset=4)
            axis.setPen(pg.mkPen('#222', width=1))
            axis.setTextPen(pg.mkPen('#222'))
        self.spectrum_plot.setLabel('bottom', 'Column Index', **{'font-size': '13pt', 'font-family': 'Arial'})
        self.spectrum_plot.setLabel('left', 'Normalized intensity', **{'font-size': '13pt', 'font-family': 'Arial'})

    def import_tiff(self):
        path, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import Image",
            "",
            "TIFF files (*.tif *.tiff);;All files (*.*)",
        )
        if not path:
            return
        previous_regions = self._current_region_state() if self.imm is not None else None
        try:
            self.imm = _read_tiff_array(path)
            self.filepaths = [path]
            self._after_image_loaded("Raw image", previous_regions=previous_regions)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Import Image", str(exc))

    def import_tiff_stack(self):
        paths, _selected = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Import Stack Image",
            "",
            "TIFF files (*.tif *.tiff);;All files (*.*)",
        )
        if not paths:
            return
        previous_regions = self._current_region_state() if self.imm is not None else None
        try:
            first = _read_tiff_array(paths[0])
            total = first.astype(np.float64, copy=True)
            expected_shape = first.shape
            for path in paths[1:]:
                image = _read_tiff_array(path)
                if image.shape != expected_shape:
                    raise ValueError(f"{path} has shape {image.shape}; expected {expected_shape}.")
                total += image.astype(np.float64, copy=False)
            self.imm = total
            self.filepaths = list(paths)
            title = f"Stacked raw image ({len(paths)} images)" if len(paths) > 1 else "Raw image"
            self._after_image_loaded(title, previous_regions=previous_regions)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Import Stack Image", str(exc))

    def _after_image_loaded(self, title, previous_regions=None):
        self.immm = None
        self.spect_processor = None
        self.raw_gap_mask = None
        self.processed_gap_mask = None
        self.processed_detector_background_mask = None
        self.current_tilt_text = "Tilt: --"
        if hasattr(self, 'manual_tilt_spin'):
            self.manual_tilt_spin.blockSignals(True)
            self.manual_tilt_spin.setValue(0.0)
            self.manual_tilt_spin.blockSignals(False)
        self.last_x = np.array([], dtype=float)
        self.last_y = np.array([], dtype=float)
        if hasattr(self, 'last_peak_fit_profile'):
            del self.last_peak_fit_profile
        if hasattr(self, 'last_iad_mc_sample'):
            del self.last_iad_mc_sample
        self.run_label.setText(_run_label_from_paths(self.filepaths))
        if hasattr(self, 'file_path_edit'):
            self.file_path_edit.setText("; ".join(self.filepaths))
        if hasattr(self, 'spectrum_plot'):
            self._clear_spectrum()
        self._clear_peak_fit_results()
        self._update_spectrum_header()
        self._set_spin_limits(self.imm.shape)
        if previous_regions is None:
            self._set_default_regions(self.imm.shape)
            self.roi2_check.setChecked(False)
            self._update_roi2_widgets()
        else:
            self._restore_region_state(previous_regions, self.imm.shape)
        finite = self.imm[np.isfinite(self.imm)]
        if finite.size:
            self.vmin_spin.blockSignals(True)
            self.vmax_spin.blockSignals(True)
            self.vmin_spin.setValue(int(round(float(np.nanpercentile(finite, 1)))))
            self.vmax_spin.setValue(int(round(float(np.nanpercentile(finite, 99.5)))))
            self.vmin_spin.blockSignals(False)
            self.vmax_spin.blockSignals(False)
        self._process_image_with_tilt(0.0, display_title=title)
        self.statusBar().showMessage(f"Loaded {title} and initialized spectrum extraction: {self.imm.shape}")

    def _set_spin_limits(self, shape):
        rows, cols = shape
        for spin in (
            self.roi_row_start,
            self.roi_row_end,
            self.roi2_row_start,
            self.roi2_row_end,
            self.bg_row_start,
            self.bg_row_end,
            self.bg2_row_start,
            self.bg2_row_end,
        ):
            spin.setRange(0, max(rows - 1, 0))
        for spin in (self.roi_col_start, self.roi_col_end):
            spin.setRange(0, max(cols - 1, 0))
        self._sync_range_bar_limits()

    def _current_region_state(self):
        return {
            "roi_first": (self.roi_row_start.value(), self.roi_row_end.value()),
            "roi_second": (self.roi2_row_start.value(), self.roi2_row_end.value()),
            "roi2_enabled": self.roi2_check.isChecked(),
            "columns": (self.roi_col_start.value(), self.roi_col_end.value()),
            "bg_first": (self.bg_row_start.value(), self.bg_row_end.value()),
            "bg_second": (self.bg2_row_start.value(), self.bg2_row_end.value()),
        }

    def _clamp_region_pair(self, pair, max_value):
        max_value = max(int(max_value), 0)
        begin, end = _safe_int_pair(pair[0], pair[1])
        begin = max(0, min(int(begin), max_value))
        end = max(0, min(int(end), max_value))
        if max_value > 0 and end <= begin:
            if begin >= max_value:
                begin = max_value - 1
                end = max_value
            else:
                end = begin + 1
        return begin, end

    def _restore_region_state(self, state, shape):
        rows, cols = shape
        row_max = max(rows - 1, 0)
        col_max = max(cols - 1, 0)
        roi_first = self._clamp_region_pair(state.get("roi_first", (0, 1)), row_max)
        roi_second = self._clamp_region_pair(state.get("roi_second", roi_first), row_max)
        columns = self._clamp_region_pair(state.get("columns", (0, col_max)), col_max)
        bg_first = self._clamp_region_pair(state.get("bg_first", (0, 1)), row_max)
        bg_second = self._clamp_region_pair(state.get("bg_second", bg_first), row_max)
        self.roi2_check.setChecked(bool(state.get("roi2_enabled", False)))
        self._update_roi2_widgets()
        self._set_spin_values(
            {
                self.roi_row_start: roi_first[0],
                self.roi_row_end: roi_first[1],
                self.roi2_row_start: roi_second[0],
                self.roi2_row_end: roi_second[1],
                self.roi_col_start: columns[0],
                self.roi_col_end: columns[1],
                self.bg_row_start: bg_first[0],
                self.bg_row_end: bg_first[1],
                self.bg2_row_start: bg_second[0],
                self.bg2_row_end: bg_second[1],
            }
        )

    def _set_default_regions(self, shape):
        rows, cols = shape
        center = rows // 2
        roi_half = max(4, rows // 80)
        bg_gap = max(10, rows // 40)
        bg_height = max(roi_half * 2, 8)
        self._set_spin_values(
            {
                self.roi_row_start: max(center - roi_half, 0),
                self.roi_row_end: min(center + roi_half, rows - 1),
                self.roi2_row_start: min(center + bg_gap, rows - 1),
                self.roi2_row_end: min(center + bg_gap + roi_half * 2, rows - 1),
                self.roi_col_start: 0,
                self.roi_col_end: max(cols - 1, 0),
                self.bg_row_start: max(center + bg_gap + bg_height, 0),
                self.bg_row_end: min(center + bg_gap + 2 * bg_height, rows - 1),
                self.bg2_row_start: max(center - bg_gap - 2 * bg_height, 0),
                self.bg2_row_end: max(center - bg_gap - bg_height, 0),
            }
        )

    def _set_spin_values(self, mapping):
        for spin, value in mapping.items():
            spin.blockSignals(True)
            spin.setValue(int(value))
            spin.blockSignals(False)
        self.refresh_regions()
        self._sync_range_bars_from_spins()

    def process_image(self):
        if self.imm is None:
            QtWidgets.QMessageBox.information(self, "Tilt Correction", "Import an image first.")
            return
        try:
            tilt = estimate_image_tilt_angle(self.imm)
            self.manual_tilt_spin.blockSignals(True)
            self.manual_tilt_spin.setValue(float(tilt))
            self.manual_tilt_spin.blockSignals(False)
            self._process_image_with_tilt(tilt)
            self.statusBar().showMessage(f"Applied auto tilt correction {tilt:.2f} deg")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Tilt Correction", str(exc))

    def apply_manual_tilt(self):
        if self.imm is None:
            QtWidgets.QMessageBox.information(self, "Manual Tilt", "Import an image first.")
            return
        try:
            tilt = float(self.manual_tilt_spin.value())
            self._process_image_with_tilt(tilt)
            self.statusBar().showMessage(f"Processed image with manual tilt {tilt:.2f} deg")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Manual Tilt", str(exc))

    def _process_image_with_tilt(self, tilt, display_title="Processed image"):
        self.current_tilt_text = f"Tilt: {tilt:.2f} deg"
        gap_defaults = self.parm['ccd_gap']
        self.raw_gap_mask = detect_detector_gap_mask(
            self.imm,
            max_width=gap_defaults.get('mask_max_width', 16),
            drop_fraction=gap_defaults.get('mask_drop_fraction', 0.55),
            dilate=gap_defaults.get('mask_dilate', 1),
            center_window=gap_defaults.get('mask_center_window', 0.22),
            row_center_window=gap_defaults.get('mask_row_center_window', 0.22),
            col_center_window=gap_defaults.get('mask_col_center_window', 0.20),
        )
        self.processed_gap_mask = rotate_detector_gap_mask(self.raw_gap_mask, tilt)
        self.processed_detector_background_mask = rotate_detector_background_mask(self.imm.shape, tilt)
        self.immm = scipy.ndimage.rotate(self.imm, angle=tilt)
        self.last_x = np.array([], dtype=float)
        self.last_y = np.array([], dtype=float)
        if hasattr(self, 'last_peak_fit_profile'):
            del self.last_peak_fit_profile
        if hasattr(self, 'last_iad_mc_sample'):
            del self.last_iad_mc_sample
        self._clear_spectrum()
        self._clear_peak_fit_results()
        self._update_spectrum_header()
        if self.processed_gap_mask is not None and self.processed_gap_mask.shape != self.immm.shape:
            self.processed_gap_mask = None
        if (
            self.processed_detector_background_mask is not None
            and self.processed_detector_background_mask.shape != self.immm.shape
        ):
            self.processed_detector_background_mask = None
        self.spect_processor = SpectrumProcessor(
            self.immm,
            self.parm['n_moveavg'],
            invalid_pixel_mask=self.processed_gap_mask,
            detector_background_mask=self.processed_detector_background_mask,
            edge_valid_fraction_threshold=gap_defaults.get('edge_valid_fraction', 0.80),
        )
        self.spect_processor.set_gap_correction_params(
            max_width=gap_defaults.get('max_width', 8),
            drop_fraction=gap_defaults.get('drop_fraction', 0.65),
        )
        self._set_spin_limits(self.immm.shape)
        self.display_image(self.immm, display_title)

    def display_image(self, data, title):
        self._set_image_title(title)
        self.image_item.setImage(
            np.asarray(data, dtype=float),
            autoLevels=False,
            levels=(self.vmin_spin.value(), self.vmax_spin.value()),
        )
        self.update_cmap()
        self.colorbar.setLevels((self.vmin_spin.value(), self.vmax_spin.value()))
        rows, cols = data.shape
        self.image_plot.setLimits(xMin=0, xMax=cols, yMin=0, yMax=rows)
        self.image_plot.setRange(xRange=(0, cols), yRange=(0, rows), padding=0.0)
        self.refresh_regions()

    def refresh_image(self):
        data = self.immm if self.immm is not None else self.imm
        if data is None:
            return
        self.image_item.setLevels((self.vmin_spin.value(), self.vmax_spin.value()))
        self.colorbar.setLevels((self.vmin_spin.value(), self.vmax_spin.value()))
        self.refresh_regions()

    def refresh_regions(self):
        if not hasattr(self, 'image_plot'):
            return
        for item in self._image_region_items:
            self.image_plot.removeItem(item)
        self._image_region_items = []
        data = self.immm if self.immm is not None else self.imm
        if data is None:
            return

        col_begin, col_end = self.col_range()
        x0 = float(col_begin)
        x1 = float(col_end + 1)
        for begin, end in self.roi_ranges():
            self._add_rect_region(
                x0,
                float(begin),
                x1,
                float(end + 1),
                brush_color=None,
                pen_color=QtGui.QColor(255, 255, 255, 230),
                dashed=True,
                width=1.2,
                z_value=34,
            )
        for begin, end in self.bg_ranges():
            self._add_rect_region(
                x0,
                float(begin),
                x1,
                float(end + 1),
                brush_color=QtGui.QColor(245, 245, 245, 78),
                pen_color=None,
                dashed=False,
                width=0.0,
                z_value=28,
            )


    def _add_region(self, values, horizontal, color, pen):
        orientation = pg.LinearRegionItem.Horizontal if horizontal else pg.LinearRegionItem.Vertical
        region = pg.LinearRegionItem(
            values=values,
            orientation=orientation,
            movable=False,
            brush=QtGui.QBrush(QtGui.QColor(*color)),
            pen=pg.mkPen(pen, width=1),
        )
        region.setZValue(30)
        self.image_plot.addItem(region)
        self._image_region_items.append(region)

    def _add_rect_region(self, x0, y0, x1, y1, brush_color, pen_color, dashed=False, width=1.0, z_value=30):
        rect = QtWidgets.QGraphicsRectItem(QtCore.QRectF(x0, y0, max(x1 - x0, 0.0), max(y1 - y0, 0.0)))
        if brush_color is None:
            rect.setBrush(QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush))
        else:
            rect.setBrush(QtGui.QBrush(brush_color))
        if pen_color is None or width <= 0:
            pen = QtGui.QPen(QtGui.QColor(0, 0, 0, 0))
            pen.setStyle(QtCore.Qt.PenStyle.NoPen)
        else:
            pen = QtGui.QPen(pen_color)
            pen.setWidthF(float(width))
            if dashed:
                pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        rect.setPen(pen)
        rect.setZValue(z_value)
        self.image_plot.addItem(rect)
        self._image_region_items.append(rect)

    def roi_ranges(self):
        ranges = []
        begin, end = _safe_int_pair(self.roi_row_start.value(), self.roi_row_end.value())
        if end >= begin:
            ranges.append((begin, end))
        if self.roi2_check.isChecked():
            begin, end = _safe_int_pair(self.roi2_row_start.value(), self.roi2_row_end.value())
            if end >= begin:
                ranges.append((begin, end))
        return ranges

    def bg_ranges(self):
        ranges = []
        for first, second in (
            (self.bg_row_start, self.bg_row_end),
            (self.bg2_row_start, self.bg2_row_end),
        ):
            begin, end = _safe_int_pair(first.value(), second.value())
            if end > begin:
                ranges.append((begin, end))
        return ranges

    def col_range(self):
        return _safe_int_pair(self.roi_col_start.value(), self.roi_col_end.value())

    def auto_background(self):
        data = self.immm
        if data is None:
            QtWidgets.QMessageBox.information(self, "Auto BG", "Import an image before selecting background.")
            return
        roi_ranges = self.roi_ranges()
        if not roi_ranges:
            return
        row_begin = min(begin for begin, _end in roi_ranges)
        row_end = max(end for _begin, end in roi_ranges)
        col_begin, col_end = self.col_range()
        row_count = data.shape[0]
        height = max(max(end - begin + 1 for begin, end in roi_ranges), 1)
        separation = max(3, int(round(height * 0.35)))
        search_margin = max(60, height * 4)
        roi_center = (row_begin + row_end) / 2.0
        invalid = self.processed_gap_mask
        above_min = max(0, row_begin - search_margin - height)
        above_max = max(-1, row_begin - separation - height)
        below_min = min(row_count - height, row_end + separation + 1)
        below_max = min(row_count - height, row_end + separation + search_margin)
        above_candidates = range(above_min, above_max + 1) if above_max >= above_min else []
        below_candidates = range(below_min, below_max + 1) if below_max >= below_min else []
        above = _best_background_band(data, invalid, above_candidates, height, col_begin, col_end, roi_center)
        below = _best_background_band(data, invalid, below_candidates, height, col_begin, col_end, roi_center)
        if above is None and below is None:
            QtWidgets.QMessageBox.information(self, "Auto BG", "Could not find a suitable local background strip.")
            return
        if above is None:
            above = below
        if below is None:
            below = above
        self._set_spin_values(
            {
                self.bg_row_start: above[0],
                self.bg_row_end: above[1],
                self.bg2_row_start: below[0],
                self.bg2_row_end: below[1],
            }
        )

    def configure_processor(self):
        if self.spect_processor is None:
            return
        gap_defaults = self.parm['ccd_gap']
        self.spect_processor.toggle_gap_correction(self._gap_correction_enabled())
        self.spect_processor.set_gap_correction_params(
            max_width=gap_defaults.get('max_width', 8),
            drop_fraction=gap_defaults.get('drop_fraction', 0.65),
        )
        self.spect_processor.toggle_background_subtraction(self._background_removal_enabled())
        if self._background_removal_enabled():
            col_begin, col_end = self.col_range()
            bg_rois = [
                (row_begin, row_end + 1, col_begin, col_end + 1)
                for row_begin, row_end in self.bg_ranges()
            ]
            self.spect_processor.set_background_rois(bg_rois)

    def plot_spectrum(self):
        if self.spect_processor is None:
            QtWidgets.QMessageBox.information(self, "Plot", "Import an image before plotting a spectrum.")
            return
        try:
            self.configure_processor()
            col_begin, col_end = self.col_range()
            rois = [
                (row_begin, row_end + 1, col_begin, col_end + 1)
                for row_begin, row_end in self.roi_ranges()
            ]
            spectrum = self.spect_processor.get_spectrum_for_rois(rois)
            y_data = _normalize_spectrum(spectrum)
            x_data = np.arange(col_begin, col_begin + y_data.size, dtype=float)
            self.last_x = x_data
            self.last_y = y_data
            raw_spectrum = getattr(self.spect_processor, 'last_uncorrected_spectrum', spectrum)
            raw_y = _normalize_spectrum(raw_spectrum)
            self.last_raw_y = raw_y if raw_y.size == y_data.size else y_data.copy()
            self.last_raw_x = np.arange(col_begin, col_begin + self.last_raw_y.size, dtype=float)
            view_label = "Corrected" if (self._gap_correction_enabled() or self._background_removal_enabled()) else "Raw"
            self._set_spectrum_view_mode("gap" if (self._gap_correction_enabled() or self._background_removal_enabled()) else "raw")
            self._draw_spectrum(x_data, y_data, label=view_label, show_gap_markers=False, view_label=view_label)
            self.statusBar().showMessage(f"Plotted spectrum with {len(rois)} ROI strip(s)")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Plot", str(exc))

    def _clear_spectrum(self):
        for item in self._spectrum_items:
            self.spectrum_plot.removeItem(item)
        self._spectrum_items = []
        if self._legend is not None:
            self._legend.clear()
            self.spectrum_plot.removeItem(self._legend)
            self._legend = None

    def _set_spectrum_view_range(self, x_data, y_arrays):
        finite_x = np.asarray(x_data, dtype=float)
        finite_x = finite_x[np.isfinite(finite_x)]
        finite_y_parts = []
        for y_values in y_arrays:
            values = np.asarray(y_values, dtype=float)
            finite_y_parts.append(values[np.isfinite(values)])
        finite_y_parts = [values for values in finite_y_parts if values.size]
        if not finite_x.size or not finite_y_parts:
            return
        finite_y = np.concatenate(finite_y_parts)
        x_min, x_max = float(np.nanmin(finite_x)), float(np.nanmax(finite_x))
        y_min, y_max = float(np.nanmin(finite_y)), float(np.nanmax(finite_y))
        x_pad = max((x_max - x_min) * 0.03, 1.0)
        y_pad = max((y_max - y_min) * 0.10, 1e-6)
        self.spectrum_plot.getViewBox().setLimits(
            xMin=x_min - x_pad,
            xMax=x_max + x_pad,
            yMin=y_min - y_pad,
            yMax=y_max + y_pad,
        )
        self.spectrum_plot.setRange(
            xRange=(x_min - x_pad, x_max + x_pad),
            yRange=(y_min - y_pad, y_max + y_pad),
            padding=0.0,
        )

    def _gap_mask(self, attr_name, expected_len):
        mask = getattr(self.spect_processor, attr_name, None)
        if mask is None:
            return np.zeros(expected_len, dtype=bool)
        mask = np.asarray(mask, dtype=bool)
        if mask.size != expected_len:
            return np.zeros(expected_len, dtype=bool)
        return mask

    def _add_gap_tick_group(self, x_data, mask, y0, y1, color, label):
        indices = np.flatnonzero(np.asarray(mask, dtype=bool))
        if indices.size == 0:
            return
        xs = np.empty(indices.size * 3, dtype=float)
        ys = np.empty(indices.size * 3, dtype=float)
        xs[0::3] = x_data[indices]
        xs[1::3] = x_data[indices]
        xs[2::3] = np.nan
        ys[0::3] = y0
        ys[1::3] = y1
        ys[2::3] = np.nan
        ticks = pg.PlotDataItem(xs, ys, pen=pg.mkPen(color, width=1.4))
        self.spectrum_plot.addItem(ticks)
        self._spectrum_items.append(ticks)
        self._legend.addItem(ticks, label)

    def _plot_gap_ticks(self, x_data, y_data):
        if self.spect_processor is None:
            return
        expected_len = len(y_data)
        combined = self._gap_mask('last_gap_mask', expected_len)
        if not np.any(combined):
            return
        roi_mask = self._gap_mask('last_roi_gap_mask', expected_len) | self._gap_mask('last_roi_edge_mask', expected_len)
        bg_mask = self._gap_mask('last_background_gap_mask', expected_len) | self._gap_mask('last_background_edge_mask', expected_len)
        if not np.any(roi_mask) and not np.any(bg_mask):
            roi_mask = combined
        both_mask = roi_mask & bg_mask
        roi_only = roi_mask & ~bg_mask
        bg_only = bg_mask & ~roi_mask
        other_mask = combined & ~(roi_mask | bg_mask)

        finite = y_data[np.isfinite(y_data)]
        if finite.size:
            y_min = float(np.nanmin(finite))
            y_max = float(np.nanmax(finite))
            span = max(y_max - y_min, np.finfo(float).eps)
            y0 = y_min - 0.075 * span
            y1 = y_min - 0.025 * span
        else:
            y0, y1 = -0.01, 0.0

        self._add_gap_tick_group(x_data, roi_only, y0, y1, QtGui.QColor(255, 150, 150, 210), "Gap fill")
        self._add_gap_tick_group(x_data, bg_only, y0, y1, QtGui.QColor(255, 199, 120, 220), "BG gap fill")
        self._add_gap_tick_group(x_data, both_mask, y0, y1, QtGui.QColor(205, 175, 255, 220), "ROI+BG gap fill")
        self._add_gap_tick_group(x_data, other_mask, y0, y1, QtGui.QColor(255, 175, 175, 190), "CCD gap fill")

    def _draw_spectrum(self, x_data, y_data, label=None, show_gap_markers=False, view_label=None):
        self._clear_spectrum()
        self.spectrum_plot.setLabel('bottom', 'Column Index', **{'font-size': '13pt', 'font-family': 'Arial'})
        self.spectrum_plot.setLabel('left', 'Normalized intensity', **{'font-size': '13pt', 'font-family': 'Arial'})
        self._legend = self._new_spectrum_legend()

        line = pg.PlotDataItem(x_data, y_data, pen=self._selected_line_pen())
        self.spectrum_plot.addItem(line)
        self._spectrum_items.append(line)
        legend_label = self.run_label.text()
        if label:
            legend_label = f"{label}, {legend_label}"
        self._legend.addItem(line, legend_label)

        self._set_spectrum_view_range(x_data, [y_data])
        self._update_spectrum_header(view_label)

    def save_spectrum_png(self):
        if self.last_x.size == 0:
            QtWidgets.QMessageBox.information(self, "Save PNG", "Plot a spectrum first.")
            return
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save spectrum PNG",
            self._default_save_path(f"{self._run_basename()}_spectrum.png"),
            "PNG image (*.png)",
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path = f"{path}.png"
        exporter = ImageExporter(self.spectrum_plot)
        exporter.parameters()['width'] = max(int(self.spectrum_plot.width()), 2400)
        exporter.export(path)

    def save_spectrum_svg(self):
        if self.last_x.size == 0:
            QtWidgets.QMessageBox.information(self, "Export Image", "Plot a spectrum first.")
            return
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export spectrum image",
            self._default_save_path(f"{self._run_basename()}_spectrum.svg"),
            "SVG vector (*.svg)",
        )
        if not path:
            return
        if not path.lower().endswith(".svg"):
            path = f"{path}.svg"
        SVGExporter(self.spectrum_plot).export(path)


def run_gui():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    window = QtXESAnalyzer()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    run_gui()
