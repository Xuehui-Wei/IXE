# -*- coding: utf-8 -*-
"""Peak fitting helpers for IXE IAD calculations.

The fitting model follows the constrained pseudo-Voigt peak workflow used by
VibFit, adapted here for 1D IXE spectra without the VibFit Qt/Hyperspy stack.
"""

import numpy as np
from scipy.optimize import curve_fit

try:
    from scipy.signal import find_peaks
except Exception:  # pragma: no cover - scipy.signal is expected with scipy
    find_peaks = None


_EPS = np.finfo(float).eps


class PeakFitError(ValueError):
    """Raised when a spectrum cannot be fitted with the requested setup."""


def _normalize_peak_shape(peak_shape):
    value = str(peak_shape or "pseudo_voigt").strip().lower().replace("-", "_").replace(" ", "_")
    if value in ("pseudo_voigt", "pseudovoigt", "pseudo", "voigt"):
        return "pseudo_voigt"
    if value in ("lorentzian", "lorentz"):
        return "lorentzian"
    raise PeakFitError("Peak model must be pseudo-Voigt or Lorentzian.")


class PeakFitConfig:
    def __init__(self, n_peaks=3, x_min=None, x_max=None, model="kbeta", peak_shape="pseudo_voigt"):
        self.n_peaks = int(n_peaks)
        self.x_min = x_min
        self.x_max = x_max
        self.model = model
        self.peak_shape = _normalize_peak_shape(peak_shape)
        if self.n_peaks <= 0:
            raise PeakFitError("Peak count must be a positive integer.")


class PeakFitResult:
    def __init__(
        self,
        x,
        raw_y,
        best_fit,
        normalized_fit,
        baseline,
        peak_curves,
        centers,
        sigmas,
        amplitudes,
        fractions,
        peak_labels=None,
        widths=None,
        peak_shape="pseudo_voigt",
    ):
        self.x = x
        self.raw_y = raw_y
        self.best_fit = best_fit
        self.normalized_fit = normalized_fit
        self.baseline = baseline
        self.peak_curves = peak_curves
        self.centers = centers
        self.sigmas = sigmas
        self.amplitudes = amplitudes
        self.fractions = fractions
        self.peak_labels = peak_labels or [f"Peak {index + 1}" for index in range(len(centers))]
        self.widths = widths if widths is not None else [2.0 * float(value) for value in sigmas]
        self.peak_shape = _normalize_peak_shape(peak_shape)


class FittedIADResult:
    def __init__(self, x, roi_fit, ref_fit, roi_raw, ref_raw, roi_result, ref_result, iad_value):
        self.x = x
        self.roi_fit = roi_fit
        self.ref_fit = ref_fit
        self.roi_raw = roi_raw
        self.ref_raw = ref_raw
        self.roi_result = roi_result
        self.ref_result = ref_result
        self.iad_value = iad_value


def _normalize_spectrum(y):
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        raise PeakFitError("Cannot normalize an empty spectrum.")
    shifted = y - np.nanmin(y)
    total = np.nansum(shifted)
    if not np.isfinite(total) or abs(total) <= _EPS:
        total = np.nansum(np.abs(y))
        if not np.isfinite(total) or abs(total) <= _EPS:
            raise PeakFitError("Cannot normalize a flat or invalid fitted spectrum.")
        return y / total
    return shifted / total


def _match_data_scale(model_y, data_y):
    model_y = np.asarray(model_y, dtype=float)
    data_y = np.asarray(data_y, dtype=float)
    finite = np.isfinite(model_y) & np.isfinite(data_y)
    if finite.sum() < 3:
        return _normalize_spectrum(model_y)

    model = model_y[finite]
    data = data_y[finite]
    data_min = float(np.nanmin(data))
    data_max = float(np.nanmax(data))
    data_range = max(data_max - data_min, _EPS)
    edge_count = max(3, int(round(data.size * 0.10)))
    tail_level = float(np.nanmedian(np.r_[data[:edge_count], data[-edge_count:]]))

    peak_weight = np.clip((data - tail_level) / data_range, 0.0, 1.0)
    weights = 1.0 + 1.5 * peak_weight
    weights[:edge_count] += 2.5
    weights[-edge_count:] += 2.5

    x_scaled = np.linspace(-1.0, 1.0, model.size)
    design = np.column_stack([model, x_scaled, np.ones_like(model)])
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_data = data * np.sqrt(weights)
    try:
        scale, slope, offset = np.linalg.lstsq(weighted_design, weighted_data, rcond=None)[0]
    except np.linalg.LinAlgError:
        return _normalize_spectrum(model_y)
    if not np.isfinite(scale) or not np.isfinite(slope) or not np.isfinite(offset):
        return _normalize_spectrum(model_y)

    full_x_scaled = np.linspace(-1.0, 1.0, model_y.size)
    calibrated = model_y * scale + slope * full_x_scaled + offset
    calibrated = np.maximum(calibrated, min(data_min, tail_level))

    calibrated_finite = calibrated[finite]
    left_delta = float(np.nanmedian(data[:edge_count]) - np.nanmedian(calibrated_finite[:edge_count]))
    right_delta = float(np.nanmedian(data[-edge_count:]) - np.nanmedian(calibrated_finite[-edge_count:]))
    tail_correction = np.interp(full_x_scaled, [-1.0, 1.0], [left_delta, right_delta])
    calibrated = calibrated + tail_correction

    calibrated_finite = calibrated[finite]
    calibrated_peak = float(np.nanmax(calibrated_finite))
    if calibrated_peak > data_max and calibrated_peak > tail_level:
        excess = calibrated_peak - data_max
        peak_shape = np.clip((data_y - tail_level) / data_range, 0.0, 1.0) ** 2
        calibrated = calibrated - excess * peak_shape

    calibrated_finite = calibrated[finite]
    left_delta = float(np.nanmedian(data[:edge_count]) - np.nanmedian(calibrated_finite[:edge_count]))
    right_delta = float(np.nanmedian(data[-edge_count:]) - np.nanmedian(calibrated_finite[-edge_count:]))
    tail_correction = np.interp(full_x_scaled, [-1.0, 1.0], [left_delta, right_delta])
    calibrated = calibrated + tail_correction
    calibrated = np.maximum(calibrated, min(data_min, tail_level))
    calibrated_peak = float(np.nanmax(calibrated[finite]))
    if calibrated_peak > data_max:
        excess = calibrated_peak - data_max
        peak_shape = np.clip((data_y - tail_level) / data_range, 0.0, 1.0) ** 2
        calibrated = calibrated - excess * peak_shape
        calibrated = np.maximum(calibrated, min(data_min, tail_level))
    return calibrated


def _clean_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise PeakFitError("X and Y arrays must have the same length.")
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 5:
        raise PeakFitError("Fit range must contain at least 5 finite points.")
    order = np.argsort(x)
    return x[order], y[order]


def _fit_mask(x, config):
    mask = np.ones_like(x, dtype=bool)
    if config.x_min is not None:
        mask &= x >= float(config.x_min)
    if config.x_max is not None:
        mask &= x <= float(config.x_max)
    if mask.sum() < max(5, config.n_peaks * 4 + 2):
        raise PeakFitError("Fit range is too small for the requested number of peaks.")
    return mask


def _x_step(x):
    if x.size < 2:
        return 1.0
    diffs = np.diff(np.unique(x))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 1.0
    return float(np.nanmedian(diffs))


def _pseudo_voigt(x, amplitude, center, sigma, fraction):
    sigma = max(float(sigma), _EPS)
    fraction = min(max(float(fraction), 0.0), 1.0)
    gaussian = np.exp(-0.5 * ((x - center) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    lorentzian = (sigma / np.pi) / ((x - center) ** 2 + sigma ** 2)
    return amplitude * ((1.0 - fraction) * gaussian + fraction * lorentzian)


def _lorentzian(x, amplitude, center, sigma):
    sigma = max(float(sigma), _EPS)
    return amplitude * (sigma / np.pi) / ((x - center) ** 2 + sigma ** 2)


def _peak_param_count(peak_shape):
    return 3 if _normalize_peak_shape(peak_shape) == "lorentzian" else 4


def _initial_centers(x, y, n_peaks):
    span = float(np.nanmax(x) - np.nanmin(x))
    y_range = float(np.nanmax(y) - np.nanmin(y))
    centers = []

    if find_peaks is not None and y.size >= 3:
        distance = max(1, int(y.size / max(n_peaks * 2, 1)))
        prominence = max(y_range * 0.03, _EPS)
        peak_indices, _props = find_peaks(y, prominence=prominence, distance=distance)
        if peak_indices.size:
            strongest = peak_indices[np.argsort(y[peak_indices])[-n_peaks:]]
            centers.extend(float(value) for value in x[np.sort(strongest)])

    if len(centers) < n_peaks:
        quantiles = np.linspace(0.18, 0.82, n_peaks)
        for value in np.quantile(x, quantiles):
            centers.append(float(value))

    unique_centers = []
    min_sep = max(span / max(n_peaks * 12.0, 1.0), _x_step(x))
    for center in sorted(centers):
        if not unique_centers or abs(center - unique_centers[-1]) >= min_sep:
            unique_centers.append(center)
        if len(unique_centers) == n_peaks:
            break

    while len(unique_centers) < n_peaks:
        fraction = (len(unique_centers) + 1.0) / (n_peaks + 1.0)
        unique_centers.append(float(np.nanmin(x) + span * fraction))

    return sorted(unique_centers[:n_peaks])


def _moving_average_for_initial(y):
    y = np.asarray(y, dtype=float)
    if y.size < 7:
        return y
    window = max(5, int(round(y.size * 0.035)))
    if window % 2 == 0:
        window += 1
    window = min(window, y.size if y.size % 2 == 1 else y.size - 1)
    if window < 3:
        return y
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(y, kernel, mode="same")


def _right_half_width(x, y, peak_index):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    peak_y = float(y[peak_index])
    edge_count = max(3, int(round(y.size * 0.08)))
    edge_level = float(np.nanmedian(np.r_[y[:edge_count], y[-edge_count:]]))
    if peak_y <= edge_level:
        return max(_x_step(x), (float(np.nanmax(x)) - float(np.nanmin(x))) * 0.05)
    half_level = edge_level + 0.5 * (peak_y - edge_level)
    right = peak_index + 1
    while right < y.size and y[right] > half_level:
        right += 1
    if right >= y.size:
        return max(_x_step(x), (float(np.nanmax(x)) - float(np.nanmin(x))) * 0.05)
    left_x = x[right - 1]
    right_x = x[right]
    left_y = y[right - 1]
    right_y = y[right]
    if abs(right_y - left_y) <= _EPS:
        crossing = right_x
    else:
        fraction = (half_level - left_y) / (right_y - left_y)
        crossing = left_x + fraction * (right_x - left_x)
    return max(float(crossing - x[peak_index]), _x_step(x))


def _clamp(value, lower, upper):
    if lower > upper:
        lower, upper = upper, lower
    return min(max(float(value), float(lower)), float(upper))


def _append_peak_params(p0, lower, upper, amplitude, center, sigma, fraction, bounds, peak_shape="pseudo_voigt"):
    amp_lower, amp_upper, center_lower, center_upper, sigma_lower, sigma_upper, fraction_lower, fraction_upper = bounds
    if _normalize_peak_shape(peak_shape) == "lorentzian":
        p0.extend(
            [
                _clamp(amplitude, amp_lower, amp_upper),
                _clamp(center, center_lower, center_upper),
                _clamp(sigma, sigma_lower, sigma_upper),
            ]
        )
        lower.extend([amp_lower, center_lower, sigma_lower])
        upper.extend([amp_upper, center_upper, sigma_upper])
        return

    p0.extend(
        [
            _clamp(amplitude, amp_lower, amp_upper),
            _clamp(center, center_lower, center_upper),
            _clamp(sigma, sigma_lower, sigma_upper),
            _clamp(fraction, fraction_lower, fraction_upper),
        ]
    )
    lower.extend([amp_lower, center_lower, sigma_lower, fraction_lower])
    upper.extend([amp_upper, center_upper, sigma_upper, fraction_upper])


def _build_kbeta_initial_params_and_bounds(x, y, peak_shape="pseudo_voigt"):
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    span = max(x_max - x_min, _x_step(x))
    x_mid = x_min + span / 2.0
    step = _x_step(x)
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    y_range = max(y_max - y_min, _EPS)
    y_abs_scale = max(float(np.nanmax(np.abs(y))), y_range, _EPS)

    p0 = [y_min, 0.0]
    lower = [y_min - 5.0 * y_abs_scale, -10.0 * y_abs_scale]
    upper = [y_max + 5.0 * y_abs_scale, 10.0 * y_abs_scale]

    y_smooth = _moving_average_for_initial(y)
    main_index = int(np.nanargmax(y_smooth))
    main_center = float(x[main_index])
    main_right_half_width = _right_half_width(x, y_smooth, main_index)
    satellite_region = x < (main_center - 0.12 * span)
    if np.any(satellite_region):
        sat_indices = np.flatnonzero(satellite_region)
        sat_index = int(sat_indices[np.nanargmax(y_smooth[satellite_region])])
        satellite_center = float(x[sat_index])
    else:
        satellite_center = x_min + 0.35 * span

    if satellite_center >= main_center:
        satellite_center = x_min + 0.35 * span
        main_center = x_min + 0.78 * span

    sat_to_main = max(main_center - satellite_center, span * 0.30)
    residual_center = satellite_center + 0.70 * sat_to_main
    residual_center = min(residual_center, main_center - 0.07 * span)

    main_low = max(x_min, main_center - max(main_right_half_width * 1.2, 0.025 * span))
    main_high = min(x_max, main_center + max(main_right_half_width * 0.8, 0.020 * span))
    satellite_low = x_min
    satellite_high = min(x_max, main_center - 0.14 * span)
    residual_low = max(x_min, satellite_center + 0.15 * sat_to_main)
    residual_high = min(x_max, main_center - 0.05 * span)
    if satellite_low >= satellite_high or residual_low >= residual_high or main_low >= main_high:
        return _build_generic_initial_params_and_bounds(x, y, 3)

    amp_upper = y_abs_scale * span * 20.0
    sat_height = max(float(np.interp(satellite_center, x, y) - y_min), y_range * 0.12)
    res_height = max(float(np.interp(residual_center, x, y) - y_min), y_range * 0.18)
    main_height = max(float(y[main_index] - y_min), y_range)

    # The center bounds keep the fitted components in the K-beta physical order:
    # Kbeta prime satellite, Kbeta_res shoulder, and Kbeta1,3 main peak.
    _append_peak_params(
        p0,
        lower,
        upper,
        amplitude=sat_height * max(0.16 * sat_to_main, step) * np.sqrt(2.0 * np.pi),
        center=satellite_center,
        sigma=max(0.12 * sat_to_main, step * 3.0),
        fraction=0.55,
        bounds=(
            0.0,
            amp_upper,
            satellite_low,
            satellite_high,
            max(step * 2.0, span * 0.025),
            max(step * 3.0, span * 0.35),
            0.0,
            1.0,
        ),
        peak_shape=peak_shape,
    )
    _append_peak_params(
        p0,
        lower,
        upper,
        amplitude=res_height * max(0.10 * sat_to_main, step) * np.sqrt(2.0 * np.pi),
        center=residual_center,
        sigma=max(0.07 * sat_to_main, step * 2.0),
        fraction=0.45,
        bounds=(
            0.0,
            amp_upper,
            residual_low,
            residual_high,
            max(step * 1.5, span * 0.018),
            max(step * 2.5, span * 0.25),
            0.0,
            1.0,
        ),
        peak_shape=peak_shape,
    )
    _append_peak_params(
        p0,
        lower,
        upper,
        amplitude=main_height * max(main_right_half_width / 1.25, step) * np.sqrt(2.0 * np.pi),
        center=main_center,
        sigma=max(main_right_half_width / 1.3, step * 1.5),
        fraction=0.35,
        bounds=(
            0.0,
            amp_upper,
            main_low,
            main_high,
            max(step * 0.8, main_right_half_width * 0.45),
            max(step * 1.2, main_right_half_width * 1.35),
            0.0,
            0.85,
        ),
        peak_shape=peak_shape,
    )
    return p0, lower, upper, x_mid, span, [
        r"$K_{\beta'}$",
        r"$K_{\beta,\mathrm{res}}$",
        r"$K_{\beta_{1,3}}$",
    ]


def _build_generic_initial_params_and_bounds(x, y, n_peaks, peak_shape="pseudo_voigt"):
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    span = max(x_max - x_min, _x_step(x))
    x_mid = x_min + span / 2.0
    step = _x_step(x)
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    y_range = max(y_max - y_min, _EPS)
    y_abs_scale = max(float(np.nanmax(np.abs(y))), y_range, _EPS)
    sigma_guess = max(span / max(n_peaks * 6.0, 6.0), step)

    p0 = [y_min, 0.0]
    lower = [y_min - 5.0 * y_abs_scale, -10.0 * y_abs_scale]
    upper = [y_max + 5.0 * y_abs_scale, 10.0 * y_abs_scale]

    centers = _initial_centers(x, y, n_peaks)
    for center in centers:
        local_index = int(np.nanargmin(np.abs(x - center)))
        local_height = max(float(y[local_index] - y_min), y_range / max(n_peaks, 1))
        amplitude_guess = max(local_height * sigma_guess * np.sqrt(2.0 * np.pi), _EPS)
        if _normalize_peak_shape(peak_shape) == "lorentzian":
            p0.extend([amplitude_guess, center, sigma_guess])
            lower.extend([0.0, x_min, max(step * 0.25, _EPS)])
            upper.extend([y_abs_scale * span * 20.0, x_max, span])
        else:
            p0.extend([amplitude_guess, center, sigma_guess, 0.5])
            lower.extend([0.0, x_min, max(step * 0.25, _EPS), 0.0])
            upper.extend([y_abs_scale * span * 20.0, x_max, span, 1.0])

    return p0, lower, upper, x_mid, span, [f"Peak {index + 1}" for index in range(n_peaks)]


def _build_initial_params_and_bounds(x, y, config):
    n_peaks = int(config.n_peaks)
    if str(getattr(config, "model", "")).lower() == "kbeta" and n_peaks == 3:
        return _build_kbeta_initial_params_and_bounds(x, y, config.peak_shape)
    return _build_generic_initial_params_and_bounds(x, y, n_peaks, config.peak_shape)


def _evaluate_model(x, params, n_peaks, x_mid, span, peak_shape="pseudo_voigt"):
    peak_shape = _normalize_peak_shape(peak_shape)
    intercept = params[0]
    slope = params[1]
    baseline = intercept + slope * ((x - x_mid) / max(span, _EPS))
    peak_curves = []
    y = np.asarray(baseline, dtype=float).copy()
    idx = 2
    for _peak_index in range(n_peaks):
        if peak_shape == "lorentzian":
            curve = _lorentzian(x, params[idx], params[idx + 1], params[idx + 2])
            idx += 3
        else:
            curve = _pseudo_voigt(x, params[idx], params[idx + 1], params[idx + 2], params[idx + 3])
            idx += 4
        peak_curves.append(curve)
        y = y + curve
    return y, baseline, peak_curves


def _curve_fwhm(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or y.size != x.size or not np.any(np.isfinite(y)):
        return np.nan
    peak_index = int(np.nanargmax(y))
    peak_height = float(y[peak_index])
    if not np.isfinite(peak_height) or peak_height <= _EPS:
        return np.nan
    half_height = peak_height / 2.0

    def crossing_x(x0, y0, x1, y1):
        if abs(y1 - y0) <= _EPS:
            return float(x0)
        fraction = (half_height - y0) / (y1 - y0)
        return float(x0 + fraction * (x1 - x0))

    left_cross = np.nan
    for index in range(peak_index, 0, -1):
        if y[index - 1] <= half_height <= y[index] or y[index] <= half_height <= y[index - 1]:
            left_cross = crossing_x(x[index - 1], y[index - 1], x[index], y[index])
            break

    right_cross = np.nan
    for index in range(peak_index, y.size - 1):
        if y[index] >= half_height >= y[index + 1] or y[index + 1] >= half_height >= y[index]:
            right_cross = crossing_x(x[index], y[index], x[index + 1], y[index + 1])
            break

    if not np.isfinite(left_cross) or not np.isfinite(right_cross):
        return np.nan
    return max(right_cross - left_cross, _x_step(x))


def _fit_sigma(y):
    y = np.asarray(y, dtype=float)
    shifted = y - np.nanmin(y)
    y_range = max(float(np.nanmax(shifted) - np.nanmin(shifted)), _EPS)
    relative_weight = 0.75 + 0.25 * shifted / y_range
    edge_count = max(3, int(round(y.size * 0.10)))
    relative_weight[:edge_count] += 0.45
    relative_weight[-edge_count:] += 0.45
    return 1.0 / np.sqrt(relative_weight)


def fit_spectrum(x, y, config=None):
    """Fit a 1D spectrum with constrained pseudo-Voigt or Lorentzian peaks."""
    if config is None:
        config = PeakFitConfig()
    x, y = _clean_xy(x, y)
    mask = _fit_mask(x, config)
    x_fit = x[mask]
    y_fit = y[mask]
    n_peaks = int(config.n_peaks)
    p0, lower, upper, x_mid, span, peak_labels = _build_initial_params_and_bounds(x_fit, y_fit, config)

    def model_func(x_values, *params):
        model_y, _baseline, _peaks = _evaluate_model(x_values, params, n_peaks, x_mid, span, config.peak_shape)
        return model_y

    try:
        popt, _pcov = curve_fit(
            model_func,
            x_fit,
            y_fit,
            p0=p0,
            bounds=(lower, upper),
            sigma=_fit_sigma(y_fit),
            maxfev=40000,
        )
    except Exception as exc:
        raise PeakFitError("Peak fitting failed: {0}".format(exc))

    best_fit, baseline, peak_curves = _evaluate_model(x_fit, popt, n_peaks, x_mid, span, config.peak_shape)
    normalized_fit = _match_data_scale(best_fit, y_fit)
    centers = []
    sigmas = []
    amplitudes = []
    fractions = []
    widths = []
    idx = 2
    for _peak_index in range(n_peaks):
        amplitudes.append(float(popt[idx]))
        centers.append(float(popt[idx + 1]))
        sigmas.append(float(popt[idx + 2]))
        if config.peak_shape == "lorentzian":
            fractions.append(1.0)
            idx += 3
        else:
            fractions.append(float(popt[idx + 3]))
            idx += 4

    for curve, sigma in zip(peak_curves, sigmas):
        width = _curve_fwhm(x_fit, curve)
        if not np.isfinite(width):
            width = 2.0 * float(sigma)
        widths.append(float(width))

    return PeakFitResult(
        x=x_fit,
        raw_y=y_fit,
        best_fit=best_fit,
        normalized_fit=normalized_fit,
        baseline=baseline,
        peak_curves=peak_curves,
        centers=centers,
        sigmas=sigmas,
        amplitudes=amplitudes,
        fractions=fractions,
        peak_labels=peak_labels,
        widths=widths,
        peak_shape=config.peak_shape,
    )


def calculate_fitted_iad(x, roi_y, ref_y, config=None):
    """Fit ROI/reference spectra and return IAD from the normalized fitted curves."""
    if config is None:
        config = PeakFitConfig()
    x_input = np.asarray(x, dtype=float)
    x_roi, roi_y = _clean_xy(x_input, roi_y)
    x_ref, ref_y = _clean_xy(x_input, ref_y)
    if x_roi.shape != x_ref.shape or not np.allclose(x_roi, x_ref):
        raise PeakFitError("ROI and reference spectra must share the same x-axis.")

    roi_result = fit_spectrum(x_roi, roi_y, config)
    ref_result = fit_spectrum(x_ref, ref_y, config)
    if roi_result.x.shape != ref_result.x.shape or not np.allclose(roi_result.x, ref_result.x):
        raise PeakFitError("ROI and reference fit domains do not match.")

    iad_value = float(np.sum(np.abs(roi_result.normalized_fit - ref_result.normalized_fit)))
    return FittedIADResult(
        x=roi_result.x,
        roi_fit=roi_result.normalized_fit,
        ref_fit=ref_result.normalized_fit,
        roi_raw=roi_result.raw_y,
        ref_raw=ref_result.raw_y,
        roi_result=roi_result,
        ref_result=ref_result,
        iad_value=iad_value,
    )
