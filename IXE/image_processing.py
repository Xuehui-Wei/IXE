# -*- coding: utf-8 -*-
import numpy as np
import re
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from skimage.filters import threshold_otsu
#from IXE.remove_cross import delete_zero_rows_and_columns

def _get_image_cmap(self):
    cmap_name = self.parm.get('cmap', 'viridis') if hasattr(self, 'parm') else 'viridis'
    if hasattr(self, 'image_cmap'):
        cmap_name = self.image_cmap.get().strip() or cmap_name
    try:
        plt.get_cmap(cmap_name)
    except ValueError:
        cmap_name = 'viridis'
        if hasattr(self, 'image_cmap'):
            self.image_cmap.set(cmap_name)
    if hasattr(self, 'parm'):
        self.parm['cmap'] = cmap_name
    return cmap_name

def _remove_artist(artist):
    if artist is None:
        return
    try:
        artist.remove()
    except (NotImplementedError, ValueError):
        pass

def _image_points(self, value):
    return float(value) * float(getattr(self, 'image_point_scale', 1.0))

def _style_image_colorbar(self):
    if not hasattr(self, 'cbar') or self.cbar is None:
        return
    self.cbar.locator = MaxNLocator(nbins=3)
    self.cbar.update_ticks()
    self.cbar.ax.tick_params(
        labelsize=_image_points(self, 10),
        width=_image_points(self, 0.3),
        length=_image_points(self, 1.4),
        pad=_image_points(self, 1.5),
    )
    self.cbar.outline.set_linewidth(_image_points(self, 0.25))
    for spine in self.cbar.ax.spines.values():
        spine.set_linewidth(_image_points(self, 0.25))
    for tick_label in self.cbar.ax.get_yticklabels():
        tick_label.set_fontname('Arial')
        tick_label.set_fontsize(_image_points(self, 10))

def _style_image_axis_ticks(self, ax):
    ax.tick_params(
        axis='both',
        which='major',
        labelsize=_image_points(self, 11),
        width=_image_points(self, 0.4),
        length=_image_points(self, 1.4),
        pad=_image_points(self, 2),
    )
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontname('Arial')
        tick_label.set_fontsize(_image_points(self, 11))

def _clear_loaded_image_state(self):
    if hasattr(self, 'reset_spectrum_state'):
        self.reset_spectrum_state()
    if hasattr(self, 'immm'):
        delattr(self, 'immm')
    for mask_name in ('raw_gap_mask', 'processed_gap_mask', 'processed_detector_background_mask'):
        if hasattr(self, mask_name):
            delattr(self, mask_name)
    roi_lines = list(getattr(self, 'roi_lines', []) or [])
    if roi_lines:
        for line in roi_lines:
            _remove_artist(line)
        self.roi_lines = []
    else:
        if hasattr(self, 'line_begin') and self.line_begin:
            _remove_artist(self.line_begin)
        if hasattr(self, 'line_end') and self.line_end:
            _remove_artist(self.line_end)
    self.line_begin = None
    self.line_end = None
    bg_rects = list(getattr(self, 'bg_rects', []) or [])
    if bg_rects:
        for rect in bg_rects:
            _remove_artist(rect)
        self.bg_rects = []
    elif hasattr(self, 'bg_rect') and self.bg_rect:
        _remove_artist(self.bg_rect)
    self.bg_rect = None

def _read_tiff_array(filepath):
    with Image.open(filepath) as image:
        return np.asarray(image, dtype=np.float32)

def _update_image_limits(self, image):
    if hasattr(self, 'update_roi_row_slider_limits'):
        self.update_roi_row_slider_limits(image.shape[0])
    if hasattr(self, 'update_bg_row_slider_limits'):
        self.update_bg_row_slider_limits(image.shape[0])
    if hasattr(self, 'update_roi_col_slider_limits'):
        self.update_roi_col_slider_limits(image.shape[1])

def _set_entry_text(entry, text):
    if entry is None:
        return
    entry.delete(0, tk.END)
    entry.insert(0, text)

def _set_run_labels(self, label):
    if hasattr(self, 'run_number_label'):
        self.run_number_label.config(text=label)
    if hasattr(self, 'run_number_label_spectrum'):
        self.run_number_label_spectrum.config(text=label)

def _run_label_from_paths(paths):
    run_numbers = []
    for path in paths:
        match = re.search(r"Run_(\d+)", path)
        if match:
            run_numbers.append(match.group(1))
    if not run_numbers:
        return None
    if len(paths) == 1:
        return f"Run: {run_numbers[0]}"
    if len(set(run_numbers)) == 1:
        return f"Run: {run_numbers[0]} stack ({len(paths)} images)"
    return f"Run: stack {run_numbers[0]}-{run_numbers[-1]} ({len(paths)} images)"

def _set_image_source_metadata(self, paths):
    paths = list(paths)
    self.image_stack_files = paths
    self.image_stack_count = len(paths)
    self.original_filepath = paths[0] if paths else ""
    path_text = self.original_filepath
    if len(paths) > 1:
        basenames = [path.rsplit('/', 1)[-1] for path in paths]
        preview = "; ".join(basenames[:3])
        if len(basenames) > 3:
            preview += f"; ... +{len(basenames) - 3} more"
        path_text = f"{len(paths)} TIFF stack: {preview}"
    _set_entry_text(getattr(self, 'file_path_entry', None), path_text)

    run_label = _run_label_from_paths(paths)
    if run_label:
        _set_run_labels(self, run_label)
    else:
        fallback = f"Run: stack ({len(paths)} images)" if len(paths) > 1 else "Run: Not Loaded"
        _set_run_labels(self, fallback)
        print("Error: Run number not found in filename.")

def _stack_display_note(self, title):
    stack_count = int(getattr(self, 'image_stack_count', 1) or 1)
    if stack_count <= 1:
        return title
    return f"{title} | Stack: {stack_count} images summed"

def load_tiff(self):
    """Load one TIFF file and display it in the image panel."""
    filepath = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tiff *.tif")])
    if not filepath:
        return
    try:
        image = _read_tiff_array(filepath)
        _clear_loaded_image_state(self)
        _set_image_source_metadata(self, [filepath])
        self.im = None
        self.imm = image
        _update_image_limits(self, self.imm)
        self.display_image(self.imm, self.ax_image, "Raw Image")
        self.canvas_image.draw()
    except Exception as e:
        messagebox.showwarning("Import TIFF", str(e))
        print(f"Error loading TIFF: {e}")

def load_tiff_stack(self):
    """Load multiple TIFF files, sum pixels, and display the stacked raw image."""
    filepaths = filedialog.askopenfilenames(filetypes=[("TIFF files", "*.tiff *.tif")])
    if not filepaths:
        return
    filepaths = list(filepaths)
    try:
        first_image = _read_tiff_array(filepaths[0])
        stacked_image = first_image.astype(np.float64, copy=True)
        expected_shape = first_image.shape
        for filepath in filepaths[1:]:
            image = _read_tiff_array(filepath)
            if image.shape != expected_shape:
                raise ValueError(
                    "All TIFF images in a stack must have the same pixel shape. "
                    f"{filepath} has {image.shape}, expected {expected_shape}."
                )
            stacked_image += image.astype(np.float64, copy=False)

        _clear_loaded_image_state(self)
        self.im = None
        self.imm = stacked_image
        _set_image_source_metadata(self, filepaths)
        _update_image_limits(self, self.imm)
        title = "Stacked Raw Image" if len(filepaths) > 1 else "Raw Image"
        self.display_image(self.imm, self.ax_image, title)
        self.canvas_image.draw()
        print(f"Loaded TIFF stack: {len(filepaths)} image(s), shape {self.imm.shape}")
    except Exception as e:
        messagebox.showwarning("Import TIFF Stack", str(e))
        print(f"Error loading TIFF stack: {e}")

def auto_threshold(self):
    """Calculate threshold using Otsu's method."""
    if hasattr(self, 'imm'):
        thresh = threshold_otsu(self.imm)
        self.parm['threshold'] = thresh

def remove_cross_raw(self):
    """Remove the zero-intensity cross from the raw image."""
    from IXE.remove_cross import delete_zero_rows_and_columns
    if not hasattr(self, 'imm'):
        print("Error: No image loaded. Import a TIFF first.")
        return
    try:
        self.imm = delete_zero_rows_and_columns(self.imm)
        for mask_name in ('raw_gap_mask', 'processed_gap_mask', 'processed_detector_background_mask'):
            if hasattr(self, mask_name):
                delattr(self, mask_name)
        if hasattr(self, 'reset_spectrum_state'):
            self.reset_spectrum_state()
        if hasattr(self, 'update_roi_row_slider_limits'):
            self.update_roi_row_slider_limits(self.imm.shape[0])
        if hasattr(self, 'update_bg_row_slider_limits'):
            self.update_bg_row_slider_limits(self.imm.shape[0])
        if hasattr(self, 'update_roi_col_slider_limits'):
            self.update_roi_col_slider_limits(self.imm.shape[1])
        self.display_image(self.imm, self.ax_image, "Raw Image (Cross Removed)")#
        self.canvas_image.draw()
    except Exception as e:
        print(f"Error removing cross: {e}")

def _short_runs(mask, max_width):
    mask = np.asarray(mask, dtype=bool)
    keep = np.zeros_like(mask, dtype=bool)
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        if start is not None and (not value or index == mask.size - 1):
            end = index if not value else index + 1
            if 0 < end - start <= max_width:
                keep[start:end] = True
            start = None
    return keep

def _true_runs(mask):
    mask = np.asarray(mask, dtype=bool)
    runs = []
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        if start is not None and (not value or index == mask.size - 1):
            end = index if not value else index + 1
            runs.append((start, end))
            start = None
    return runs

def _rolling_median(values, half_window):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values.copy()
    half_window = max(int(half_window), 1)
    padded = np.pad(values, half_window, mode='edge')
    return np.array([
        np.nanmedian(padded[index:index + 2 * half_window + 1])
        for index in range(values.size)
    ])

def _detect_dark_gap_axis(profile, max_width=16, drop_fraction=0.55, center_window=0.35):
    profile = np.asarray(profile, dtype=float)
    if profile.size < 5 or not np.any(np.isfinite(profile)):
        return np.zeros(profile.size, dtype=bool)

    fill_value = float(np.nanmedian(profile[np.isfinite(profile)]))
    profile = np.nan_to_num(profile, nan=fill_value, posinf=fill_value, neginf=fill_value)
    percentile_span = float(np.nanpercentile(profile, 95) - np.nanpercentile(profile, 5))
    full_span = float(np.nanmax(profile) - np.nanmin(profile))
    span = max(percentile_span, full_span)
    if not np.isfinite(span) or span <= np.finfo(float).eps:
        return np.zeros(profile.size, dtype=bool)

    half_window = max(max_width * 3, 5)
    baseline = _rolling_median(profile, half_window)
    depth = baseline - profile
    candidate = (
        (profile < baseline * drop_fraction)
        & (depth > span * 0.04)
        & (baseline > span * 0.02)
    )
    gap_axis = _short_runs(candidate, max_width)

    # Do not mark outer detector borders as CCD gaps.
    if np.any(gap_axis):
        edge_margin = max(2, max_width)
        gap_axis[:edge_margin] = False
        gap_axis[-edge_margin:] = False

    runs = _true_runs(gap_axis)
    if not runs:
        return np.zeros(profile.size, dtype=bool)

    center = (profile.size - 1) / 2.0
    half_center_window = max(profile.size * float(center_window) / 2.0, max_width)
    central_runs = [
        (start, end)
        for start, end in runs
        if abs(((start + end - 1) / 2.0) - center) <= half_center_window
    ]
    if not central_runs:
        return np.zeros(profile.size, dtype=bool)

    best_start, best_end = max(
        central_runs,
        key=lambda run: (
            float(np.nansum(depth[run[0]:run[1]]))
            / (abs(((run[0] + run[1] - 1) / 2.0) - center) + 1.0)
        ),
    )
    focused_gap_axis = np.zeros(profile.size, dtype=bool)
    focused_gap_axis[best_start:best_end] = True
    return focused_gap_axis

def detect_detector_gap_mask(image, max_width=16, drop_fraction=0.55, dilate=1, center_window=0.35):
    """Detect fixed CCD seam/gap pixels in detector coordinates."""
    data = np.asarray(image, dtype=np.float32)
    if data.ndim != 2 or data.size == 0:
        return np.zeros_like(data, dtype=bool)

    finite = np.isfinite(data)
    fill_value = float(np.nanmedian(data[finite])) if np.any(finite) else 0.0
    work = np.nan_to_num(data, nan=fill_value, posinf=fill_value, neginf=fill_value)
    row_profile = np.nanmedian(work, axis=1)
    col_profile = np.nanmedian(work, axis=0)
    gap_rows = _detect_dark_gap_axis(
        row_profile,
        max_width=max_width,
        drop_fraction=drop_fraction,
        center_window=center_window,
    )
    gap_cols = _detect_dark_gap_axis(
        col_profile,
        max_width=max_width,
        drop_fraction=drop_fraction,
        center_window=center_window,
    )

    mask = ~finite
    if gap_rows.any():
        mask[gap_rows, :] = True
    if gap_cols.any():
        mask[:, gap_cols] = True
    if dilate > 0 and np.any(mask):
        mask = scipy.ndimage.binary_dilation(mask, iterations=int(dilate))
    return mask.astype(bool, copy=False)

def rotate_detector_gap_mask(mask, angle):
    if mask is None:
        return None
    rotated = scipy.ndimage.rotate(
        np.asarray(mask, dtype=np.uint8),
        angle=angle,
        reshape=True,
        order=0,
        mode='constant',
        cval=0,
        prefilter=False,
    )
    rotated = rotated.astype(bool)
    if np.any(rotated):
        rotated = scipy.ndimage.binary_dilation(rotated, iterations=1)
    return rotated

def rotate_detector_background_mask(image_shape, angle):
    """Return pixels introduced by rotation outside the original detector area."""
    support = np.ones(tuple(image_shape), dtype=np.uint8)
    rotated_support = scipy.ndimage.rotate(
        support,
        angle=angle,
        reshape=True,
        order=0,
        mode='constant',
        cval=0,
        prefilter=False,
    )
    return ~rotated_support.astype(bool)

def _build_tilt_signal(image):
    """Return a bright-signal image for automatic tilt estimation."""
    data = np.asarray(image, dtype=np.float32)
    if data.ndim != 2 or data.size == 0 or not np.any(np.isfinite(data)):
        return None

    finite = data[np.isfinite(data)]
    fill_value = float(np.nanmedian(finite)) if finite.size else 0.0
    work = np.nan_to_num(data, nan=fill_value, posinf=fill_value, neginf=fill_value)

    rows, cols = work.shape
    row_margin = max(2, int(rows * 0.03))
    col_margin = max(2, int(cols * 0.03))
    if rows > 2 * row_margin + 20 and cols > 2 * col_margin + 20:
        work = work[row_margin:rows - row_margin, col_margin:cols - col_margin]

    baseline = float(np.nanmedian(work))
    work = work - baseline
    work[work < 0] = 0

    positive = work[work > 0]
    if positive.size < 25:
        return None

    high_clip = float(np.nanpercentile(positive, 99.8))
    if high_clip > 0:
        work = np.minimum(work, high_clip)

    n_rows, n_cols = work.shape
    min_cols = max(12, int(0.08 * n_cols))
    for percentile in (99.5, 99.0, 98.0, 97.0, 95.0, 90.0):
        threshold = float(np.nanpercentile(work, percentile))
        signal = work - threshold
        signal[signal < 0] = 0
        if np.count_nonzero(signal) < 25:
            continue
        active_cols = np.count_nonzero(signal.sum(axis=0) > 0)
        active_rows = np.count_nonzero(signal.sum(axis=1) > 0)
        if active_cols >= min_cols and active_rows >= 2:
            return signal.astype(np.float32, copy=False)

    return work.astype(np.float32, copy=False)

def _tilt_projection_score(signal, angle):
    rotated = scipy.ndimage.rotate(
        signal,
        angle=angle,
        reshape=False,
        order=1,
        mode='constant',
        cval=0.0,
        prefilter=False,
    )
    row_profile = rotated.sum(axis=1)
    total = float(row_profile.sum())
    if total <= 0 or not np.isfinite(total):
        return -np.inf
    row_profile = scipy.ndimage.gaussian_filter1d(row_profile, sigma=1.0)
    return float(np.sum(row_profile * row_profile) / (total * total))

def estimate_image_tilt_angle(image, max_abs_angle=8.0, coarse_step=0.25, fine_step=0.05):
    """Estimate the correction angle that makes bright CCD bands horizontal."""
    signal = _build_tilt_signal(image)
    if signal is None:
        return 0.0

    coarse_angles = np.arange(-max_abs_angle, max_abs_angle + coarse_step * 0.5, coarse_step)
    coarse_scores = np.array([_tilt_projection_score(signal, angle) for angle in coarse_angles])
    if coarse_scores.size == 0 or not np.any(np.isfinite(coarse_scores)):
        return 0.0

    best_angle = float(coarse_angles[int(np.nanargmax(coarse_scores))])
    fine_start = max(-max_abs_angle, best_angle - coarse_step)
    fine_end = min(max_abs_angle, best_angle + coarse_step)
    fine_angles = np.arange(fine_start, fine_end + fine_step * 0.5, fine_step)
    fine_scores = np.array([_tilt_projection_score(signal, angle) for angle in fine_angles])
    if fine_scores.size and np.any(np.isfinite(fine_scores)):
        best_angle = float(fine_angles[int(np.nanargmax(fine_scores))])

    return round(best_angle, 4)

def _set_tilt_readout(self, tilt):
    text = f"{tilt:.2f}"
    if hasattr(self, 'tilt_value'):
        self.tilt_value.set(text)
    elif hasattr(self, 'tilt_entry'):
        state = self.tilt_entry.cget('state')
        self.tilt_entry.configure(state='normal')
        self.tilt_entry.delete(0, tk.END)
        self.tilt_entry.insert(0, text)
        self.tilt_entry.configure(state=state)

def process_image(self):
    """Process the image with automatic tilt correction while preserving pixel intensities."""
    from IXE.spectrum_utils import SpectrumProcessor
    if not hasattr(self, 'imm'):
        return
    try:
        self.parm['threshold'] = 0
        tilt = estimate_image_tilt_angle(self.imm)
        self.parm['tilt_cor'] = tilt
        self.auto_tilt_angle = tilt
        _set_tilt_readout(self, tilt)
        gap_defaults = self.parm.get('ccd_gap', {})
        self.raw_gap_mask = detect_detector_gap_mask(
            self.imm,
            max_width=gap_defaults.get('mask_max_width', 16),
            drop_fraction=gap_defaults.get('mask_drop_fraction', 0.55),
            dilate=gap_defaults.get('mask_dilate', 1),
            center_window=gap_defaults.get('mask_center_window', 0.35),
        )
        self.processed_gap_mask = rotate_detector_gap_mask(self.raw_gap_mask, tilt)
        self.processed_detector_background_mask = rotate_detector_background_mask(self.imm.shape, tilt)
        self.immm = scipy.ndimage.rotate(self.imm, angle=tilt)
        if self.processed_gap_mask is not None and self.processed_gap_mask.shape != self.immm.shape:
            print("Warning: processed CCD gap mask shape does not match tilted image; mask disabled")
            self.processed_gap_mask = None
        if (
            self.processed_detector_background_mask is not None
            and self.processed_detector_background_mask.shape != self.immm.shape
        ):
            print("Warning: detector background mask shape does not match tilted image; mask disabled")
            self.processed_detector_background_mask = None
        self.spect_processor = SpectrumProcessor(
            self.immm,
            self.parm['n_moveavg'],
            invalid_pixel_mask=self.processed_gap_mask,
            detector_background_mask=self.processed_detector_background_mask,
            edge_valid_fraction_threshold=gap_defaults.get('edge_valid_fraction', 0.80),
        )
        if hasattr(self.spect_processor, 'toggle_gap_correction'):
            self.spect_processor.toggle_gap_correction(getattr(self, 'gap_correction_enabled', gap_defaults.get('enabled', True)))
            self.spect_processor.set_gap_correction_params(
                max_width=gap_defaults.get('max_width', 8),
                drop_fraction=gap_defaults.get('drop_fraction', 0.65),
            )
        if hasattr(self.spect_processor, 'toggle_trace_extraction'):
            trace_defaults = self.parm.get('trace_extraction', {})
            self.spect_processor.toggle_trace_extraction(getattr(self, 'trace_extraction_enabled', trace_defaults.get('enabled', False)))
            self.spect_processor.set_trace_extraction_params(
                search_margin=trace_defaults.get('search_margin', 60),
            )
        if hasattr(self, 'update_roi_row_slider_limits'):
            self.update_roi_row_slider_limits(self.immm.shape[0])
        if hasattr(self, 'update_bg_row_slider_limits'):
            self.update_bg_row_slider_limits(self.immm.shape[0])
        if hasattr(self, 'update_roi_col_slider_limits'):
            self.update_roi_col_slider_limits(self.immm.shape[1])
        self.display_image(self.immm, self.ax_image, f"Processed Image (Auto Tilt {tilt:.2f} deg)")
        self.canvas_image.draw()
        masked_pixels = int(np.count_nonzero(self.processed_gap_mask)) if self.processed_gap_mask is not None else 0
        background_pixels = (
            int(np.count_nonzero(self.processed_detector_background_mask))
            if self.processed_detector_background_mask is not None
            else 0
        )
        print(
            f"Auto tilt correction: {tilt:.2f} deg; "
            f"CCD gap mask pixels: {masked_pixels}; "
            f"rotation background pixels: {background_pixels}"
        )
    except Exception as e:
        print(f"Error processing image: {e}")

def show_gap_mask(self):
    """Overlay the processed CCD gap mask on the tilted image."""
    if not hasattr(self, 'processed_gap_mask') or self.processed_gap_mask is None:
        messagebox.showwarning("CCD Gap Mask", "Process an image first to create the tilted CCD gap mask.")
        return
    if not hasattr(self, 'immm'):
        messagebox.showwarning("CCD Gap Mask", "Process an image first to view the tilted CCD image.")
        return
    self.display_image(self.immm, self.ax_image, "Processed Image + CCD Gap Mask")
    mask_overlay = np.ma.masked_where(~self.processed_gap_mask, self.processed_gap_mask)
    self.gap_mask_overlay = self.ax_image.imshow(
        mask_overlay,
        interpolation='nearest',
        resample=False,
        aspect='auto',
        cmap='Reds',
        alpha=0.55,
        vmin=0,
        vmax=1,
        zorder=1.0,
    )
    self.ax_image.set_aspect('auto', adjustable='box')
    self.ax_image.set_xlim(0, self.immm.shape[1])
    self.ax_image.set_ylim(self.immm.shape[0], 0)
    self.canvas_image.draw_idle()

def _overlay_boolean_mask(ax, mask, color, alpha=0.65, zorder=0.7):
    rgba = np.zeros(mask.shape + (4,), dtype=float)
    rgba[mask, 0] = color[0]
    rgba[mask, 1] = color[1]
    rgba[mask, 2] = color[2]
    rgba[mask, 3] = alpha
    return ax.imshow(rgba, interpolation='nearest', zorder=zorder)

def _draw_detector_background_overlay(self, data, ax, title):
    mask = getattr(self, 'processed_detector_background_mask', None)
    if mask is None:
        return
    data_array = np.asarray(data)
    if mask.shape != data_array.shape:
        return
    processed_image = getattr(self, 'immm', None)
    is_processed_view = processed_image is data or str(title).startswith("Processed Image")
    if not is_processed_view:
        return
    self.detector_background_overlay = _overlay_boolean_mask(
        ax,
        mask,
        color=(0.28, 0.0, 0.04),
        alpha=0.82,
        zorder=0.7,
    )

def display_image(self, data, ax, title):
    """Display an image in the specified axes"""
    self.root.update_idletasks()

    ax.clear()
    self.current_display_data = data
    self.current_display_title = title
    self.line_begin = None
    self.line_end = None
    self.roi_lines = []
    self.bg_rect = None
    self.bg_rects = []
    cmap_name = _get_image_cmap(self)

    canvas_widget = self.canvas_image.get_tk_widget()
    canvas_width = max(canvas_widget.winfo_width(), 420)
    canvas_height = max(canvas_widget.winfo_height(), 320)
    image_dpi = float(getattr(self, 'image_fig_dpi', getattr(self, 'fig_dpi', 100)))
    self.fig_image.set_size_inches(
        canvas_width / image_dpi,
        canvas_height / image_dpi,
        forward=False,
    )
    self.fig_image.set_dpi(image_dpi)
    self.fig_image.patch.set_facecolor('white')
    self.fig_image.subplots_adjust(left=0.10, bottom=0.10, right=0.93, top=0.97)

    img = ax.imshow(
        data,
        interpolation='nearest',
        resample=False,
        vmin=float(self.vmin_entry.get()),
        vmax=float(self.vmax_entry.get()),
        cmap=cmap_name,
    )
    ax.set_facecolor('white')
    _draw_detector_background_overlay(self, data, ax, title)
    if hasattr(self, 'image_title_label'):
        self.image_title_label.config(text=_stack_display_note(self, title))
    if hasattr(self, 'image_xlabel_label'):
        self.image_xlabel_label.config(text='Columns')
    if hasattr(self, 'update_image_ylabel'):
        self.update_image_ylabel()

    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    _style_image_axis_ticks(self, ax)
    if not hasattr(self, 'cbar') or self.cbar is None:
        self.cbar = self.fig_image.colorbar(img, ax=ax, fraction=0.022, pad=0.018)
        _style_image_colorbar(self)
    else:
        self.cbar.update_normal(img)
        _style_image_colorbar(self)
    ax.set_aspect('auto', adjustable='box')
    ax.set_xlim(0, data.shape[1])
    ax.set_ylim(data.shape[0], 0)
    for spine in ax.spines.values():
        spine.set_linewidth(_image_points(self, 0.5))
    if hasattr(self, 'refresh_image_overlays'):
        self.refresh_image_overlays()
    else:
        self.canvas_image.draw_idle()

def update_image_cmap(self):
    """Refresh the image display using the selected colormap."""
    if not hasattr(self, 'current_display_data'):
        return
    title = getattr(self, 'current_display_title', '')
    self.display_image(self.current_display_data, self.ax_image, title)
    self.canvas_image.draw_idle()

def _display_image_as_rgba(self, data):
    """Render image data to an RGBA PIL image using the current display range."""
    array = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(array)
    if not np.any(finite):
        array = np.zeros_like(array, dtype=np.float32)
        finite = np.ones_like(array, dtype=bool)
    fill_value = float(np.nanmedian(array[finite]))
    array = np.nan_to_num(array, nan=fill_value, posinf=fill_value, neginf=fill_value)
    try:
        vmin = float(self.vmin_entry.get())
        vmax = float(self.vmax_entry.get())
    except ValueError:
        vmin = float(np.nanpercentile(array, 1))
        vmax = float(np.nanpercentile(array, 99))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(array))
        vmax = float(np.nanmax(array))
    if vmax <= vmin:
        vmax = vmin + 1.0
    normalized = np.clip((array - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba = plt.get_cmap(_get_image_cmap(self))(normalized)
    return Image.fromarray((rgba * 255).astype(np.uint8), mode='RGBA')

def draw_lines(self):
    """Draw lines at active ROI row boundaries on the processed image."""
    if not hasattr(self, 'current_display_data'):
        return
    try:
        data = self.current_display_data
        row_ranges = self.get_roi_row_ranges() if hasattr(self, 'get_roi_row_ranges') else [self.get_roi_row_range()]
        col_begin, col_end = self.get_roi_col_range()
        if col_begin < 0 or col_end < 0 or col_begin >= data.shape[1] or col_end >= data.shape[1]:
            print("Invalid column indices.")
            return
        roi_lines = list(getattr(self, 'roi_lines', []) or [])
        if roi_lines:
            for line in roi_lines:
                _remove_artist(line)
        else:
            if hasattr(self, 'line_begin') and self.line_begin:
                _remove_artist(self.line_begin)
            if hasattr(self, 'line_end') and self.line_end:
                _remove_artist(self.line_end)
        self.roi_lines = []
        self.line_begin = None
        self.line_end = None
        for row_begin, row_end in row_ranges:
            if row_begin < 0 or row_end < 0 or row_begin >= data.shape[0] or row_end >= data.shape[0]:
                print("Invalid row indices.")
                continue
            begin_line = self.ax_image.plot(
                [col_begin, col_end],
                [row_begin, row_begin],
                color='w',
                linestyle='--',
                linewidth=_image_points(self, 0.45),
            )[0]
            end_line = self.ax_image.plot(
                [col_begin, col_end],
                [row_end, row_end],
                color='w',
                linestyle='--',
                linewidth=_image_points(self, 0.45),
            )[0]
            self.roi_lines.extend([begin_line, end_line])
        if self.roi_lines:
            self.line_begin = self.roi_lines[0]
            self.line_end = self.roi_lines[1] if len(self.roi_lines) > 1 else None
        self.canvas_image.draw()
    except Exception as e:
        print(f"Error drawing lines: {e}")

def draw_background_roi(self):
    """Draw translucent background ROI rectangles on the current image."""
    if not hasattr(self, 'current_display_data'):
        return
    try:
        data = self.current_display_data
        bg_ranges = self.get_bg_row_ranges() if hasattr(self, 'get_bg_row_ranges') else [self.get_bg_row_range()]
        col_begin, col_end = self.get_roi_col_range()
        if col_begin < 0 or col_end < 0 or col_begin >= data.shape[1] or col_end >= data.shape[1]:
            return
        bg_rects = list(getattr(self, 'bg_rects', []) or [])
        if bg_rects:
            for rect in bg_rects:
                _remove_artist(rect)
        elif hasattr(self, 'bg_rect') and self.bg_rect:
            _remove_artist(self.bg_rect)
        self.bg_rects = []
        colors = ('white', '#d7f5ff')
        for index, (bg_row_begin, bg_row_end) in enumerate(bg_ranges[:2]):
            if bg_row_begin < 0 or bg_row_end < 0 or bg_row_begin >= data.shape[0] or bg_row_end >= data.shape[0]:
                continue
            rect = Rectangle(
                (col_begin, bg_row_begin),
                max(col_end - col_begin + 1, 1),
                max(bg_row_end - bg_row_begin + 1, 1),
                linewidth=_image_points(self, 0.45),
                edgecolor='white',
                facecolor=colors[index % len(colors)],
                alpha=0.28,
            )
            self.ax_image.add_patch(rect)
            self.bg_rects.append(rect)
        self.bg_rect = self.bg_rects[0] if self.bg_rects else None
        self.canvas_image.draw_idle()
    except Exception as e:
        print(f"Error drawing background ROI: {e}")

def save_processed_image(self):
    """Save the processed image without relying on Matplotlib savefig backends."""
    if not hasattr(self, 'immm'):
        messagebox.showwarning("Reminder", "Error: No processed image to save")
        return
    try:
        # Do not pass filetypes here. Some macOS/Tk builds crash in the
        # native save dialog when allowed file types include multi-extension
        # patterns, before Python can catch an exception.
        save_path = filedialog.asksaveasfilename(
            initialfile='processed_image.tiff',
            defaultextension='.tiff',
        )
        if not save_path:
            return
        extension = save_path.lower().rsplit('.', 1)[-1] if '.' in save_path else 'tiff'
        if extension in ('tif', 'tiff'):
            image_data = np.asarray(self.immm, dtype=np.float32)
            finite = np.isfinite(image_data)
            fill_value = float(np.nanmedian(image_data[finite])) if np.any(finite) else 0.0
            image_data = np.nan_to_num(image_data, nan=fill_value, posinf=fill_value, neginf=fill_value)
            Image.fromarray(image_data.astype(np.float32, copy=False)).save(save_path)
        else:
            image = _display_image_as_rgba(self, self.immm)
            if extension in ('jpg', 'jpeg'):
                image = image.convert('RGB')
            image.save(save_path)
        print(f"Image saved to {save_path}")
    except Exception as e:
        messagebox.showwarning("Save Image", f"Could not save image:\n{e}")
        print(f"Error saving image: {e}")
# image_processing.py
# Image loading, cross removal, thresholding, and display functions for TIFFAnalyzer.

# All functions will be moved here unchanged in the next step.
