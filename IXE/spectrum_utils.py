# -*- coding: utf-8 -*-
# spectrum_utils.py
import numpy as np
import matplotlib.pyplot as plt
import random


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


def _short_true_runs(mask, max_width):
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


def _smooth_1d(values, window):
    values = np.asarray(values, dtype=float)
    window = max(int(window), 1)
    if window <= 1 or values.size < 3:
        return values.copy()
    if window % 2 == 0:
        window += 1
    window = min(window, values.size if values.size % 2 == 1 else values.size - 1)
    if window < 3:
        return values.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode='same')


def _fill_missing_1d(values):
    values = np.asarray(values, dtype=float).copy()
    if values.size == 0:
        return values
    finite = np.isfinite(values)
    if finite.all():
        return values
    if finite.sum() == 0:
        return np.zeros_like(values)
    x = np.arange(values.size, dtype=float)
    values[~finite] = np.interp(x[~finite], x[finite], values[finite])
    return values


class SpectrumProcessor:
    def __init__(
        self,
        processed_image,
        n_moveavg,
        invalid_pixel_mask=None,
        detector_background_mask=None,
        edge_valid_fraction_threshold=0.80,
    ):
        if not isinstance(processed_image, np.ndarray):
            raise ValueError("processed_image must be a numpy array")
        if not isinstance(n_moveavg, int) or n_moveavg <= 0:
            raise ValueError("n_moveavg must be a positive integer")
        self.processed_image = processed_image.copy()
        self.invalid_pixel_mask = None
        self.set_invalid_pixel_mask(invalid_pixel_mask)
        self.detector_background_mask = None
        self.set_detector_background_mask(detector_background_mask)
        self.edge_valid_fraction_threshold = min(max(float(edge_valid_fraction_threshold), 0.05), 1.0)
        self.background_roi = None
        self.background_rois = []
        self.background_spectrum = None
        self.background_gap_mask = None
        self.background_height = None
        self.bg_subtracted = False  # Track subtraction state internally
        self.n_moveavg = n_moveavg 
        self.gap_correction_enabled = True
        self.gap_max_width = 8
        self.gap_drop_fraction = 0.65
        self.last_gap_mask = None
        self.last_roi_gap_mask = None
        self.last_background_gap_mask = None
        self.last_edge_mask = None
        self.last_roi_edge_mask = None
        self.last_background_edge_mask = None
        self.last_uncorrected_spectrum = None
        self.trace_extraction_enabled = False
        self.trace_search_margin = 60
        self.last_trace_centers = None
        self.last_extraction_pixel_gap_mask = None
        self.last_extraction_edge_mask = None

    def set_invalid_pixel_mask(self, invalid_pixel_mask):
        if invalid_pixel_mask is None:
            self.invalid_pixel_mask = None
            return
        mask = np.asarray(invalid_pixel_mask, dtype=bool)
        if mask.shape != self.processed_image.shape:
            print("Warning: CCD gap mask shape does not match processed image; ignoring mask")
            self.invalid_pixel_mask = None
            return
        self.invalid_pixel_mask = mask.copy()

    def set_detector_background_mask(self, detector_background_mask):
        if detector_background_mask is None:
            self.detector_background_mask = None
            return
        mask = np.asarray(detector_background_mask, dtype=bool)
        if mask.shape != self.processed_image.shape:
            print("Warning: detector background mask shape does not match processed image; ignoring mask")
            self.detector_background_mask = None
            return
        self.detector_background_mask = mask.copy()

    def toggle_gap_correction(self, enable):
        """Toggle narrow CCD detector-gap correction."""
        self.gap_correction_enabled = bool(enable)

    def toggle_trace_extraction(self, enable):
        """Toggle column-by-column trace-following spectrum extraction."""
        self.trace_extraction_enabled = bool(enable)

    def set_gap_correction_params(self, max_width=None, drop_fraction=None):
        if max_width is not None:
            self.gap_max_width = max(1, int(max_width))
        if drop_fraction is not None:
            self.gap_drop_fraction = min(max(float(drop_fraction), 0.05), 0.95)

    def set_edge_correction_params(self, valid_fraction_threshold=None):
        if valid_fraction_threshold is not None:
            self.edge_valid_fraction_threshold = min(max(float(valid_fraction_threshold), 0.05), 1.0)

    def set_trace_extraction_params(self, search_margin=None):
        if search_margin is not None:
            self.trace_search_margin = max(1, int(search_margin))

    def set_background_roi(self, row_begin, row_end, col_begin, col_end):
        self.set_background_rois([(row_begin, row_end, col_begin, col_end)])

    def set_background_rois(self, rois):
        """Set one or more background strips in processed-image coordinates."""
        valid_rois = []
        for roi in rois or []:
            if roi is None or len(roi) != 4:
                continue
            row_begin, row_end, col_begin, col_end = (int(value) for value in roi)
            row_begin, row_end = sorted((row_begin, row_end))
            col_begin, col_end = sorted((col_begin, col_end))
            if row_end <= row_begin or col_end <= col_begin:
                continue
            valid_rois.append((row_begin, row_end, col_begin, col_end))

        self.background_rois = valid_rois
        self.background_roi = valid_rois[0] if valid_rois else None
        if not valid_rois:
            self.background_spectrum = None
            self.background_gap_mask = None
            self.background_height = None
            return

        spectra = []
        masks = []
        heights = []
        for row_begin, row_end, col_begin, col_end in valid_rois:
            spectrum, gap_mask, _edge_mask, height = self._extract_rect_spectrum(
                row_begin,
                row_end,
                col_begin,
                col_end,
                apply_mask=self.gap_correction_enabled,
            )
            if height > 0:
                spectra.append(np.asarray(spectrum, dtype=float))
                masks.append(np.asarray(gap_mask, dtype=bool))
                heights.append(int(height))

        if not spectra:
            self.background_spectrum = None
            self.background_gap_mask = None
            self.background_height = None
            return

        min_len = min(spectrum.size for spectrum in spectra)
        spectra = [spectrum[:min_len] for spectrum in spectra]
        masks = [mask[:min_len] for mask in masks]
        total_height = max(int(np.sum(heights)), 1)
        self.background_spectrum = np.nansum(np.vstack(spectra), axis=0) / float(total_height)
        self.background_gap_mask = np.any(np.vstack(masks), axis=0)
        self.background_height = total_height

    def toggle_background_subtraction(self, enable):
        """Toggle whether subtraction should be applied"""
        self.bg_subtracted = enable

    def _region_bounds(self, row_begin, row_end, col_begin, col_end):
        image = self.processed_image
        row_begin = max(0, int(row_begin))
        row_end = min(int(row_end), image.shape[0])
        col_begin = max(0, int(col_begin))
        col_end = min(int(col_end), image.shape[1])
        return row_begin, row_end, col_begin, col_end

    def _masked_region(self, mask, row_begin, row_end, col_begin, col_end):
        if mask is None:
            return None
        return mask[row_begin:row_end, col_begin:col_end]

    def _combine_invalid_masks(self, *masks):
        combined = None
        for mask in masks:
            if mask is None:
                continue
            mask = np.asarray(mask, dtype=bool)
            combined = mask.copy() if combined is None else (combined | mask)
        return combined

    def _edge_column_too_sparse(self, detector_background_column):
        if detector_background_column is None:
            return False
        detector_background_column = np.asarray(detector_background_column, dtype=bool)
        if detector_background_column.size == 0:
            return False
        valid_fraction = np.count_nonzero(~detector_background_column) / float(detector_background_column.size)
        return valid_fraction < self.edge_valid_fraction_threshold

    def _scaled_column_sum(self, values, invalid=None):
        values = np.asarray(values, dtype=float)
        if invalid is None:
            valid = np.isfinite(values)
        else:
            valid = (~np.asarray(invalid, dtype=bool)) & np.isfinite(values)
        nominal = values.shape[0]
        count = int(np.count_nonzero(valid))
        if nominal <= 0:
            return np.nan, False
        if count <= 0:
            return np.nan, True
        total = float(np.nansum(values[valid]))
        missing_fraction = (nominal - count) / float(nominal)
        return total * (nominal / count), missing_fraction >= 0.5

    def _extract_rect_spectrum(self, row_begin, row_end, col_begin, col_end, apply_mask=False):
        row_begin, row_end, col_begin, col_end = self._region_bounds(row_begin, row_end, col_begin, col_end)
        roi = np.asarray(self.processed_image[row_begin:row_end, col_begin:col_end], dtype=float)
        if roi.size == 0:
            length = roi.shape[1] if roi.ndim == 2 else 0
            return roi.sum(axis=0), np.zeros(length, dtype=bool), np.zeros(length, dtype=bool), 0

        gap_invalid = None
        detector_background = None
        if apply_mask:
            gap_invalid = self._masked_region(self.invalid_pixel_mask, row_begin, row_end, col_begin, col_end)
            detector_background = self._masked_region(
                self.detector_background_mask,
                row_begin,
                row_end,
                col_begin,
                col_end,
            )

        spectrum = np.zeros(roi.shape[1], dtype=float)
        pixel_gap_mask = np.zeros(roi.shape[1], dtype=bool)
        edge_mask = np.zeros(roi.shape[1], dtype=bool)
        for col_offset in range(roi.shape[1]):
            col_gap_invalid = gap_invalid[:, col_offset] if gap_invalid is not None else None
            col_detector_background = detector_background[:, col_offset] if detector_background is not None else None
            if self._edge_column_too_sparse(col_detector_background):
                spectrum[col_offset] = np.nan
                edge_mask[col_offset] = True
                continue
            col_invalid = self._combine_invalid_masks(col_gap_invalid, col_detector_background)
            spectrum[col_offset], pixel_gap_mask[col_offset] = self._scaled_column_sum(roi[:, col_offset], col_invalid)
        spectrum = _fill_missing_1d(spectrum)
        self.last_extraction_pixel_gap_mask = pixel_gap_mask.copy()
        self.last_extraction_edge_mask = edge_mask.copy()
        return spectrum, pixel_gap_mask, edge_mask, roi.shape[0]

    def _extract_selected_spectrum(self, row_begin, row_end, col_begin, col_end, apply_mask=False):
        if self.trace_extraction_enabled:
            spectrum = self.get_traced_spectrum(row_begin, row_end, col_begin, col_end, apply_mask=apply_mask)
            gap_mask = getattr(self, 'last_extraction_pixel_gap_mask', None)
            if gap_mask is None or len(gap_mask) != len(spectrum):
                gap_mask = np.zeros_like(spectrum, dtype=bool)
            edge_mask = getattr(self, 'last_extraction_edge_mask', None)
            if edge_mask is None or len(edge_mask) != len(spectrum):
                edge_mask = np.zeros_like(spectrum, dtype=bool)
            height = max(int(row_end) - int(row_begin), 0)
            return spectrum, np.asarray(gap_mask, dtype=bool), np.asarray(edge_mask, dtype=bool), height
        spectrum, gap_mask, edge_mask, height = self._extract_rect_spectrum(
            row_begin,
            row_end,
            col_begin,
            col_end,
            apply_mask=apply_mask,
        )
        self.last_trace_centers = None
        self.last_extraction_pixel_gap_mask = gap_mask.copy()
        self.last_extraction_edge_mask = edge_mask.copy()
        return spectrum, gap_mask, edge_mask, height

    def _normalize_spectrum_rois(self, rois):
        valid_rois = []
        for roi in rois or []:
            if roi is None or len(roi) != 4:
                continue
            row_begin, row_end, col_begin, col_end = (int(value) for value in roi)
            row_begin, row_end = sorted((row_begin, row_end))
            col_begin, col_end = sorted((col_begin, col_end))
            row_begin, row_end, col_begin, col_end = self._region_bounds(
                row_begin,
                row_end,
                col_begin,
                col_end,
            )
            if row_end <= row_begin or col_end <= col_begin:
                continue
            valid_rois.append((row_begin, row_end, col_begin, col_end))
        return valid_rois

    def _extract_multi_spectrum(self, rois, apply_mask=False):
        spectra = []
        masks = []
        edge_masks = []
        heights = []
        for row_begin, row_end, col_begin, col_end in rois:
            spectrum, gap_mask, edge_mask, height = self._extract_selected_spectrum(
                row_begin,
                row_end,
                col_begin,
                col_end,
                apply_mask=apply_mask,
            )
            if height > 0 and spectrum.size:
                spectra.append(np.asarray(spectrum, dtype=float))
                masks.append(np.asarray(gap_mask, dtype=bool))
                edge_masks.append(np.asarray(edge_mask, dtype=bool))
                heights.append(int(height))

        if not spectra:
            return np.array([], dtype=float), np.array([], dtype=bool), np.array([], dtype=bool), 0

        min_len = min(spectrum.size for spectrum in spectra)
        spectra = [spectrum[:min_len] for spectrum in spectra]
        masks = [mask[:min_len] for mask in masks]
        edge_masks = [mask[:min_len] for mask in edge_masks]
        spectrum = np.nansum(np.vstack(spectra), axis=0)
        gap_mask = np.any(np.vstack(masks), axis=0)
        edge_mask = np.any(np.vstack(edge_masks), axis=0)
        height = int(np.sum(heights))
        return spectrum, gap_mask, edge_mask, height

    def _get_background_spectrum(self, roi_height, apply_mask=False):
        if getattr(self, 'background_rois', None):
            spectra = []
            masks = []
            edge_masks = []
            heights = []
            for bg_row_begin, bg_row_end, bg_col_begin, bg_col_end in self.background_rois:
                background, gap_mask, edge_mask, bg_height = self._extract_rect_spectrum(
                    bg_row_begin,
                    bg_row_end,
                    bg_col_begin,
                    bg_col_end,
                    apply_mask=apply_mask,
                )
                if bg_height > 0:
                    spectra.append(np.asarray(background, dtype=float))
                    masks.append(np.asarray(gap_mask, dtype=bool))
                    edge_masks.append(np.asarray(edge_mask, dtype=bool))
                    heights.append(int(bg_height))
            if not spectra:
                return None, None, None
            min_len = min(spectrum.size for spectrum in spectra)
            spectra = [spectrum[:min_len] for spectrum in spectra]
            masks = [mask[:min_len] for mask in masks]
            edge_masks = [mask[:min_len] for mask in edge_masks]
            total_height = max(int(np.sum(heights)), 1)
            per_pixel_background = np.nansum(np.vstack(spectra), axis=0) / float(total_height)
            background = per_pixel_background * float(max(roi_height, 0))
            gap_mask = np.any(np.vstack(masks), axis=0)
            edge_mask = np.any(np.vstack(edge_masks), axis=0)
            return background, gap_mask, edge_mask

        if self.background_roi is None:
            if self.background_spectrum is None:
                return None, None, None
            background = np.asarray(self.background_spectrum, dtype=float).copy()
            if roi_height > 0 and self.background_height in (None, 1):
                background = background * float(roi_height)
            gap_mask = np.asarray(self.background_gap_mask, dtype=bool) if self.background_gap_mask is not None else np.zeros_like(background, dtype=bool)
            edge_mask = np.zeros_like(background, dtype=bool)
            return background, gap_mask, edge_mask

        bg_row_begin, bg_row_end, bg_col_begin, bg_col_end = self.background_roi
        background, gap_mask, edge_mask, bg_height = self._extract_rect_spectrum(
            bg_row_begin,
            bg_row_end,
            bg_col_begin,
            bg_col_end,
            apply_mask=apply_mask,
        )
        if bg_height > 0 and roi_height > 0:
            background = background * (float(roi_height) / float(bg_height))
        return background, gap_mask, edge_mask

    def get_spectrum_for_rois(self, rois):
        """
        Get a spectrum for one or more ROI strips, applying the current
        background subtraction and CCD gap settings.
        """
        rois = self._normalize_spectrum_rois(rois)
        if not rois:
            empty = np.array([], dtype=float)
            self.last_uncorrected_spectrum = empty.copy()
            self.last_gap_mask = np.zeros(0, dtype=bool)
            self.last_roi_gap_mask = np.zeros(0, dtype=bool)
            self.last_background_gap_mask = np.zeros(0, dtype=bool)
            self.last_edge_mask = np.zeros(0, dtype=bool)
            self.last_roi_edge_mask = np.zeros(0, dtype=bool)
            self.last_background_edge_mask = np.zeros(0, dtype=bool)
            return empty

        raw_spectrum, _, _, roi_height = self._extract_multi_spectrum(rois, apply_mask=False)
        raw_background = None
        if self.bg_subtracted:
            raw_background, _, _ = self._get_background_spectrum(roi_height, apply_mask=False)

        if raw_background is not None and len(raw_spectrum) == len(raw_background):
            raw_display_spectrum = raw_spectrum - raw_background
        elif raw_background is not None:
            print("Warning: Background length doesn't match spectrum")
            raw_display_spectrum = raw_spectrum.copy()
        else:
            raw_display_spectrum = raw_spectrum.copy()
        self.last_uncorrected_spectrum = raw_display_spectrum.copy()

        if not self.gap_correction_enabled:
            self.last_gap_mask = np.zeros_like(raw_display_spectrum, dtype=bool)
            self.last_roi_gap_mask = np.zeros_like(raw_display_spectrum, dtype=bool)
            self.last_background_gap_mask = np.zeros_like(raw_display_spectrum, dtype=bool)
            self.last_edge_mask = np.zeros_like(raw_display_spectrum, dtype=bool)
            self.last_roi_edge_mask = np.zeros_like(raw_display_spectrum, dtype=bool)
            self.last_background_edge_mask = np.zeros_like(raw_display_spectrum, dtype=bool)
            return raw_display_spectrum

        roi_spectrum, roi_pixel_gap_mask, roi_edge_mask, roi_height = self._extract_multi_spectrum(rois, apply_mask=True)
        roi_spectrum, roi_1d_gap_mask = self.correct_narrow_gaps(roi_spectrum)
        roi_gap_mask = roi_pixel_gap_mask | roi_1d_gap_mask

        background_spectrum = None
        background_gap_mask = np.zeros_like(roi_gap_mask, dtype=bool)
        background_edge_mask = np.zeros_like(roi_edge_mask, dtype=bool)
        if self.bg_subtracted:
            background_spectrum, background_pixel_gap_mask, background_edge_pixel_mask = self._get_background_spectrum(roi_height, apply_mask=True)
            if background_spectrum is not None and len(background_spectrum) == len(roi_spectrum):
                background_spectrum, background_1d_gap_mask = self.correct_narrow_gaps(background_spectrum)
                background_gap_mask = background_pixel_gap_mask | background_1d_gap_mask
                if background_edge_pixel_mask is not None:
                    background_edge_mask = np.asarray(background_edge_pixel_mask, dtype=bool)
            elif background_spectrum is not None:
                print("Warning: Background length doesn't match spectrum")
                background_spectrum = None

        if background_spectrum is None:
            spectrum = roi_spectrum
            gap_mask = roi_gap_mask
        else:
            spectrum = roi_spectrum - background_spectrum
            gap_mask = roi_gap_mask | background_gap_mask

        self.last_roi_gap_mask = roi_gap_mask
        self.last_background_gap_mask = background_gap_mask
        self.last_roi_edge_mask = roi_edge_mask
        self.last_background_edge_mask = background_edge_mask
        self.last_edge_mask = roi_edge_mask | background_edge_mask
        self.last_gap_mask = gap_mask | self.last_edge_mask
        return spectrum

    def get_spectrum(self, row_begin, row_end, col_begin, col_end):
        """
        Get spectrum for one ROI strip, applying background subtraction if enabled.
        Note: No subtract_bg parameter - uses internal state instead.
        """
        return self.get_spectrum_for_rois([(row_begin, row_end, col_begin, col_end)])

    def get_traced_spectrum(self, row_begin, row_end, col_begin, col_end, apply_mask=None):
        """Extract a spectrum by following the local emission trace across columns."""
        if apply_mask is None:
            apply_mask = self.gap_correction_enabled
        image = self.processed_image
        row_begin = max(0, int(row_begin))
        row_end = min(int(row_end), image.shape[0])
        col_begin = max(0, int(col_begin))
        col_end = min(int(col_end), image.shape[1])
        if row_end <= row_begin or col_end <= col_begin:
            self.last_trace_centers = None
            self.last_extraction_pixel_gap_mask = np.zeros(max(col_end - col_begin, 0), dtype=bool)
            self.last_extraction_edge_mask = np.zeros(max(col_end - col_begin, 0), dtype=bool)
            return image[row_begin:row_end, col_begin:col_end].sum(axis=0)

        aperture_height = max(row_end - row_begin, 1)
        aperture_half = max(aperture_height / 2.0, 0.5)
        center_guess = (row_begin + row_end - 1) / 2.0
        margin = max(int(self.trace_search_margin), aperture_height)
        search_top = max(0, row_begin - margin)
        search_bottom = min(image.shape[0], row_end + margin)
        search_rows = np.arange(search_top, search_bottom, dtype=float)
        sub = np.asarray(image[search_top:search_bottom, col_begin:col_end], dtype=float)
        if sub.size == 0:
            self.last_trace_centers = None
            self.last_extraction_pixel_gap_mask = np.zeros(max(col_end - col_begin, 0), dtype=bool)
            self.last_extraction_edge_mask = np.zeros(max(col_end - col_begin, 0), dtype=bool)
            return image[row_begin:row_end, col_begin:col_end].sum(axis=0)
        gap_invalid_sub = None
        detector_background_sub = None
        invalid_sub = None
        if apply_mask:
            gap_invalid_sub = self._masked_region(self.invalid_pixel_mask, search_top, search_bottom, col_begin, col_end)
            detector_background_sub = self._masked_region(
                self.detector_background_mask,
                search_top,
                search_bottom,
                col_begin,
                col_end,
            )
            invalid_sub = self._combine_invalid_masks(gap_invalid_sub, detector_background_sub)

        prior_sigma = max(aperture_height * 1.5, margin / 2.0, 1.0)
        prior = np.exp(-0.5 * ((search_rows - center_guess) / prior_sigma) ** 2)
        centers = np.full(sub.shape[1], np.nan, dtype=float)
        strengths = np.full(sub.shape[1], np.nan, dtype=float)

        for col_index in range(sub.shape[1]):
            profile = sub[:, col_index]
            if invalid_sub is not None:
                profile = np.where(invalid_sub[:, col_index], np.nan, profile)
            if not np.any(np.isfinite(profile)):
                continue
            floor = np.nanpercentile(profile, 20)
            profile_for_smoothing = np.nan_to_num(profile, nan=floor)
            smooth_profile = _smooth_1d(profile_for_smoothing, max(3, int(round(aperture_height / 3.0))))
            floor = np.nanpercentile(smooth_profile, 20)
            signal = np.clip(smooth_profile - floor, 0.0, None)
            weighted_signal = signal * prior
            strength = float(np.nansum(weighted_signal))
            if not np.isfinite(strength) or strength <= 0:
                continue
            local_peak = int(np.nanargmax(weighted_signal))
            local_left = max(0, int(round(local_peak - aperture_half)))
            local_right = min(signal.size, int(round(local_peak + aperture_half + 1)))
            local_signal = signal[local_left:local_right]
            local_rows = search_rows[local_left:local_right]
            local_total = float(np.nansum(local_signal))
            if local_total <= 0:
                centers[col_index] = search_rows[local_peak]
            else:
                centers[col_index] = float(np.nansum(local_rows * local_signal) / local_total)
            strengths[col_index] = strength

        finite_strengths = strengths[np.isfinite(strengths)]
        if finite_strengths.size:
            threshold = max(np.nanmedian(finite_strengths) * 0.10, np.nanmax(finite_strengths) * 0.02)
            centers[strengths < threshold] = np.nan

        valid = np.isfinite(centers)
        x = np.arange(centers.size, dtype=float)
        if valid.sum() >= 2:
            centers = np.interp(x, x[valid], centers[valid])
            centers = _smooth_1d(centers, max(5, int(round(centers.size * 0.03))))
        else:
            centers[:] = center_guess

        centers = np.clip(centers, aperture_half, image.shape[0] - aperture_half - 1)
        self.last_trace_centers = centers.copy()
        spectrum = np.zeros(col_end - col_begin, dtype=float)
        pixel_gap_mask = np.zeros(col_end - col_begin, dtype=bool)
        edge_mask = np.zeros(col_end - col_begin, dtype=bool)
        for col_offset, center in enumerate(centers):
            row_start = int(round(center - aperture_half))
            row_stop = row_start + aperture_height
            if row_start < 0:
                row_stop -= row_start
                row_start = 0
            if row_stop > image.shape[0]:
                row_start -= row_stop - image.shape[0]
                row_stop = image.shape[0]
            row_start = max(row_start, 0)
            row_stop = max(row_stop, row_start + 1)
            gap_invalid = None
            detector_background = None
            invalid = None
            if apply_mask:
                gap_invalid = self._masked_region(
                    self.invalid_pixel_mask,
                    row_start,
                    row_stop,
                    col_begin + col_offset,
                    col_begin + col_offset + 1,
                )
                detector_background = self._masked_region(
                    self.detector_background_mask,
                    row_start,
                    row_stop,
                    col_begin + col_offset,
                    col_begin + col_offset + 1,
                )
                if detector_background is not None:
                    detector_background = detector_background[:, 0]
                if gap_invalid is not None:
                    gap_invalid = gap_invalid[:, 0]
                if self._edge_column_too_sparse(detector_background):
                    spectrum[col_offset] = np.nan
                    edge_mask[col_offset] = True
                    continue
                invalid = self._combine_invalid_masks(gap_invalid, detector_background)
            spectrum[col_offset], pixel_gap_mask[col_offset] = self._scaled_column_sum(
                image[row_start:row_stop, col_begin + col_offset],
                invalid,
            )
        self.last_extraction_pixel_gap_mask = pixel_gap_mask
        self.last_extraction_edge_mask = edge_mask
        spectrum = _fill_missing_1d(spectrum)
        return spectrum

    def correct_narrow_gaps(self, spectrum):
        """Interpolate across narrow detector-gap dips in a 1D spectrum."""
        spectrum = np.asarray(spectrum, dtype=float).copy()
        gap_mask = np.zeros_like(spectrum, dtype=bool)
        if spectrum.size < 5:
            return spectrum, gap_mask

        max_width = max(1, min(int(self.gap_max_width), max(spectrum.size - 2, 1)))
        shifted = spectrum - np.nanpercentile(spectrum, 5)
        scale = np.nanmax(shifted) - np.nanmin(shifted)
        if not np.isfinite(scale) or scale <= np.finfo(float).eps:
            return spectrum, gap_mask

        half_window = max(max_width * 2, 3)
        baseline = _rolling_median(shifted, half_window)
        candidate = (shifted < baseline * self.gap_drop_fraction) & (baseline > scale * 0.08)
        raw_baseline = _rolling_median(spectrum, half_window)
        depth = raw_baseline - spectrum
        local_candidate = (
            (spectrum < raw_baseline * self.gap_drop_fraction)
            & (depth > scale * 0.04)
            & (raw_baseline > scale * 0.02)
        )
        candidate = candidate | local_candidate
        gap_mask = _short_true_runs(candidate, max_width)

        # Keep only interior gaps with valid neighbors on both sides.
        valid = ~gap_mask & np.isfinite(spectrum)
        if valid.sum() < 2 or not np.any(gap_mask):
            return spectrum, np.zeros_like(gap_mask, dtype=bool)
        first_valid = np.flatnonzero(valid)[0]
        last_valid = np.flatnonzero(valid)[-1]
        gap_mask[:first_valid + 1] = False
        gap_mask[last_valid:] = False
        if not np.any(gap_mask):
            return spectrum, gap_mask

        x = np.arange(spectrum.size, dtype=float)
        spectrum[gap_mask] = np.interp(x[gap_mask], x[valid], spectrum[valid])
        return spectrum, gap_mask

    def moving_average(self, a):
        """Apply moving average to the data"""
        n = self.n_moveavg # Use the value of n_moveavg from the class parameter
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n  # Get the moving average window size from the user input

    def get_norm_spectrum(self, spectrum):
        """Normalize the spectrum data based on the sum of its values."""
        spect = (spectrum - spectrum.min())  # Normalize between 0 and max
        norm_fac = spect.sum()  # Normalization factor (sum of the spectrum)
        return spect / norm_fac  # Return the normalized spectrum
    def random_color(self):
        """Generate a random color in hexadecimal format."""
        return "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    def calculate_satellite_peak_iad(self, spec_n, spec_r_n, i_l=0, i_r=None, i_g=None):
        """
        Calculate the integrated difference of the satellite peak between ROI (spec_n) and Ref (spec_r_n).
        Steps:
        1. Align spectra by main peak center.
        2. Cut left end to align.
        3. Find first intersection (transition point) after i_l, within [i_l, i_r] (i_l, i_r are passed from self.parm['PK intersect']).
        4. Integrate difference up to transition point.
        """
        # 1. Find main peak centers
        i_max = np.argmax(spec_n)
        i_ref_max = np.argmax(spec_r_n)
        i_dif = i_ref_max - i_max
        # 2. Align spectra
        if i_dif < 0:
            spec_align = spec_n[-i_dif:]
            spec_r_align = spec_r_n[:i_dif]
        elif i_dif > 0:
            spec_align = spec_n[:-i_dif]
            spec_r_align = spec_r_n[i_dif:]
        else:
            spec_align = spec_n
            spec_r_align = spec_r_n
        # 3. Set right bound if not provided
        #if i_r is None:
            #i_r = min(len(spec_align), len(spec_r_align)) - 1
        # 4. Set fallback transition index if not provided
        #if i_g is None:
            #i_g = i_r
        # 5. Find transition point. The spectra may be passed in either
        # order, so accept either sign-change direction and fall back to the
        # closest approach inside the requested cross window.
        transition_point = None
        if i_g is None:
            max_index = min(len(spec_align), len(spec_r_align)) - 1
            start = max(0, min(int(i_l), max_index - 1))
            end = max(start + 1, min(int(i_r), max_index))
            diff = np.asarray(spec_align, dtype=float) - np.asarray(spec_r_align, dtype=float)
            for i in range(start, end):
                left = diff[i]
                right = diff[i + 1]
                if not np.isfinite(left) or not np.isfinite(right):
                    continue
                if left == 0:
                    transition_point = i
                    break
                if right == 0:
                    transition_point = i + 1
                    break
                if np.sign(left) != np.sign(right):
                    transition_point = i + 1
                    break
            if transition_point is None:
                window = np.abs(diff[start:end + 1])
                finite = np.isfinite(window)
                if np.any(finite):
                    finite_indices = np.flatnonzero(finite)
                    transition_point = start + int(finite_indices[int(np.nanargmin(window[finite]))])
        else:
            transition_point = i_g
        
        # 6. Calculate integrated difference up to transition point
        iad_satellite = np.sum(np.abs(spec_align[:transition_point] - spec_r_align[:transition_point]))
        return iad_satellite, transition_point, spec_align, spec_r_align

    
# Now test the random_color function
#if __name__ == "__main__":
    # Create an instance of SpectrumProcessor
    #spect_processor = SpectrumProcessor()

    # Call the random_color function
    #color = spect_processor.random_color()
    #print(f"Generated random color: {color}")
    
    
