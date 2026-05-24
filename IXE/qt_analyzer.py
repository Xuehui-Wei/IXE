# -*- coding: utf-8 -*-
"""Qt/pyqtgraph analyzer with embedded CCD and spectrum displays."""

import re
import sys

import numpy as np
from PIL import Image
import scipy.ndimage

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph.exporters import ImageExporter, SVGExporter

try:
    from .spectrum_utils import SpectrumProcessor
    from .image_processing import (
        detect_detector_gap_mask,
        estimate_image_tilt_angle,
        rotate_detector_background_mask,
        rotate_detector_gap_mask,
    )
except ImportError:
    from spectrum_utils import SpectrumProcessor
    from image_processing import (
        detect_detector_gap_mask,
        estimate_image_tilt_angle,
        rotate_detector_background_mask,
        rotate_detector_gap_mask,
    )


pg.setConfigOptions(antialias=True, imageAxisOrder='row-major')


def _read_tiff_array(filepath):
    with Image.open(filepath) as image:
        return np.asarray(image, dtype=np.float32)


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


def _safe_int_pair(first, second):
    first, second = sorted((int(first), int(second)))
    return first, second


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


class QtXESAnalyzer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IXE Analyzer - Qt")
        self.resize(1320, 860)

        self.parm = {
            'n_moveavg': 5,
            'ccd_gap': {
                'enabled': True,
                'max_width': 8,
                'drop_fraction': 0.65,
                'mask_max_width': 16,
                'mask_drop_fraction': 0.55,
                'mask_dilate': 1,
                'mask_center_window': 0.35,
                'edge_valid_fraction': 0.80,
            },
        }
        self.filepaths = []
        self.imm = None
        self.immm = None
        self.raw_gap_mask = None
        self.processed_gap_mask = None
        self.processed_detector_background_mask = None
        self.spect_processor = None
        self.last_x = np.array([], dtype=float)
        self.last_y = np.array([], dtype=float)

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

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_image_panel())
        splitter.addWidget(self._build_spectrum_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([520, 780])
        layout.addWidget(splitter, stretch=1)

        self.statusBar().showMessage("Ready")

    def _build_controls(self):
        panel = QtWidgets.QGroupBox("Controls")
        grid = QtWidgets.QGridLayout(panel)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(5)

        self.import_button = QtWidgets.QPushButton("Import TIFF")
        self.import_stack_button = QtWidgets.QPushButton("Import Stack")
        self.process_button = QtWidgets.QPushButton("Process")
        self.plot_button = QtWidgets.QPushButton("Plot")
        self.auto_bg_button = QtWidgets.QPushButton("Auto BG")
        self.save_png_button = QtWidgets.QPushButton("Save PNG")
        self.save_svg_button = QtWidgets.QPushButton("Save SVG")

        self.import_button.clicked.connect(self.import_tiff)
        self.import_stack_button.clicked.connect(self.import_tiff_stack)
        self.process_button.clicked.connect(self.process_image)
        self.plot_button.clicked.connect(self.plot_spectrum)
        self.auto_bg_button.clicked.connect(self.auto_background)
        self.save_png_button.clicked.connect(self.save_spectrum_png)
        self.save_svg_button.clicked.connect(self.save_spectrum_svg)

        self.run_label = QtWidgets.QLabel("No image loaded")
        self.run_label.setMinimumWidth(240)

        self.vmin_spin = QtWidgets.QDoubleSpinBox()
        self.vmin_spin.setRange(-1e9, 1e9)
        self.vmin_spin.setDecimals(3)
        self.vmin_spin.setValue(0.0)
        self.vmax_spin = QtWidgets.QDoubleSpinBox()
        self.vmax_spin.setRange(-1e9, 1e9)
        self.vmax_spin.setDecimals(3)
        self.vmax_spin.setValue(100.0)
        self.vmin_spin.valueChanged.connect(self.refresh_image)
        self.vmax_spin.valueChanged.connect(self.refresh_image)

        self.gap_check = QtWidgets.QCheckBox("Gap mask")
        self.gap_check.setChecked(True)
        self.bg_check = QtWidgets.QCheckBox("BG remove")
        self.roi2_check = QtWidgets.QCheckBox("ROI 2")
        self.gap_overlay_check = QtWidgets.QCheckBox("Show gap overlay")
        self.gap_overlay_check.stateChanged.connect(self.refresh_image)

        self.tilt_label = QtWidgets.QLabel("Tilt: --")

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

        row = 0
        grid.addWidget(self.import_button, row, 0)
        grid.addWidget(self.import_stack_button, row, 1)
        grid.addWidget(self.process_button, row, 2)
        grid.addWidget(self.plot_button, row, 3)
        grid.addWidget(self.auto_bg_button, row, 4)
        grid.addWidget(self.save_png_button, row, 5)
        grid.addWidget(self.save_svg_button, row, 6)
        grid.addWidget(self.run_label, row, 7, 1, 4)

        row += 1
        grid.addWidget(QtWidgets.QLabel("vmin"), row, 0)
        grid.addWidget(self.vmin_spin, row, 1)
        grid.addWidget(QtWidgets.QLabel("vmax"), row, 2)
        grid.addWidget(self.vmax_spin, row, 3)
        grid.addWidget(self.gap_check, row, 4)
        grid.addWidget(self.bg_check, row, 5)
        grid.addWidget(self.gap_overlay_check, row, 6)
        grid.addWidget(self.tilt_label, row, 7)

        row += 1
        grid.addWidget(QtWidgets.QLabel("ROI rows"), row, 0)
        grid.addWidget(self.roi_row_start, row, 1)
        grid.addWidget(self.roi_row_end, row, 2)
        grid.addWidget(self.roi2_check, row, 3)
        grid.addWidget(self.roi2_row_start, row, 4)
        grid.addWidget(self.roi2_row_end, row, 5)
        grid.addWidget(QtWidgets.QLabel("ROI cols"), row, 6)
        grid.addWidget(self.roi_col_start, row, 7)
        grid.addWidget(self.roi_col_end, row, 8)

        row += 1
        grid.addWidget(QtWidgets.QLabel("BG rows"), row, 0)
        grid.addWidget(self.bg_row_start, row, 1)
        grid.addWidget(self.bg_row_end, row, 2)
        grid.addWidget(QtWidgets.QLabel("BG rows 2"), row, 3)
        grid.addWidget(self.bg2_row_start, row, 4)
        grid.addWidget(self.bg2_row_end, row, 5)

        return panel

    def _spin(self):
        spin = QtWidgets.QSpinBox()
        spin.setRange(0, 99999)
        spin.setMinimumWidth(76)
        return spin

    def _build_image_panel(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self.image_title = QtWidgets.QLabel("CCD image")
        self.image_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_title.setFont(QtGui.QFont("Arial", 13))
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
        return panel

    def _build_spectrum_panel(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
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
            "Import TIFF",
            "",
            "TIFF files (*.tif *.tiff);;All files (*.*)",
        )
        if not path:
            return
        try:
            self.imm = _read_tiff_array(path)
            self.filepaths = [path]
            self._after_image_loaded("Raw image")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Import TIFF", str(exc))

    def import_tiff_stack(self):
        paths, _selected = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Import TIFF Stack",
            "",
            "TIFF files (*.tif *.tiff);;All files (*.*)",
        )
        if not paths:
            return
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
            self._after_image_loaded(title)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Import TIFF Stack", str(exc))

    def _after_image_loaded(self, title):
        self.immm = None
        self.spect_processor = None
        self.raw_gap_mask = None
        self.processed_gap_mask = None
        self.processed_detector_background_mask = None
        self.run_label.setText(_run_label_from_paths(self.filepaths))
        self._set_spin_limits(self.imm.shape)
        self._set_default_regions(self.imm.shape)
        finite = self.imm[np.isfinite(self.imm)]
        if finite.size:
            self.vmin_spin.blockSignals(True)
            self.vmax_spin.blockSignals(True)
            self.vmin_spin.setValue(float(np.nanpercentile(finite, 1)))
            self.vmax_spin.setValue(float(np.nanpercentile(finite, 99.5)))
            self.vmin_spin.blockSignals(False)
            self.vmax_spin.blockSignals(False)
        self.display_image(self.imm, title)
        self.statusBar().showMessage(f"Loaded {title}: {self.imm.shape}")

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

    def process_image(self):
        if self.imm is None:
            QtWidgets.QMessageBox.information(self, "Process", "Import a TIFF image first.")
            return
        try:
            tilt = estimate_image_tilt_angle(self.imm)
            self.tilt_label.setText(f"Tilt: {tilt:.2f} deg")
            gap_defaults = self.parm['ccd_gap']
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
            self.display_image(self.immm, f"Processed image (tilt {tilt:.2f} deg)")
            self.statusBar().showMessage(f"Processed image with auto tilt {tilt:.2f} deg")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Process", str(exc))

    def display_image(self, data, title):
        self.image_title.setText(title)
        self.image_item.setImage(
            np.asarray(data, dtype=float),
            autoLevels=False,
            levels=(self.vmin_spin.value(), self.vmax_spin.value()),
        )
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

        for begin, end in self.roi_ranges():
            self._add_region((begin, end + 1), horizontal=True, color=(255, 255, 255, 70), pen=(255, 255, 255))
        for begin, end in self.bg_ranges():
            self._add_region((begin, end + 1), horizontal=True, color=(180, 180, 180, 75), pen=(200, 200, 200))
        col_begin, col_end = self.col_range()
        self._add_region((col_begin, col_end + 1), horizontal=False, color=(255, 255, 255, 28), pen=(255, 255, 255, 160))

        if self.gap_overlay_check.isChecked():
            mask = self.processed_gap_mask
            if mask is not None and mask.shape == data.shape:
                overlay = np.zeros(mask.shape + (4,), dtype=np.ubyte)
                overlay[mask, 0] = 155
                overlay[mask, 3] = 120
                item = pg.ImageItem(overlay)
                item.setZValue(20)
                self.image_plot.addItem(item)
                self._image_region_items.append(item)

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
            QtWidgets.QMessageBox.information(self, "Auto BG", "Process an image before selecting background.")
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
        self.spect_processor.toggle_gap_correction(self.gap_check.isChecked())
        self.spect_processor.set_gap_correction_params(
            max_width=gap_defaults.get('max_width', 8),
            drop_fraction=gap_defaults.get('drop_fraction', 0.65),
        )
        self.spect_processor.toggle_background_subtraction(self.bg_check.isChecked())
        if self.bg_check.isChecked():
            col_begin, col_end = self.col_range()
            bg_rois = [
                (row_begin, row_end + 1, col_begin, col_end + 1)
                for row_begin, row_end in self.bg_ranges()
            ]
            self.spect_processor.set_background_rois(bg_rois)

    def plot_spectrum(self):
        if self.spect_processor is None:
            QtWidgets.QMessageBox.information(self, "Plot", "Process an image before plotting a spectrum.")
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
            self._draw_spectrum(x_data, y_data)
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

    def _draw_spectrum(self, x_data, y_data):
        self._clear_spectrum()
        self._legend = pg.LegendItem(offset=(-12, 12), labelTextColor=(40, 40, 40))
        self._legend.setParentItem(self.spectrum_plot.getViewBox())
        self._legend.anchor((1, 0), (1, 0), offset=(-12, 12))

        line = pg.PlotDataItem(x_data, y_data, pen=pg.mkPen('#202020', width=1.6))
        self.spectrum_plot.addItem(line)
        self._spectrum_items.append(line)
        self._legend.addItem(line, self.run_label.text())

        gap_mask = getattr(self.spect_processor, 'last_gap_mask', None)
        if gap_mask is not None and len(gap_mask) == len(y_data) and np.any(gap_mask):
            finite = y_data[np.isfinite(y_data)]
            if finite.size:
                y_min = float(np.nanmin(finite))
                y_max = float(np.nanmax(finite))
                span = max(y_max - y_min, np.finfo(float).eps)
                marker_y = y_min - 0.08 * span
            else:
                marker_y = 0.0
            marker = pg.ScatterPlotItem(
                x_data[np.asarray(gap_mask, dtype=bool)],
                np.full(int(np.count_nonzero(gap_mask)), marker_y),
                symbol='t',
                size=8,
                brush=pg.mkBrush(210, 70, 70, 210),
                pen=pg.mkPen(210, 70, 70, 210),
            )
            self.spectrum_plot.addItem(marker)
            self._spectrum_items.append(marker)
            self._legend.addItem(marker, "Gap fill")

        finite_x = x_data[np.isfinite(x_data)]
        finite_y = y_data[np.isfinite(y_data)]
        if finite_x.size and finite_y.size:
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

    def save_spectrum_png(self):
        if self.last_x.size == 0:
            QtWidgets.QMessageBox.information(self, "Save PNG", "Plot a spectrum first.")
            return
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save spectrum PNG",
            "qt_spectrum.png",
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
            QtWidgets.QMessageBox.information(self, "Save SVG", "Plot a spectrum first.")
            return
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save spectrum SVG",
            "qt_spectrum.svg",
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
