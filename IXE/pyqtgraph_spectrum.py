# -*- coding: utf-8 -*-
"""Optional pyqtgraph spectrum viewer for sharper interactive line plots."""

import numpy as np
from matplotlib.colors import to_rgba


def pyqtgraph_available():
    try:
        import pyqtgraph  # noqa: F401
        from pyqtgraph.Qt import QtWidgets  # noqa: F401
    except Exception:
        return False
    return True


def _load_qtgraph():
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

    return pg, QtCore, QtGui, QtWidgets


def _rgba_to_qcolor(QtGui, color, alpha=None):
    try:
        rgba = np.asarray(color, dtype=float).ravel()
    except (TypeError, ValueError):
        rgba = np.asarray(to_rgba(color), dtype=float).ravel()
    if rgba.size < 3:
        rgba = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    if rgba.size == 3:
        rgba = np.append(rgba, 1.0)
    if np.nanmax(rgba) <= 1.0:
        rgba = rgba * 255.0
    if alpha is not None:
        rgba[3] = float(alpha) * 255.0 if float(alpha) <= 1.0 else float(alpha)
    rgba = np.clip(rgba, 0, 255).astype(int)
    return QtGui.QColor(int(rgba[0]), int(rgba[1]), int(rgba[2]), int(rgba[3]))


def _qt_pen_style(QtCore, linestyle):
    styles = {
        '-': QtCore.Qt.PenStyle.SolidLine,
        '--': QtCore.Qt.PenStyle.DashLine,
        ':': QtCore.Qt.PenStyle.DotLine,
        '-.': QtCore.Qt.PenStyle.DashDotLine,
        'None': QtCore.Qt.PenStyle.NoPen,
        'none': QtCore.Qt.PenStyle.NoPen,
        '': QtCore.Qt.PenStyle.NoPen,
    }
    return styles.get(str(linestyle), QtCore.Qt.PenStyle.SolidLine)


class PyQtGraphSpectrumViewer:
    """A separate Qt/pyqtgraph spectrum window synchronized from Matplotlib axes."""

    def __init__(self, title="Sharp Spectrum"):
        self.pg, self.QtCore, self.QtGui, self.QtWidgets = _load_qtgraph()
        self.app = self.QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = self.QtWidgets.QApplication([])
        self.pg.setConfigOptions(antialias=True, useOpenGL=False)

        self.window = self.QtWidgets.QMainWindow()
        self.window.setWindowTitle(title)
        self.window.resize(900, 620)
        self._setup_toolbar()

        self.layout_widget = self.pg.GraphicsLayoutWidget()
        self.layout_widget.setBackground('w')
        self.window.setCentralWidget(self.layout_widget)

        self.plot_item = self.layout_widget.addPlot(row=0, col=0)
        self.plot_item.setMenuEnabled(True)
        self.plot_item.showGrid(x=True, y=True, alpha=0.12)
        self.plot_item.setMouseEnabled(x=True, y=True)
        self.plot_item.disableAutoRange()
        self.view_box = self.plot_item.getViewBox()
        self.legend = None
        self._items = []
        self._apply_axis_style()

    def _setup_toolbar(self):
        toolbar = self.QtWidgets.QToolBar("Export")
        toolbar.setMovable(False)
        save_png = self.QtWidgets.QAction("Save PNG", self.window)
        save_svg = self.QtWidgets.QAction("Save SVG", self.window)
        save_png.triggered.connect(self.save_png)
        save_svg.triggered.connect(self.save_svg)
        toolbar.addAction(save_png)
        toolbar.addAction(save_svg)
        self.window.addToolBar(toolbar)

    def show(self):
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()
        self.app.processEvents()

    def process_events(self):
        self.app.processEvents()

    def is_visible(self):
        return bool(self.window.isVisible())

    def close(self):
        self.window.close()
        self.app.processEvents()

    def save_png(self):
        from pyqtgraph.exporters import ImageExporter

        path, _selected = self.QtWidgets.QFileDialog.getSaveFileName(
            self.window,
            "Save spectrum PNG",
            "sharp_spectrum.png",
            "PNG image (*.png)",
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path = f"{path}.png"
        exporter = ImageExporter(self.plot_item)
        width = max(int(self.plot_item.width()), 2400)
        exporter.parameters()['width'] = width
        exporter.export(path)

    def save_svg(self):
        from pyqtgraph.exporters import SVGExporter

        path, _selected = self.QtWidgets.QFileDialog.getSaveFileName(
            self.window,
            "Save spectrum SVG",
            "sharp_spectrum.svg",
            "SVG vector (*.svg)",
        )
        if not path:
            return
        if not path.lower().endswith(".svg"):
            path = f"{path}.svg"
        exporter = SVGExporter(self.plot_item)
        exporter.export(path)

    def _apply_axis_style(self):
        font = self.QtGui.QFont("Arial", 11)
        label_style = {'font-family': 'Arial', 'font-size': '13pt', 'color': '#222'}
        for axis_name in ('left', 'bottom'):
            axis = self.plot_item.getAxis(axis_name)
            axis.setTickFont(font)
            axis.setStyle(tickLength=5, tickTextOffset=4)
            axis.setPen(self.pg.mkPen('#222', width=1))
            axis.setTextPen(self.pg.mkPen('#222'))
        self.plot_item.setLabel('bottom', 'Column Index', **label_style)
        self.plot_item.setLabel('left', 'Normalized intensity', **label_style)

    def _clear(self):
        for item in self._items:
            self.plot_item.removeItem(item)
        self._items = []
        if self.legend is not None:
            self.legend.clear()
            self.plot_item.removeItem(self.legend)
            self.legend = None

    def _add_legend(self):
        self.legend = self.pg.LegendItem(offset=(-12, 12), labelTextColor=(40, 40, 40))
        self.legend.setParentItem(self.view_box)
        self.legend.anchor((1, 0), (1, 0), offset=(-12, 12))
        return self.legend

    def _line_to_item(self, line):
        x_data = np.asarray(line.get_xdata(orig=False), dtype=float)
        y_data = np.asarray(line.get_ydata(orig=False), dtype=float)
        if x_data.size == 0 or y_data.size == 0 or x_data.size != y_data.size:
            return None
        finite = np.isfinite(x_data) & np.isfinite(y_data)
        if not np.any(finite):
            return None

        color = _rgba_to_qcolor(self.QtGui, line.get_color(), alpha=line.get_alpha())
        width = max(float(line.get_linewidth()) * 1.6, 1.0)
        style = _qt_pen_style(self.QtCore, line.get_linestyle())
        pen = self.pg.mkPen(color=color, width=width, style=style)
        item = self.pg.PlotDataItem(x_data, y_data, pen=pen)
        return item

    def _scatter_to_item(self, collection):
        if not hasattr(collection, 'get_offsets'):
            return None
        offsets = np.asarray(collection.get_offsets(), dtype=float)
        if offsets.ndim != 2 or offsets.shape[1] < 2 or offsets.size == 0:
            return None
        finite = np.isfinite(offsets[:, 0]) & np.isfinite(offsets[:, 1])
        if not np.any(finite):
            return None

        facecolors = collection.get_facecolors()
        if len(facecolors):
            color = _rgba_to_qcolor(self.QtGui, facecolors[0], alpha=collection.get_alpha())
        else:
            color = self.QtGui.QColor(210, 70, 70, 210)
        brush = self.pg.mkBrush(color)
        pen = self.pg.mkPen(color=color, width=1)
        return self.pg.ScatterPlotItem(
            offsets[finite, 0],
            offsets[finite, 1],
            symbol='t',
            size=8,
            pen=pen,
            brush=brush,
            pxMode=True,
        )

    def update_from_axes(self, axes, xlabel="Column Index", ylabel="Normalized intensity", show_legend=True):
        self._clear()
        self.plot_item.setLabel('bottom', xlabel)
        self.plot_item.setLabel('left', ylabel)
        if show_legend:
            self._add_legend()

        x_values = []
        y_values = []
        used_labels = set()

        for line in axes.lines:
            item = self._line_to_item(line)
            if item is None:
                continue
            self.plot_item.addItem(item)
            self._items.append(item)
            label = str(line.get_label())
            if show_legend and label and not label.startswith('_') and label not in used_labels:
                self.legend.addItem(item, label)
                used_labels.add(label)
            x_data = np.asarray(line.get_xdata(orig=False), dtype=float)
            y_data = np.asarray(line.get_ydata(orig=False), dtype=float)
            finite = np.isfinite(x_data) & np.isfinite(y_data)
            if np.any(finite):
                x_values.append(x_data[finite])
                y_values.append(y_data[finite])

        for collection in axes.collections:
            item = self._scatter_to_item(collection)
            if item is None:
                continue
            self.plot_item.addItem(item)
            self._items.append(item)
            label = str(collection.get_label())
            if show_legend and label and not label.startswith('_') and label not in used_labels:
                self.legend.addItem(item, label)
                used_labels.add(label)
            offsets = np.asarray(collection.get_offsets(), dtype=float)
            finite = np.isfinite(offsets[:, 0]) & np.isfinite(offsets[:, 1])
            if np.any(finite):
                x_values.append(offsets[finite, 0])
                y_values.append(offsets[finite, 1])

        if x_values and y_values:
            x_all = np.concatenate(x_values)
            y_all = np.concatenate(y_values)
            x_min, x_max = float(np.nanmin(x_all)), float(np.nanmax(x_all))
            y_min, y_max = float(np.nanmin(y_all)), float(np.nanmax(y_all))
            x_pad = max((x_max - x_min) * 0.03, 1.0)
            y_pad = max((y_max - y_min) * 0.08, 1e-6)
            self.view_box.setLimits(
                xMin=x_min - x_pad,
                xMax=x_max + x_pad,
                yMin=y_min - y_pad,
                yMax=y_max + y_pad,
            )
            self.view_box.setRange(
                xRange=(x_min - x_pad, x_max + x_pad),
                yRange=(y_min - y_pad, y_max + y_pad),
                padding=0.0,
            )
        self.app.processEvents()


def axes_have_visible_data(axes):
    for line in axes.lines:
        if len(line.get_xdata(orig=False)) and len(line.get_ydata(orig=False)):
            return True
    for collection in axes.collections:
        if hasattr(collection, 'get_offsets') and len(collection.get_offsets()):
            return True
    return False
