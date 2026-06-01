# IXE XES Analyzer

IXE is a Python GUI package for processing CCD images and analyzing X-ray
emission spectra (XES), especially K-beta spectra collected from LCLS-style TIFF
detector images.

The current recommended interface is the Qt/pyqtgraph analyzer in
`IXE/qt_analyzer.py`.  The older Tk/Matplotlib interface is still kept as a
legacy fallback in `IXE/xes_analyzer.py`.

## Main Features

- Import one TIFF image or a TIFF stack and sum the stack into one displayed CCD
  image.
- Automatically estimate image tilt, or apply a manual tilt angle when the
  automatic estimate is not reliable.
- Detect the central CCD cross/gap mask without deleting detector rows or
  columns.
- Select one or two ROI row strips, ROI columns, and one or two background row
  strips.
- Extract sharp, resizable spectra with pyqtgraph instead of a fixed Matplotlib
  bitmap display.
- Apply CCD gap correction and optional background subtraction during spectrum
  extraction.
- Fit K-beta spectra with pseudo-Voigt or Lorentzian components, including a
  physical-fit option and optional linear tail baseline correction.
- Save peak-fit text files for later reference/IAD analysis.
- Build an averaged reference spectrum from multiple fitted references after
  aligning their Kbeta 1,3 peak positions.
- Calculate full-spectrum IAD and satellite IAD against fitted or averaged
  references.
- Calibrate pixel index to energy from a calibration spectrum.
- Save and reopen full project sessions with images, masks, spectra, fits,
  references, IAD values, and calibration state.

## Installation

Use Python 3.9 or newer if possible.  From the repository root:

```bash
pip install -r requirements.txt
```

For project save/open, `h5py` is recommended:

```bash
pip install h5py
```

Editable local install:

```bash
pip install -e .
```

## Run The Current Qt App

From the repository root:

```bash
python IXE/qt_analyzer.py
```

After installation, the same Qt app can also be started with:

```bash
xes-analyzer
```

or:

```bash
xes-analyzer-qt
```

## Legacy Tk App

The old Tk/Matplotlib app is still available for comparison or fallback:

```bash
python IXE/xes_analyzer.py
```

or:

```bash
xes-analyzer-tk
```

New development is focused on the Qt/pyqtgraph app.

## Documentation

- [USER_MANUAL.md](USER_MANUAL.md): full GUI workflow.

## Typical Workflow

1. Import a TIFF image or TIFF stack.
2. Process the CCD image with auto tilt, or enter a manual tilt if needed.
3. Inspect the gap mask overlay.
4. Select ROI rows, ROI columns, and background rows.
5. Plot the extracted spectrum.
6. Fit the K-beta spectrum and save the peak-fit file.
7. Import one or more fitted references for IAD, or build an averaged reference.
8. Calculate full-spectrum IAD and satellite IAD.
9. Calibrate to energy if needed.
10. Save the project and exported text/image files.

## Local Examples

Local notebooks and older analysis examples can be kept in an `example/` folder,
but that folder is ignored for the GitHub release.
