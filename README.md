# IXE XES Analyzer

IXE is a Python GUI package for processing CCD images and analyzing X-ray
emission spectra (XES), especially K-beta spectra collected from LCLS-style TIFF
detector images.

The current recommended interface is the Qt/pyqtgraph app in
`IXE/xes_app.py`.  The older Tk/Matplotlib interface is still kept as a
legacy fallback in `IXE/xes_analyzer.py`.

## Main Features

- Import one image or an image stack and initialize spectrum extraction
  immediately after import.
- Apply automatic tilt correction, or use manual tilt when the automatic
  estimate is not reliable.
- Detect the central CCD cross/gap mask without deleting detector rows or
  columns.
- Select one or two ROI row strips, ROI columns, and one or two background row
  strips.
- Extract sharp, resizable spectra with pyqtgraph, with corrected spectra
  labeled consistently in the plot legend.
- Apply CCD gap correction and optional background subtraction during spectrum
  extraction.
- Fit K-beta spectra with pseudo-Voigt or Lorentzian components, including a
  physical-fit option and optional linear tail baseline correction.
- Export peak-fit profiles and peak parameters for later reference/IAD analysis.
- Build an averaged reference spectrum from multiple fitted references after
  aligning their Kbeta 1,3 peak positions.
- Calculate full-spectrum IAD and satellite IAD against fitted or averaged
  references.
- Calibrate pixel index to energy from a calibration spectrum.
- Save and reopen full project sessions with images, masks, spectra, fits,
  references, IAD values, and calibration state. Reusing **Save Project**
  overwrites the active project file after the first save/open.

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
python IXE/xes_app.py
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
2. Plot the spectrum immediately, or click **Tilt Correction** first when the
   image needs automatic tilt correction.
3. Select ROI rows, ROI columns, and background rows as needed.
4. Use **Gap Mask** and **BG Remove** checkboxes for corrected extraction.
5. Fit the K-beta spectrum in **Peak Fitting** and export PKfit/parameter files.
6. Import one or more fitted references, or build an averaged spectrum reference.
7. Calculate **IAD** and **Satellite IAD**, including Monte Carlo error
   estimates when needed.
8. Import a calibration spectrum, calculate the map, compare/apply calibration,
   and export calibrated results.
9. Use **Save Project** to save the full session, then continue saving to the
   same project file unless you open/export a different one.

## Local Examples

Local notebooks and older analysis examples can be kept in an `example/` folder,
but that folder is ignored for the GitHub release.
