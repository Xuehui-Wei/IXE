# IXE XES Analyzer User Manual

This manual describes the current IXE graphical workflow for CCD TIFF image
processing, ROI spectrum extraction, K-beta peak fitting, IAD comparison,
calibration, and project save/open.

## 1. Start The Software

Run the current Qt/pyqtgraph GUI from the repository root:

```bash
python IXE/qt_analyzer.py
```

If IXE was installed with `pip install -e .`, you can also start the same Qt
GUI with:

```bash
xes-analyzer
```

The older Tk/Matplotlib GUI is still available as a legacy fallback:

```bash
python IXE/xes_analyzer.py
```

The main Qt window has four controller tabs at the top and two display panels
below:

- **Image Processing**: import TIFF files, process/tilt CCD images, inspect the
  CCD gap mask, save processed images, and save/open full projects.
- **Spectrum Analysis**: define ROI/background ranges, plot spectra, enable CCD
  gap correction and background removal, fit peaks, build averaged references,
  and save spectra.
- **IAD Calculation**: import reference spectra and calculate full-spectrum or
  satellite-region IAD.
- **Calibration**: import an energy/intensity calibrant, fit it, calculate the
  pixel-to-energy map, and save calibrated spectra.

## 2. Recommended Complete Workflow

1. Click **Import TIFF** for one detector image, or **Import TIFF Stack** to sum
   several images into one displayed CCD image.
2. Set image display contrast with `vmin`, `vmax`, and `Cmap` if needed.
3. Click **Process>**.
   - The software estimates the image tilt automatically.
   - The processed image is rotated to make the XES signal closer to level.
   - A center-aware CCD gap mask is generated and rotated with the image.
   - If the automatic tilt is poor, enter a manual tilt angle and click
     **Apply Tilt**.
4. Click **Gap Mask** to inspect the detected CCD gap overlay.
5. In **Spectrum Analysis**, set:
   - **ROI Rows** around the emission signal.
   - **ROI Columns** over the useful horizontal signal range.
   - **BG Rows** over nearby background rows.
6. Keep **Gap Mask (ON)** if the detector gap crosses the signal.
7. Click **BG Remove** if background subtraction is needed.
8. Click **Plot** to extract and display the ROI spectrum.
9. Click **Peak Fit** to fit the K-beta profile.
10. Use **Save pkfit** if this fitted profile should be reused as a reference.
11. If several fitted reference spectra are available, use **Spectrum
    Analysis** -> **Reference** to average them before IAD.
12. For IAD, go to **IAD Calculation**, import a reference with **Import Ref.**,
    then click **Integrated Diff.** or **Satellite Diff.**
13. For energy calibration, go to **Calibration**, import and fit a calibrant,
    calculate the map, apply it, then save with **Save Cal.**
14. Click **Save Project** or the top **Save** button to store the whole
    analysis session.

## 3. Image Processing Panel

### Import TIFF

Loads a `.tif` or `.tiff` detector image. The run number is read from file names
containing a pattern such as `Run_288`.

### Import TIFF Stack

Loads multiple TIFF files with the same pixel shape and sums their pixel values
into one working image. Use this when several images were collected for the same
run to improve counting statistics.

The display title and file path note show that a stack was loaded.

### vmin, vmax, Cmap

These controls affect image display contrast only. They do not threshold or
delete image pixels.

Available colormaps include:

- `viridis`
- `plasma`
- `inferno`
- `magma`
- `cividis`
- `gray`
- `Greys`
- `turbo`

### Auto Tilt / Manual Tilt

After **Process>**, this readonly field shows the automatically estimated image
tilt angle in degrees.

If the auto tilt is distorted by strong noise or side artifacts, type a manual
angle into the tilt entry and click **Apply Tilt**. The software will reuse the
same image-processing pipeline with the manual angle.

### Process>

Runs the current image-processing workflow:

- preserves the raw image intensity values;
- estimates detector/image tilt automatically;
- rotates the image by the estimated tilt angle;
- detects the detector-centered gap/cross mask before rotation;
- rotates the gap mask together with the image;
- prepares the processed image for spectrum extraction.

The CCD gap detector does not remove detector rows or columns. It only marks gap
pixels so spectrum extraction can avoid or correct them.

### Gap Mask

Displays the processed image with the CCD gap mask overlaid in red. Use this to
check whether the mask covers the central horizontal and vertical detector gaps
without marking too much normal CCD area.

The mask detector is center-aware. It computes the image center from the current
CCD shape, so it can handle different detector dimensions without hard-coded
center pixel numbers.

### Save CCD

Saves the processed image.

- `.tif` or `.tiff`: saves the processed numeric image data.
- Other image extensions such as `.png` or `.jpg`: saves a rendered display
  image using the current contrast and colormap.

If the CCD view has been zoomed or panned, the saved rendered image follows the
currently displayed view.

### Open Project / Save Project

Open or save a full IXE project file. See [Project Files](#10-project-files).

## 4. Spectrum Analysis Panel

The **Spectrum Analysis** controller has three subpanels:

- **Plotting**: ROI/BG selection, gap correction, background removal, and
  spectrum export.
- **Fitting**: peak-fit model controls, physical-fit constraints, tail baseline,
  fitting output, and fitted-profile export.
- **Reference**: import several saved peak-fit references, align them internally
  by Kbeta 1,3 position, average checked references, and save the averaged
  reference for IAD.

### ROI Rows

Selects the vertical row range integrated to make the spectrum. Choose rows that
cover the full emission band. If the row range cuts through the signal, fake
dips or sharp artifacts can appear in the extracted spectrum.

Use the extra ROI row controls when one CCD image contains two separated signal
strips. Both checked ROI strips are integrated into the same spectrum.

### ROI Columns

Selects the horizontal column range used for the plotted spectrum. This range
sets the displayed `Column Index` axis after extraction.

### BG Rows

Selects rows used for background subtraction when **BG Remove** is enabled. The
background rows should be near the signal but should not include the signal.

Two background strips can be used. This is usually better than one strip when
the local detector background changes above and below the signal.

### Plot

Extracts the ROI spectrum from the processed image and plots it in the Spectrum
display panel.

### Gap Mask (ON/OFF)

Controls whether CCD gap correction is used during spectrum extraction.

- **ON**: masked gap pixels are handled during extraction/correction.
- **OFF**: the spectrum is plotted without CCD gap correction.

The button changes appearance when active.

### BG Remove

Toggles background subtraction. The button changes appearance when active.

### Pick Color, Line Style, Line Width

Controls the plotted current spectrum style.

### Peak Fit

Fits the current ROI spectrum with the three-peak K-beta model:

- Kbeta prime: satellite peak
- Kbeta res: intermediate/residual peak
- Kbeta 1,3: main peak

The plot shows:

- original spectrum;
- gap-corrected spectrum, when available;
- total peak fit;
- the three fitted components.

The **Fit Model** controls provide pseudo-Voigt and Lorentzian options. The
default model is pseudo-Voigt.

Recommended fitting options:

- **Physical Fit**: applies stronger physical constraints so Kbeta 1,3 remains
  the dominant broad main component, Kbeta res does not become unrealistically
  sharp, and Kbeta prime stays in a reasonable satellite region.
- **Tail Baseline**: estimates a linear baseline from the left and right tails,
  subtracts it for fitting, and plots the corrected spectrum with the fitted
  profile. Use this when one tail is clearly higher than the other.

### Save pkfit

Saves the latest fitted peak profile as a tab-delimited text file. This is the
recommended file type for later peak-fit-based IAD comparisons.

The saved file includes:

- metadata for background/gap/ROI settings;
- `X`;
- `Raw Y`;
- `Tail baseline`, when tail baseline correction was used;
- `Corrected Y`, when tail baseline correction was used;
- `Peak fit`;
- the fitted component columns.

### Save Spectrum

Saves the currently displayed ROI spectrum, not necessarily the peak fit. If gap
correction or background removal is active, the saved spectrum reflects the
currently plotted extracted spectrum.

### Save Image

Saves the current spectrum plot as a `.png` image.

### Save SVG

Saves the current spectrum plot as vector-style SVG output from the pyqtgraph
scene. Use this when a sharper editable plot is needed for publication
preparation.

## 5. Spectrum View Buttons

The bottom of the Spectrum panel contains four quick display buttons:

- **Original**: show the raw extracted spectrum.
- **Gap C.**: show the gap-corrected spectrum.
- **Peak fit**: show the peak fit and fitted components.
- **Calibration**: show the calibrated spectrum view, identical to **Apply Cal.**,
  if an energy calibration has already been calculated. If no calibration
  relationship is available, the current spectrum display is left unchanged.

These buttons change only the spectrum display. They do not rerun image
processing.

## 6. Peak Fitting Notes

The current fitting model is intended for first-row transition-metal K-beta XES
profiles and uses a constrained three-component pseudo-Voigt workflow:

- satellite Kbeta prime;
- Kbeta res between satellite and main peak;
- main Kbeta 1,3.

For IAD calculations, the peak-fit spectrum is usually preferred over noisy raw
spectra because it gives a smooth, physically constrained profile. The software
also applies baseline matching and area normalization when comparing imported
peak-fit references to the current peak fit.

Recommended practice:

- Use the same ROI and background logic for all compared runs.
- Keep **Gap Mask (ON)** when the detector gap crosses the signal.
- Inspect the peak fit visually before saving or using it for IAD.
- If the fit looks poor, first check ROI rows, BG rows, and the gap mask.

## 7. Reference Averaging Panel

Use **Spectrum Analysis** -> **Reference** when several reference runs are
available and no single reference is ideal.

### Why Average References?

Different reference runs can have small Kbeta 1,3 peak-position shifts and
different satellite noise. Directly averaging them without alignment would
broaden the main peak and distort the IAD baseline. The Reference panel avoids
that by aligning references internally before averaging.

### Reference Average Workflow

1. Save each reference run with **Save pkfit** after peak fitting.
2. Go to **Spectrum Analysis** -> **Reference**.
3. Click **Import** and select one or more `pkfit` text files.
4. Inspect the table:
   - **Use** controls whether a spectrum is included.
   - **Run** shows the run number or reference label.
   - **Kbeta 1,3 px** shows the fitted main-peak position.
   - **Shift** shows the alignment shift used internally.
5. Check only the references that should contribute to the average.
6. Click **Average**.
7. Inspect the display:
   - individual reference spectra are shown at their original positions;
   - the averaged reference is shown after backend alignment and averaging;
   - the shaded band shows mean plus/minus one standard deviation where
     available.
8. Click **Save** to export the averaged reference.

### Averaging Method

The software:

1. reads each reference peak-fit spectrum;
2. finds its fitted Kbeta 1,3 peak position;
3. chooses the median Kbeta 1,3 position as the target;
4. shifts each reference spectrum so its Kbeta 1,3 lands at the target;
5. interpolates the shifted spectra back onto the common x grid;
6. fills uncovered edges with `NaN`, not zero;
7. averages with `nanmean`;
8. area-normalizes the final averaged reference before saving.

Using `NaN` at shifted edges avoids artificially pulling the averaged edge
intensity down to zero.

### Saved Average Reference

The saved average reference is written in a `Save pkfit`-compatible text format
so it can be imported directly in **IAD Calculation**. The file includes:

- `Reference average True`;
- `Area normalized True`;
- number of averaged spectra;
- target Kbeta 1,3 position;
- input run numbers and shifts;
- `X` and `Peak fit` columns for IAD.

When this file is imported in the IAD panel, the legend/table label is shown as
`Ref. average` instead of the full file name.

## 8. IAD Calculation Panel

### Import Ref.

Imports a reference spectrum. The recommended reference is a `Save pkfit` text
file from another run or an averaged reference saved from the **Reference**
subpanel.

Supported reference styles:

- IXE peak-fit `.txt` files saved by **Save pkfit**;
- simple two-column text-like spectra;
- `.chi` spectra.

For IXE peak-fit files, the software uses the `Peak fit` column for plotting and
IAD. It ignores smoothing labels from older file versions.

### Reference List

Imported references appear in the IAD panel with:

- checkbox for visibility;
- color marker;
- IAD value field;
- satellite IAD value field.

### Remove

Removes the selected reference spectrum.

### Color

Changes the selected reference line color.

### Integrated Diff.

Calculates and plots the integrated absolute difference between the current
spectrum and selected reference:

```text
IAD = integral(abs(current - reference))
```

For peak-fit references, the comparison uses fitted peak profiles. Before IAD,
the current and reference spectra are aligned to a common x grid, baseline
matched when enabled, and area-normalized over the valid comparison range.

### Satellite Diff.

Calculates the IAD only over the satellite side of the spectrum. The current and
reference spectra are aligned by their main peak, baseline matched when enabled,
area-normalized over the valid comparison range, plotted together, and only the
satellite difference area is shaded.

Use **Spectra Cross** to define the window where the transition/crossing point
between spectra should be found. The crossing search works no matter which
spectrum is shifted left or right relative to the other.

### Spectra Cross

Two entries defining the index range used to search for the current/reference
crossing point in the satellite region.

### Eye Ball Cross

Optional manual crossing index. Leave blank to let the software find the
crossing automatically from **Spectra Cross**.

### Save IAD

Saves calculated IAD results as a text file. The output includes the current run,
reference run or reference label, full-spectrum IAD, satellite IAD, satellite
cross point, and reference file path.

## 9. Calibration Panel

The calibration workflow maps current-run pixel index to energy using a
calibrant spectrum with known energy on the x-axis.

### Calibration File Format

Use a `.csv` or `.txt` file with at least two numeric columns. A header row is
recommended.

Example CSV:

```csv
Energy,Intensity
7635.0,0.0012
7635.2,0.0015
7635.4,0.0021
```

Example tab-delimited text:

```text
Energy	Intensity
7635.0	0.0012
7635.2	0.0015
7635.4	0.0021
```

The first column is treated as the calibrant x-axis, usually energy. The second
usable intensity-like column is treated as y. Column names such as `Intensity`,
`Y`, or `Peak fit` are accepted.

### Calibration Workflow

1. Plot and peak-fit the current run:
   - **Spectrum Analysis** -> **Plot**
   - **Peak Fit**
2. Go to **Calibration**.
3. Click **Import Cal.** and select the two-column calibrant file.
4. Optional: enter a **Cal. fit range** if only part of the calibrant should be
   fitted.
5. Click **Fit Cal.** to fit the calibrant spectrum.
6. Click **Calc Map**.
   - The map uses Kbeta prime and Kbeta 1,3 as calibration anchors.
   - Kbeta res is shown as a residual quality check.
7. Click **Apply Cal.** to display the current spectrum on the calibrated energy
   axis.
8. Click **Compare Cal.** to overlay the calibrant fit and the calibrated
   current peak fit for visual checking.
9. Click **Save Cal.** to export the calibrated spectrum data.

### Calibration Equation

The software calculates:

```text
E = slope * pixel + intercept
```

The displayed Kbeta res residual shows how far the intermediate peak lies from
the two-anchor calibration. A small residual usually indicates better agreement
between current and calibrant peak geometry.

### Save Cal.

Saves the data used by **Apply Cal.** as a tab-delimited text file. When a
current peak fit exists, the file includes:

- `Energy`;
- current intensity interpolated onto the fitted energy axis;
- `Peak fit`;
- fitted component columns.

## 10. Project Files

Use **Save Project** or the top-row **Save** button to save the full analysis
state to a single `.ixeproj` file. The project file is an HDF5 container with a
JSON manifest and numerical arrays.

Saved project contents include:

- raw image;
- processed/tilted image;
- raw and processed CCD gap masks;
- ROI rows and columns;
- background rows;
- current plotted spectrum;
- gap/background settings;
- peak-fit profile and components;
- imported reference spectra;
- averaged reference spectra when present;
- IAD and satellite IAD values;
- calibration spectrum, fit, equation, and calibrated display data when present.

Use **Open Project** to restore the saved session.

Recommended practice:

- Save individual text exports for publication/plotting.
- Save `.ixeproj` files for continuing or auditing the full GUI analysis.

## 11. Common Output Files

### Processed Image

Created by **Save CCD** in the Image Processing panel.

### Spectrum Text

Created by **Save Spectrum**. Stores the currently plotted extracted spectrum.

### Peak-Fit Text

Created by **Save pkfit**. Recommended for reference import and peak-fit IAD.

### Averaged Reference Text

Created by **Spectrum Analysis** -> **Reference** -> **Save**. Written in a
`Save pkfit`-compatible format so it can be imported by **Import Ref.** for IAD.

### IAD Results Text

Created by **Save IAD**. Stores full-spectrum IAD, satellite IAD, satellite
cross point, and reference identity.

### Spectrum Image / SVG

Created by **Save Image** or **Save SVG** in the Spectrum Analysis panel.

### Calibrated Spectrum Text

Created by **Save Cal.**. Stores the calibrated energy-axis output from the
Calibration workflow.

### Project File

Created by **Save Project**. Uses the `.ixeproj` extension.

## 12. Troubleshooting

### The spectrum has a sharp fake dip near the CCD gap

Check these items:

- Click **Gap Mask** and confirm the mask covers the detector gap.
- Keep **Gap Mask (ON)** while plotting.
- Make sure ROI rows fully cover the signal.
- Avoid choosing background rows that include the signal or cross a different
  gap geometry.

### The gap mask marks too much of the image

Load the TIFF again, click **Process>**, then inspect **Gap Mask**. The current
detector mask logic is designed to prefer the central horizontal/vertical CCD
gap structure. If the image is unusual, verify the overlay before using it for
IAD.

If the red cross appears tilted because the auto tilt is wrong, enter a manual
tilt angle and click **Apply Tilt**. If the red cross is far from the detector
center, reload the image and process again; the detector-center search uses the
current image dimensions rather than a fixed pixel number.

### Peak fit does not match the raw spectrum

First check the extracted spectrum:

- ROI rows should contain the full signal width.
- BG rows should be signal-free.
- Gap correction should be enabled if the gap crosses the spectrum.
- The current run and reference should be processed with similar choices.

Then rerun **Plot** and **Peak Fit**.

### Imported reference appears as a vertical line

Use a current **Save pkfit** output when importing peak-fit references. The
software repairs/matches the x-axis for IXE peak-fit files, but older or
nonstandard files should still contain a numeric `X` column and a usable y
column.

### Calibration file cannot be imported

Check that the file has at least two numeric columns and at least eight finite
data points. A header like `Energy,Intensity` is recommended for CSV files.

### Save dialog crashes on macOS

The current save dialogs avoid Tk file-type filters that caused native macOS/Tk
crashes in some environments. If a crash occurs after local edits, check whether
`filetypes=` was reintroduced into a save dialog.
