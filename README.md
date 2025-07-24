# IXE (XES Analyzer)

IXE is a Python tool for image processing and spectrum analysis of X-ray emission spectra (XES) collected at the LCLS facility.

## Features

- **TIFF Data Import**  
  Load TIFF files for processing.
- **Image Processing**  
  Thresholding, contrast adjustment, and ROI selection.
- **Spectrum Analysis**  
  Extract and plot spectra from image data with customizable line styles.
- **Integrated Absolute Difference (IAD)**  
  Compare ROI spectra with reference and calculate IAD values.
- **User-Friendly GUI**  
  Built with Tkinter, displaying file paths and interactive controls.

## Installation

### 1. Prerequisites

- **Python 3.6+**  
  Verify your Python version:
  ```bash
  python --version
  ```
- **Tkinter (for GUI)**
  Usually included with Python. 

### 2. Install Dependencies

  ```bash
  pip install numpy scipy matplotlib Pillow scikit-image
  ```
### 3. Install IXE Package

  ```bash
  pip install git+https://github.com/Xuehui-Wei/IXE.git
  ```

## How to Run?
  Run the main interface directly:
  ```bash
  python IXE/xes_analyzer.py
  ```

## Example
For interactive tutorials, see the Jupyter Notebooks: See interactive tutorials:
- `XES_2024.ipynb`
