# -*- coding: utf-8 -*-
# spectrum_utils.py
import numpy as np
import matplotlib.pyplot as plt
import random

class SpectrumProcessor:
    def __init__(self, processed_image, n_moveavg):
        if not isinstance(processed_image, np.ndarray):
            raise ValueError("processed_image must be a numpy array")
        if not isinstance(n_moveavg, int) or n_moveavg <= 0:
            raise ValueError("n_moveavg must be a positive integer")
        self.processed_image = processed_image.copy()
        self.background_spectrum = None
        self.bg_subtracted = False  # Track subtraction state internally
        self.n_moveavg = n_moveavg 

    def set_background_roi(self, row_begin, row_end, col_begin, col_end):
        bg_roi = self.processed_image[row_begin:row_end, col_begin:col_end]
        self.background_spectrum = bg_roi.sum(axis=0)

    def toggle_background_subtraction(self, enable):
        """Toggle whether subtraction should be applied"""
        self.bg_subtracted = enable

    def get_spectrum(self, row_begin, row_end, col_begin, col_end):
        """
        Get spectrum, applying background subtraction if enabled
        Note: No subtract_bg parameter - uses internal state instead
        """
        roi = self.processed_image[row_begin:row_end, col_begin:col_end]
        spectrum = roi.sum(axis=0)
        
        if self.bg_subtracted and self.background_spectrum is not None:
            if len(spectrum) == len(self.background_spectrum):
                return spectrum - self.background_spectrum
            print("Warning: Background length doesn't match spectrum")
        return spectrum
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
        # 5. Find transition point
        transition_point = None
        if i_g is None:
            for i in range(i_l, i_r):
                if spec_align[i] <= spec_r_align[i] and spec_align[i + 1] > spec_r_align[i + 1]:
                    transition_point = i + 1
                    break
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
    
    
