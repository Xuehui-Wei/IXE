#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image #import glob
import json
import scipy.ndimage
import glob
from collections import OrderedDict
import numpy as np
import matplotlib.ticker as mtick

class SpectrumAnalyzer:
    def __init__(self):
        self.parm = {
            'roi' : 0, # 0 for both, 1 for top, 2 for bottom
            'threshold': 1, # for the pixel count.  do not increase more than 1.
            'tilt_cor': -0.4, # -0.4, from the Fe metal calibration
            'pix_shift': 6, # shifting bottom signal for better match. must be integer
            'n_moveavg':21, # degree of moving average
            'vmax': 200,
            ##
            'ka_r_l': 467,
            'ka_r_r': 487,
            'ka_c_l': 0,
            'ka_c_r': 300,
            'ka_r_l_bg': 445,
            'ka_r_r_bg': 465,
            'ka_r_ll': 500, 
            'ka_r_rr':540,

            'kb_r_l': None,
            'kb_r_r': None,
            'kb_c_l': 320,
            'kb_c_r': 600,
            'kb_r_l_bg': None,
            'kb_r_r_bg': None,
            #'kb_delete': None,

            #### 2022_XES
            'r_l': 467,
            'r_r': 487,
            'c_l': 5,
            'c_r': 300,
            'r_ll': 500, 
            'r_rr':540,
            'r_l_bg': 445,
            'r_r_bg': 465,
            'r_ll_bg': 488,
            'r_rr_bg': 499,
            'r_lll_bg': 541,
            'r_rrr_bg': 581,
        }
    
    def update_parameters(self, run_num):
        if int(run_num) <98: 
            self.parm['ka_r_l']=210
            self.parm['ka_r_r']=230
            
            self.parm['ka_r_l_bg']=199
            self.parm['ka_r_r_bg']= 214
            self.parm['kb_r_l']= 261
            self.parm['kb_r_r']= 282
            
            self.parm['kb_r_l_bg']= 239
            self.parm['kb_r_r_bg']= 260
            #self.parm['kb_delete']= [74, 75, 76, 77, 78, 79, 80, 81]
        if int(run_num) >= 98:

            self.parm['ka_r_l'] = 510
            self.parm['ka_r_r'] = 528

            self.parm['ka_r_l_bg'] = 489
            self.parm['ka_r_r_bg'] = 509
            self.parm['kb_r_l'] = 556
            self.parm['kb_r_r'] = 575

            self.parm['kb_r_l_bg'] = 533
            self.parm['kb_r_r_bg'] = 553
            #self.parm['kb_delete'] = [66, 67, 68, 69, 70,71, 72, 73]

    def parm_test(self):
        '''
        This function is used for Xuehui's testing only
        '''
        ka_r_l_value = self.parm['ka_r_l']
        kb_c_r_value = self.parm['kb_c_r']


        return ka_r_l_value, kb_c_r_value
                  

    def read_info_json(self, json_filen):
        f = open(json_filen)
        meta_data = json.load(f)
        f.close()
        return meta_data
    def get_norm_spectrum(self,spectrum):
        spect = (spectrum - spectrum.min()) # / (spectrum.max() - spectrum.min())
        norm_fac = spect.sum()
        return spect/norm_fac
        
    def get_roi_spectrum(self, spect):
        spect1 = spect[self.parm['r_l']:self.parm['r_r'],self.parm['c_l']:self.parm['c_r']].sum(axis=0)
        spect2 = spect[self.parm['r_ll']:self.parm['r_rr'],self.parm['c_l']:self.parm['c_r']].sum(axis=0)
        pix_s = self.parm['pix_shift']
        spect = spect1[0:-1*pix_s] + spect2[pix_s:]
        if self.parm['roi'] == 0:
            return spect
        elif self.parm['roi'] == 1:
            return spect1
        else:
            return spect2
        
    def get_roi_spectrum_ka(self, spect):
        spect = spect[self.parm['ka_r_l']: self.parm['ka_r_r'],self.parm['ka_c_l']:self.parm['ka_c_r']].sum(axis=0)

        return spect
    def get_roi_spectrum_kb(self, spect):
        spect = spect[self.parm['kb_r_l']: self.parm['kb_r_r'], self.parm['kb_c_l']:self.parm['kb_c_r']].sum(axis=0)
        #spect= np.delete(spect, self.parm['kb_delete'])

        return spect
    def lin_bg_sub(self, spectrum):
        y_0 = spectrum[:11].mean()
        y_1 = spectrum[-11:].mean()
        x_n = len(spectrum)
        y_bg = (y_1 - y_0) / x_n * np.arange(x_n) + y_0
        return spectrum - y_bg

    def bg_sub_spectrum(self,spect):
        bkg1 = spect[self.parm['r_l_bg']: self.parm['r_r_bg'], 
                self.parm['c_l']:self.parm['c_r']].sum(axis=0)
        bkg2 = spect[self.parm['r_ll_bg']: self.parm['r_rr_bg'], 
                self.parm['c_l']:self.parm['c_r']].sum(axis=0)
        bkg3 = spect[self.parm['r_lll_bg']: self.parm['r_rrr_bg'], 
                self.parm['c_l']:self.parm['c_r']].sum(axis=0)
        pix_s = self.parm['pix_shift']
        bkg = bkg1[0:-1*pix_s] + bkg2[0:-1*pix_s] + bkg3[pix_s:]
        return self.get_roi_spectrum(spect) - bkg
        
    def bg_sub_spectrum_ka(self,spect):
        bkg = spect[self.parm['ka_r_l_bg']: self.parm['ka_r_r_bg'], 
                self.parm['ka_c_l']:self.parm['ka_c_r']].sum(axis=0)
        return self.get_roi_spectrum_ka(spect) - bkg

    def bg_sub_spectrum_kb(self,spect):
        bkg = spect[self.parm['kb_r_l_bg']: self.parm['kb_r_r_bg'], 
                self.parm['kb_c_l']:self.parm['kb_c_r']].sum(axis=0)
        #bkg = np.delete(bkg, self.parm['kb_delete'])
        return self.get_roi_spectrum_kb(spect) - bkg
    def moving_average(self, a):
        n = self.parm['n_moveavg']
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    # This function is to plot the 2d image of roi data and roi_ref
    def img_roi(self,run_num, ims, ims_ref): 
        f, axs = plt.subplots(1,2, figsize=(8,3))
        axs[0].imshow(ims[run_num], interpolation='none', vmax =self.parm['vmax'])
        axs[1].imshow(ims_ref[run_num], interpolation='none', vmax =10)
        #axs[0].set_title('Data'); axs[1].set_title('Ref')
        axs[0].hlines(self.parm['ka_r_l'], 0,772, color = 'white', linewidth = 0.5)
        axs[0].hlines(self.parm['ka_r_r'], 0,772, color = 'white', linewidth = 0.5)
        axs[0].hlines(self.parm['kb_r_l'], 0, 772, color = 'white', lw = 0.5)
        axs[0].hlines(self.parm['kb_r_r'], 0, 772, color = 'white', lw = 0.5)
        plt.show()

    # This function is to plot the 2d images of ka and kb signal from roi and roi_ref
    def img_ka_kb(self, run_num, ims, ims_ref):
        f, axs = plt.subplots(2,2, figsize=(8,3))

        axs[0,0].imshow(ims[run_num][self.parm['ka_r_l']:self.parm['ka_r_r'], 
                        self.parm['ka_c_l']:self.parm['ka_c_r']], vmax =self.parm['vmax'] )
        axs[1,0].imshow(ims[run_num][self.parm['kb_r_l']:self.parm['kb_r_r'], 
                        self.parm['kb_c_l']:self.parm['kb_c_r']], vmax =self.parm['vmax'] )
        axs[0,1].imshow(ims_ref[run_num][self.parm['ka_r_l']:self.parm['ka_r_r'], 
                        self.parm['ka_c_l']:self.parm['ka_c_r']], vmax =self.parm['vmax'] )
        axs[1,1].imshow(ims_ref[run_num][self.parm['kb_r_l']:self.parm['kb_r_r'], 
                        self.parm['kb_c_l']:self.parm['kb_c_r']], vmax =self.parm['vmax'] )
        axs[0,0].set_title('Data'); axs[0,1].set_title('Ref')
        plt.show()
        
    def plot_roi(self, run_num, ims, ims_ref, pressure, composition):
        f, axs = plt.subplots(1,4, figsize=(15,4))

        spec_raw = self.get_roi_spectrum(ims[run_num])
        spec_raw_rmbg = self.bg_sub_spectrum(ims[run_num])
        spec = self.moving_average(spec_raw_rmbg)
        spec_ref_raw = self.get_roi_spectrum(ims_ref[run_num])
        spec_ref = self.moving_average(spec_ref_raw)
        n = self.parm['n_moveavg']; mov_n = n//2
        
        x_mov = np.arange(np.trunc(n/2), len(spec)+np.trunc(n/2))
        
        axs[0].plot(spec_raw_rmbg, 'g.-', label = 'raw k-\u03B2')
        axs[0].plot(x_mov, spec, 'r.-', label = 'smooth k-\u03B2')
        axs[0].plot(spec - spec_raw_rmbg[mov_n:-mov_n], 'k.-', label = 'diff')
        axs[0].legend(loc = 'upper left')
        axs[1].plot(spec_ref_raw, 'g.-', label = 'raw ref k-\u03B2')
        axs[1].plot(x_mov, spec_ref, 'r.-', label = 'smooth ref k-\u03B2')
        axs[1].plot(spec_ref - spec_ref_raw[mov_n:-mov_n], 'k.-', label = 'diff')
        axs[1].legend(loc = 'upper left')
        axs[2].plot(self.get_norm_spectrum(spec), 'r.-', label = 'smooth k-\u03B2', zorder = 100)
        axs[2].plot(self.get_norm_spectrum(spec_ref), 'k.-',  label = 'smooth Ref k-\u03B2')
        axs[2].fill_between(np.arange(len(spec)), 0, 
                            self.get_norm_spectrum(spec) - self.get_norm_spectrum(spec_ref),
                           color='r', label = 'Integrated diff')
        axs[2].legend(loc = 'upper left')
        i_max = self.get_norm_spectrum(spec).argmax()
        i_ref_max = self.get_norm_spectrum(spec_ref).argmax()
        axs[3].plot(self.get_norm_spectrum(spec_raw_rmbg), 'r.-', label = 'raw k-\u03B2', zorder = 100)
        axs[3].plot(self.get_norm_spectrum(spec_ref_raw), 'k.-', label = 'raw Ref k-\u03B2')
        axs[3].fill_between(np.arange(len(spec_raw)), 0, 
                            self.get_norm_spectrum(spec_raw_rmbg) - self.get_norm_spectrum(spec_ref_raw),
                           color='r', label = 'Integrated diff')
        axs[3].legend(loc = 'upper left')
        #axs[1].set_xlim(0,300)
        axs[2].set_title('Run #' + run_num + 
                         ', P = ' + str(pressure[run_num]) + ' GPa' +
                        ', X = ' + composition[run_num])
        axs[0].set_title('Data' + run_num)
        axs[1].set_title('Ref' + run_num)
        return f, axs

    
    def plot_roi_ka(self,run_num, ims, ims_ref, pressure, composition):
        f, axs = plt.subplots(1,4, figsize=(15,4))

        spec_raw = self.get_roi_spectrum_ka(ims[run_num])
        spec_raw_rmbg = self.bg_sub_spectrum_ka(ims[run_num])
        spec_ref_raw = self.get_roi_spectrum_ka(ims_ref[run_num])
        spec = self.moving_average(spec_raw_rmbg)
        spec_ref = self.moving_average(spec_ref_raw)

        n = self.parm['n_moveavg']; mov_n = n//2
        x_mov = np.arange(np.trunc(n/2), len(spec)+np.trunc(n/2))
        axs[0].plot(spec_raw_rmbg, 'g.-', label = 'raw k-\u03B1')
        axs[0].plot(x_mov, spec, 'r.-', label = 'smooth k-\u03B1')
        axs[0].plot(spec - spec_raw_rmbg[mov_n:-mov_n], 'k.-', label = 'diff')
        axs[0].legend(loc = 'upper left')
        axs[1].plot(spec_ref_raw, 'g.-', label = 'raw ref k-\u03B1')
        axs[1].plot(x_mov, spec_ref, 'r.-', label = 'smooth ref k-\u03B1')
        axs[1].plot(spec_ref - spec_ref_raw[mov_n:-mov_n], 'k.-', label = 'diff')
        axs[1].legend(loc = 'upper left')
        axs[2].plot(self.get_norm_spectrum(spec), 'r.-', label = 'smooth k-\u03B1', zorder = 100)
        axs[2].plot(self.get_norm_spectrum(spec_ref), 'k.-',  label = 'smooth Ref k-\u03B1')
        axs[2].fill_between(np.arange(len(spec)), 0, 
                            self.get_norm_spectrum(spec) - self.get_norm_spectrum(spec_ref),
                           color='r', label = 'Integrated diff')
        axs[2].legend(loc = 'upper left')
        i_max = self.get_norm_spectrum(spec).argmax()
        i_ref_max = self.get_norm_spectrum(spec_ref).argmax()
        axs[3].plot(self.get_norm_spectrum(spec_raw_rmbg), 'r.-', label = 'raw k-\u03B1', zorder = 100)
        axs[3].plot(self.get_norm_spectrum(spec_ref_raw), 'k.-', label = 'raw Ref k-\u03B1')
        axs[3].fill_between(np.arange(len(spec_raw_rmbg)), 0, 
                            self.get_norm_spectrum(spec_raw_rmbg) - self.get_norm_spectrum(spec_ref_raw),
                           color='r', label = 'Integrated diff')
        axs[3].legend(loc = 'upper left')
        #axs[1].set_xlim(0,300)
        axs[2].set_title('Run #' + run_num + 
                         ', P = ' + str(pressure[run_num]) + ' GPa' +
                        ', X = ' + composition[run_num])
        axs[0].set_title('Data' + run_num)
        axs[1].set_title('Ref' + run_num)
        return f, axs
    
    def plot_roi_kb(self,run_num,ims, ims_ref, pressure, composition):
        f, axs = plt.subplots(1,4, figsize=(17,4))

        spec_raw = self.get_roi_spectrum_kb(ims[run_num])
        spec_raw_rmbg = self.bg_sub_spectrum_kb(ims[run_num])
        spec_ref_raw = self.get_roi_spectrum_kb(ims_ref[run_num])
        spec = self.moving_average(spec_raw_rmbg)
        spec_ref = self.moving_average(spec_ref_raw)

        n = self.parm['n_moveavg']; mov_n = n//2
        x_mov = np.arange(np.trunc(n/2), len(spec)+np.trunc(n/2))
        axs[0].plot(spec_raw_rmbg, 'g.-', label = 'raw k-\u03B2')
        axs[0].plot(x_mov, spec, 'r.-', label = 'smooth k-\u03B2')
        axs[0].plot(spec - spec_raw_rmbg[mov_n:-mov_n], 'k.-', label = 'diff')
        axs[0].legend(loc = 'upper left')
        axs[1].plot(spec_ref_raw, 'g.-', label = 'raw ref k-\u03B2')
        axs[1].plot(x_mov, spec_ref, 'r.-', label = 'raw ref, k-\u03B2')
        axs[1].plot(spec_ref - spec_ref_raw[mov_n:-mov_n], 'k.-', label = 'diff')
        axs[1].legend(loc = 'upper left')
        axs[2].plot(self.get_norm_spectrum(spec), 'r.-', lw = 0.8, label = 'smooth k-\u03B2',zorder = 100)
        axs[2].plot(self.get_norm_spectrum(spec_ref), 'k.-',  lw = 0.8, label = 'smooth Ref k-\u03B2')
        axs[2].fill_between(np.arange(len(spec)), 0, 
                            self.get_norm_spectrum(spec) - self.get_norm_spectrum(spec_ref),
                           color='r', label = 'Integrated diff')
        axs[2].legend(loc = 'upper left')
        i_max = self.get_norm_spectrum(spec).argmax()
        i_ref_max = self.get_norm_spectrum(spec_ref).argmax()
        axs[3].plot(self.get_norm_spectrum(spec_raw_rmbg), 'r.-', label = 'raw k-\u03B2', zorder = 100)
        axs[3].plot(self.get_norm_spectrum(spec_ref_raw), 'k.-', label = 'raw Ref k-\u03B2')
        axs[3].fill_between(np.arange(len(spec_raw_rmbg)), 0, 
                            self.get_norm_spectrum(spec_raw_rmbg) - self.get_norm_spectrum(spec_ref_raw),
                           color='r', label = 'Integrated diff')
        axs[3].legend(loc = 'upper left')
        #axs[1].set_xlim(0,300)
        axs[2].set_title('Run #' + run_num + 
                         ', P = ' + str(pressure[run_num]) + ' GPa' +
                        ', X = ' + composition[run_num])
        axs[0].set_title('Data' + run_num)
        axs[1].set_title('Ref' + run_num)
        return f, axs

    def series_plot(self,r, ims, ims_ref, pressure, composition, std = None):

        f, axs = plt.subplots(1,2, figsize=(10,4))

        spec_raw = self.get_roi_spectrum(ims[r])
        spec_raw_rmbg = self.bg_sub_spectrum(ims[r])
        spec = self.moving_average(spec_raw_rmbg)
        spec_n = self.get_norm_spectrum(self.lin_bg_sub(spec))
        if std: 
            std_r = std[composition[r]]
        else:
            std_r = r
       
        spec_ref_raw = self.get_roi_spectrum(ims_ref[std_r])
        spec_ref = self.moving_average(spec_ref_raw)
        spec_r_n = self.get_norm_spectrum(self.lin_bg_sub(spec_ref))
        

        axs[0].plot(spec_r_n, 'k.-', label = 'smooth Ref k-\u03B2')
        axs[0].plot(spec_n, 'r.-', label = 'smooth k-\u03B2')
        axs[0].fill_between(np.arange(len(spec)), 0, 
                            spec_n - spec_r_n,
                            color='r', label = 'Integrated diff')

        i_max = spec_n.argmax()
        i_ref_max = spec_r_n.argmax()
        axs[1].plot(np.arange(len(spec_r_n))-i_ref_max, spec_r_n, 'k.-', label= 'smooth Ref k-\u03B2' )
        axs[1].plot(np.arange(len(spec_n))-i_max, spec_n, 'r.-', label = 'smooth k-\u03B2')
        f.suptitle('Run #' + r + ', P = ' + 
                   str(pressure[r]) + 'GPa' + ', X = ' + composition[r])
        axs[0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        axs[1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        axs[0].legend();axs[1].legend()
        return f, axs

    def ka_series_plot(self,r, ims, ims_ref, pressure, composition, std = None):

        f, axs = plt.subplots(1,2, figsize=(10,4))

        spec_raw = self.get_roi_spectrum_ka(ims[r])
        spec_raw_rmbg = self.bg_sub_spectrum_ka(ims[r])
        spec = self.moving_average(spec_raw_rmbg)
        spec_n = self.get_norm_spectrum(spec)
        if std: 
            std_r = std[composition[r]]
        else:
            std_r = r
       
        spec_ref_raw = self.get_roi_spectrum_ka(ims_ref[std_r])
        spec_ref = self.moving_average(spec_ref_raw)
        spec_r_n = self.get_norm_spectrum(spec_ref)
        

        axs[0].plot(spec_r_n, 'k.-', label = 'smooth Ref k-\u03B1')
        axs[0].plot(spec_n, 'r.-', label = 'smooth k-\u03B1')
        axs[0].fill_between(np.arange(len(spec)), 0, 
                            spec_n - spec_r_n,
                            color='r', label = 'Integrated diff')

        axs[1].plot(self.get_norm_spectrum(spec_raw_rmbg), 'r.-', label = 'raw k-\u03B1')
        axs[1].plot(self.get_norm_spectrum(spec_ref_raw), 'k.-', label = 'raw Ref_' + std_r + ' k-\u03B1')
        f.suptitle('Run #' + r + ', P = ' + 
                   str(pressure[r]) + 'GPa' + ', X = ' + composition[r])
        axs[0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        axs[1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        axs[0].legend();axs[1].legend()
        return f, axs

    

    def kb_series_plot(self, r, ims, ims_ref, pressure, composition, std = None):

        f, axs = plt.subplots(1,2, figsize=(10,4))

        spec_raw = self.get_roi_spectrum_kb(ims[r])
        spec_raw_rmbg = self.bg_sub_spectrum_kb(ims[r])
        spec = self.moving_average(spec_raw_rmbg)
        spec_n = self.get_norm_spectrum(spec)
        if std: 
            std_r = std[composition[r]]
        else:
            std_r = r
        spec_ref_raw = self.get_roi_spectrum_kb(ims_ref[std_r])
        spec_ref = self.moving_average(spec_ref_raw)
        spec_r_n = self.get_norm_spectrum(spec_ref)
        

        axs[0].plot(spec_r_n, 'k.-', label = 'smooth Ref k-\u03B2')
        axs[0].plot(spec_n, 'r.-', label = 'smooth k-\u03B2')
        axs[0].fill_between(np.arange(len(spec)), 0, 
                            spec_n - spec_r_n,
                            color='r', label = 'Integrated diff')

        axs[1].plot(self.get_norm_spectrum(spec_raw_rmbg), 'r.-', label = 'raw k-\u03B2')
        axs[1].plot(self.get_norm_spectrum(spec_ref_raw), 'k.-', label = 'raw Ref_' + std_r + ' k-\u03B2')
 
        f.suptitle('Run #' + r + ', P = ' + str(pressure[r]) + ' GPa' + ', X = ' + composition[r])
        axs[0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        axs[1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        axs[0].legend(); axs[1].legend()
        return f, axs
    
    def kb_series_plot_new(self, r, ims, ims_ref, pressure, composition, std = None, i_l = 50, i_r = 100, i_g = 91):
        '''
        This function will plot the IAD of K_beta and K_beta satellite
        '''
        f, axs = plt.subplots(1,2, figsize=(10,4))

        spec_raw = self.get_roi_spectrum_kb(ims[r])
        spec_raw_rmbg = self.bg_sub_spectrum_kb(ims[r])
        spec = self.moving_average(spec_raw_rmbg)
        spec_n = self.get_norm_spectrum(self.lin_bg_sub(spec))
        if std: 
            std_r = std[composition[r]]
        else:
            std_r = r
        spec_ref_raw = self.get_roi_spectrum_kb(ims_ref[std_r])
        spec_ref = self.moving_average(spec_ref_raw)
        spec_r_n = self.get_norm_spectrum(self.lin_bg_sub(spec_ref))
        
        i_max = spec_n.argmax()
        i_ref_max = spec_r_n.argmax()
        i_dif =  i_ref_max - i_max
        
        if i_dif < 0:
            spec_align = spec_n[-i_dif:]
            spec_r_align = spec_r_n[:i_dif]
        elif i_dif > 0:
            spec_align = spec_n[:-i_dif]
            spec_r_align = spec_r_n[i_dif:]
        else:
            spec_align = spec_n
            spec_r_align = spec_r_n 

        transition_point = None
        for i in range(i_l, i_r):
            if spec_align[i] <= spec_r_align[i] and spec_align[i + 1] > spec_r_align[i + 1]:
                transition_point = i + 1
                break
        if transition_point is None:
            transition_point = i_g
        #x= spec_r_align[:150]
        #xpos = np.abs(x - np.max(spec_r_align)/2).argmin()
        

        axs[0].plot(spec_r_n, 'ko', markersize = 1, label = 'smooth Ref k-\u03B2')
        axs[0].plot(spec_n, 'ro', markersize = 1, label = 'smooth k-\u03B2')
        axs[0].fill_between(np.arange(len(spec)), 0, 
                            spec_n - spec_r_n,
                            color='r', label = 'Integrated diff')

        axs[1].plot(spec_align, 'ro', markersize = 1, label = 'smooth k-\u03B2')
        axs[1].plot(spec_r_align, 'ko', markersize = 1, label = 'smooth Ref k-\u03B2')
        axs[1].fill_between(np.arange(len(spec_align))[:transition_point], 0, spec_align[:transition_point] -spec_r_align[:transition_point],color='r')
        axs[1].vlines(transition_point, 0, spec_r_align[transition_point],color = 'k', linestyle='--', lw=0.5)
 
        f.suptitle('Run #' + r + ', P = ' + str(pressure[r]) + ' GPa' + ', X = ' + composition[r])
        axs[0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        axs[1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        
        axs[0].legend(); axs[1].legend()
        
        return f, axs
    
    def iad_ka(self, r, ims, ims_ref, pressure, composition, std = None):
        '''
        std: standard run number. This variable accepts a number istead of a dict or a list
        This method only calculate iad, and return a list.
        This method won't save files.
        '''
        iad_ka = {}
        spec_raw = self.get_roi_spectrum_ka(ims[r])
        spec_raw_rmbg = self.bg_sub_spectrum_ka(ims[r])
        spec = self.moving_average(spec_raw_rmbg)
        spec_n = self.get_norm_spectrum(spec)
        if std: 
            std_r = std[composition[r]]
        else:
            std_r = r
        spec_ref_raw = self.get_roi_spectrum_ka(ims_ref[std_r])
        spec_ref = self.moving_average(spec_ref_raw)
        spec_r_n = self.get_norm_spectrum(spec_ref)
        
        iad_ka['iad smooth'] = np.abs(spec_n - spec_r_n).sum()
        iad_ka['iad raw'] = np.abs(self.get_norm_spectrum(spec_raw_rmbg) - self.get_norm_spectrum(spec_ref_raw)).sum()


        iad_ka['std run number'] = std_r 

        iad_ka['sample'] = composition[r]
        iad_ka['sample run number'] = r
        iad_ka['peak'] = 'ka'
        iad_ka['pressure'] = pressure[r]
        iad_ka['params'] = {key: self.parm[key] for key in list(self.parm)[:4]}
        return iad_ka
    
    def iad_kb(self, r, ims, ims_ref, pressure, composition, std = None):

        '''
        std: standard run number. This variable accepts a number istead of a dict or a list
        This method only calculate iad, and return a list.
        This method won't save files.
        '''
        iad_kb = {}
        spec_raw = self.get_roi_spectrum_kb(ims[r])
        spec_raw_rmbg = self.bg_sub_spectrum_kb(ims[r])
        spec = self.moving_average(spec_raw_rmbg)
        spec_n = self.get_norm_spectrum(self.lin_bg_sub(spec))
        if std: 
            std_r = std[composition[r]]
        else:
            std_r = r
        spec_ref_raw = self.get_roi_spectrum_kb(ims_ref[std_r])
        spec_ref = self.moving_average(spec_ref_raw)
        spec_r_n = self.get_norm_spectrum(self.lin_bg_sub(spec_ref))
        
        iad_kb['iad smooth']= np.sum(np.abs(spec_n-spec_r_n))
        iad_kb['iad raw'] = np.sum(np.abs(self.get_norm_spectrum(spec_raw)-self.get_norm_spectrum(spec_ref_raw)))
        iad_kb['std run number'] = std_r
        iad_kb['sample']= composition[r]
        iad_kb['sample run number'] = r
        iad_kb['peak'] = 'kb'
        iad_kb['pressure'] = pressure[r]
        iad_kb['params'] = {key: self.parm[key] for key in list(self.parm)[:4]}
        return iad_kb
    
    def iad_kb_st(self, r, ims, ims_ref, pressure, composition, std = None, i_l = 50, i_r = 100, i_g = 91):
        '''
        std: standard run number. This variable accepts a number istead of a dict or a list
        This method only calculate iad, and return a list.
        This method won't save files.
        '''
        iad_kb_st = {}
        spec_raw = self.get_roi_spectrum_kb(ims[r])
        spec_raw_rmbg = self.bg_sub_spectrum_kb(ims[r])
        spec = self.moving_average(spec_raw_rmbg)
        spec_n = self.get_norm_spectrum(self.lin_bg_sub(spec))
        if std: 
            std_r = std[composition[r]]
        else:
            std_r = r
        spec_ref_raw = self.get_roi_spectrum_kb(ims_ref[std_r])
        spec_ref = self.moving_average(spec_ref_raw)
        spec_r_n = self.get_norm_spectrum(self.lin_bg_sub(spec_ref))
        
        i_max = spec_n.argmax()
        i_ref_max = spec_r_n.argmax()
        i_dif =  i_ref_max - i_max
        if i_dif < 0:
            spec_align = spec_n[-i_dif:]
            spec_r_align = spec_r_n[:i_dif]
        elif i_dif > 0:
            spec_align = spec_n[:-i_dif]
            spec_r_align = spec_r_n[i_dif:]
        else:
            spec_align = spec_n
            spec_r_align = spec_r_n 
        #print(r)
        #return len(spec_align), len(spec_r_align)
        
        transition_point = None
        for i in range(i_l, i_r):
            if spec_align[i] <= spec_r_align[i] and spec_align[i + 1] > spec_r_align[i + 1]:
                transition_point = i + 1
                break
        if transition_point is None:
            transition_point = i_g
                
        iad_kb_st['iad smooth']= np.sum(np.abs(spec_align[:transition_point]-spec_r_align[:transition_point]))
        iad_kb_st['std run number'] = std_r
        
        iad_kb_st['sample']= composition[r]
        iad_kb_st['sample run number'] = r
        iad_kb_st['peak'] = 'kb_satellite'
        iad_kb_st['pressure'] = pressure[r]
        return iad_kb_st

    def iad_kb_2022(self, r, ims, ims_ref, pressure, composition, std = None):

        '''
        std: standard run number. This variable accepts a number istead of a dict or a list
        This method only calculate iad, and return a list.
        This method won't save files.
        '''
        iad_kb = {}
        spec_raw = self.get_roi_spectrum(ims[r])
        spec_raw_rmbg = self.bg_sub_spectrum(ims[r])
        spec = self.moving_average(spec_raw_rmbg)
        spec_n = self.get_norm_spectrum(self.lin_bg_sub(spec))
        if std: 
            std_r = std[composition[r]]
        else:
            std_r = r
        spec_ref_raw = self.get_roi_spectrum(ims_ref[std_r])
        spec_ref = self.moving_average(spec_ref_raw)
        spec_r_n = self.get_norm_spectrum(self.lin_bg_sub(spec_ref))

        i_max = spec_n.argmax()
        i_ref_max = spec_r_n.argmax()
        i_dif =  i_ref_max - i_max
        if i_dif < 0:
            spec_align = spec_n[-i_dif:]
            spec_r_align = spec_r_n[:i_dif]
        elif i_dif > 0:
            spec_align = spec_n[:-i_dif]
            spec_r_align = spec_r_n[i_dif:]
        else:
            spec_align = spec_n
            spec_r_align = spec_r_n 
            
        iad_kb['iad smooth']= np.sum(np.abs(spec_n-spec_r_n))
        iad_kb['iad raw'] = np.sum(np.abs(self.get_norm_spectrum(spec_raw)-self.get_norm_spectrum(spec_ref_raw)))
        iad_kb['iad smooth align'] = np.sum(np.abs(spec_align-spec_r_align))
        iad_kb['std run number'] = std_r
        iad_kb['sample']= composition[r]
        iad_kb['sample run number'] = r
        iad_kb['peak'] = 'kb'
        iad_kb['pressure'] = pressure[r]
        iad_kb['params'] = {key: self.parm[key] for key in list(self.parm)[:5]}
        #if r in release_up:
            #iad_kb['release'] = True
        #else:
            #iad_kb['release'] = False
        return iad_kb
    
    def iad_kb_st_2022(self, r, ims, ims_ref, pressure, composition, std = None):
        '''
        std: standard run number. This variable accepts a number istead of a dict or a list
        This method only calculate iad, and return a list.
        This method won't save files.

        For 'iad_xpos', i didn't find the interception of the standard spectra and sample spectra. 
        instead, I estimate the x pos of FWHM of the main peak, and assign it as the right limit of the IAD
        
        xpos[index] = 130

        `ird` refers to the method in Zhu Mao's paper. 
        '''
        iad_kb_st = {}
        spec_raw = self.get_roi_spectrum(ims[r])
        spec_raw_rmbg = self.bg_sub_spectrum(ims[r])
        spec = self.moving_average(spec_raw_rmbg)
        spec_n = self.get_norm_spectrum(self.lin_bg_sub(spec))
        if std: 
            std_r = std[composition[r]]
        else:
            std_r = r
        spec_ref_raw = self.get_roi_spectrum(ims_ref[std_r])
        spec_ref = self.moving_average(spec_ref_raw)
        spec_r_n = self.get_norm_spectrum(self.lin_bg_sub(spec_ref))
        
        i_max = spec_n.argmax()
        i_ref_max = spec_r_n.argmax()
        i_dif =  i_ref_max - i_max
        if i_dif < 0:
            spec_align = spec_n[-i_dif:]
            spec_r_align = spec_r_n[:i_dif]
        elif i_dif > 0:
            spec_align = spec_n[:-i_dif]
            spec_r_align = spec_r_n[i_dif:]
        else:
            spec_align = spec_n
            spec_r_align = spec_r_n 
        #print(r)
        #return len(spec_align), len(spec_r_align)
        
        transition_point = None
        for i in range(50, 100):
            if spec_align[i] <= spec_r_align[i] and spec_align[i + 1] > spec_r_align[i + 1]:
                transition_point = i + 1
                break
        if transition_point is None:
            transition_point = 91
        
        x= spec_r_align[:150]
        xpos = np.abs(x - np.max(spec_r_align)/2).argmin()    
        iad_kb_st['iad smooth']= np.sum(np.abs(spec_align[:transition_point]-spec_r_align[:transition_point]))
        iad_kb_st['iad xpos'] = np.sum(np.abs(spec_align[:xpos]-spec_r_align[:xpos]))
        iad_kb_st['ird xpos'] = np.sum(np.abs((spec_align[:xpos] - spec_r_align[:xpos]) / spec_r_align[:xpos]))
        iad_kb_st['std run number'] = std_r
        
        iad_kb_st['sample']= composition[r]
        iad_kb_st['sample run number'] = r
        iad_kb_st['peak'] = 'kb_satellite'
        iad_kb_st['pressure'] = pressure[r]
        #if r in release_up:
            #iad_kb_st['release'] = True
        #else:
            #iad_kb_st['release'] = False
        return iad_kb_st

    
    def st_plot_2022(self, r, ims, ims_ref, pressure, composition, std = None):
        spec_raw = self.get_roi_spectrum(ims[r])
        spec_raw_rmbg = self.bg_sub_spectrum(ims[r])
        spec = self.moving_average(spec_raw_rmbg)
        spec_n = self.get_norm_spectrum(self.lin_bg_sub(spec))
        if std: 
            std_r = std[composition[r]]
        else:
            std_r = r
        spec_ref_raw = self.get_roi_spectrum(ims_ref[std_r])
        spec_ref = self.moving_average(spec_ref_raw)
        spec_r_n = self.get_norm_spectrum(self.lin_bg_sub(spec_ref))
        
        i_max = spec_n.argmax()
        i_ref_max = spec_r_n.argmax()
        i_dif =  i_ref_max - i_max
        if i_dif < 0:
            spec_align = spec_n[-i_dif:]
            spec_r_align = spec_r_n[:i_dif]
        elif i_dif > 0:
            spec_align = spec_n[:-i_dif]
            spec_r_align = spec_r_n[i_dif:]
        else:
            spec_align = spec_n
            spec_r_align = spec_r_n 

        transition_point = None
        for i in range(50, 100):
            if spec_align[i] <= spec_r_align[i] and spec_align[i + 1] > spec_r_align[i + 1]:
                transition_point = i + 1
                break
        if transition_point is None:
            transition_point = 91
        x= spec_r_align[:150]
        xpos = np.abs(x - np.max(spec_r_align)/2).argmin()
            
        f, ax = plt.subplots(1, 4, figsize = (16,4))

        ax[0].plot(spec_n, 'go', markersize = 0.5)
        ax[0].plot(spec_r_n, 'ko', markersize = 0.5)
        ax[0].fill_between(np.arange(len(spec_r_n)), 0, spec_n -spec_r_n,color='r')
        
        #spec_aligh
        #np.arange(len(spec_aligh)-i_dif, spec_aligh)
        ax[1].plot(spec_align, 'go', markersize = 0.5)
        ax[1].plot(spec_r_align, 'ko', markersize = 0.5)
        ax[1].fill_between(np.arange(len(spec_align)), 0, spec_align -spec_r_align,color='r')
        
        ax[2].plot(spec_align, 'go', markersize = 0.5)
        ax[2].plot(spec_r_align, 'ko', markersize = 0.5)
        ax[2].fill_between(np.arange(len(spec_align))[:transition_point], 0, spec_align[:transition_point] -spec_r_align[:transition_point],color='r')
        ax[2].vlines(transition_point, 0, spec_r_align[transition_point],color = 'k', linestyle='--', lw=0.5)

        ax[3].plot(spec_align, 'go', markersize = 0.5)
        ax[3].plot(spec_r_align, 'ko', markersize = 0.5)
        ax[3].fill_between(np.arange(len(spec_align))[:xpos], 0, spec_align[:xpos] -spec_r_align[:xpos],color='r')
        ax[3].vlines(xpos, 0, spec_r_align[xpos], color = 'k', linestyle='--', lw=0.5)
        f.suptitle('Run #' + r + ', P = ' + 
                   str(pressure[r]) + 'GPa' + ', X = ' + composition[r])
        return f, ax
    