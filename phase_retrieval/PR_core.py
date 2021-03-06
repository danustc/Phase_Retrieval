"""
Created by Dan Xie on 07/15/2016
Last edit: 05/11/2017
Class PSF_PF retrieves a the pupil plane from a given PSF measurement
Need to use @setter and @property functions to simplify this.
"""

# This should be shaped into an independent module 
# Phase retrieval 

import os
import numpy as np
import tifffile as tf
from pupil import Pupil
from numpy.lib.scimath import sqrt as _msqrt
from skimage.restoration import unwrap_phase
from psf_tools import psf_zplane

# a small zernike function

class Core(object):
    def __init__(self, PSF = None):
        self._PSF = PSF
        self.PF = None
        self.dx = None
        self.l= None
        self._NA = None
        self._nfrac = None
        self.cf = None
        self.nw = 1
        self.dw = 0.005
        print("Initialized!")
    # -----------------------Below is a couple of setting functions ---------------
    @property
    def nfrac(self):
        return self._nfrac

    @nfrac.setter
    def nfrac(self,new_nfrac):
        self._nfrac = new_nfrac

    @property
    def NA(self):
        return self._NA

    @NA.setter
    def NA(self, new_NA):
        self._NA = new_NA

    @property
    def lcenter(self):
        return self.l

    @lcenter.setter
    def lcenter(self, new_lcenter):
        # set the central wavelength
        self.l = new_lcenter

    @property
    def pxl(self):
        return self.dx
    @pxl.setter
    def pxl(self, new_pxl):
        self.dx = new_pxl

    @property
    def objf(self):
        return self.cf
    @objf.setter
    def objf(self, new_cf):
        self.cf = new_cf

    @property
    def n_wave(self):
        return self.nw
    @n_wave.setter
    def n_wave(self, new_nw):
        self.nw = new_nw

    @property
    def d_wave(self):
        return self.dw
    @d_wave.setter
    def d_wave(self, new_dw):
        self.dw = new_dw
 
    @property
    def PSF(self):
        return self._PSF
    @PSF.setter
    def PSF(self, new_PSF):
        self._PSF = new_PSF

    def load_psf(self,psf_path):
        '''
        load a psf function
        '''
        ext =  os.path.basename(psf_path).split('.')[-1]
        if ext == 'npy':
            PSF = np.load(psf_path)
        elif ext == 'tif':
            PSF = tf.imread(psf_path)

        try:
            nz, ny, nx = PSF.shape
            print(nz, ny, nx)
            self.PSF = PSF
            self.nx = np.min([ny,nx])
            self.nz = nz
            return True
        except UnboundLocalError:
            print("wrong PSF format. Please reload the psf.")
            return False

    def set_zrange(self):
        z_offset, zz = psf_zplane(self.PSF, self.dz, self.l/3.2) # This should be the reason!!!! >_<
        print( "   z_offset = ", z_offset)
        zs = zz-z_offset
        self.cz = int(-zs[0]//self.dz)
        self.zs = zs
        print("psf loaded!")


    def updateNA(self, new_NA):
        self._NA = new_NA

        self.PF.update(NA = new_NA)


    def pupil_Simulation(self):
        # simulate a pupil function using given parameters; update the list. Everything is included.
        print(self.NA)
        self.PF= Pupil(self.nx, self.dx,self.l,self.nfrac,self.NA,self.cf,wavelengths=self.n_wave, wave_step = self.d_wave) # initialize the pupil function

        in_pupil = self.PF.k <= self.PF.k_max
        self.NK = in_pupil.sum()


    def background_reset(self, mask, psf_diam):
        '''
        reset the background of the PSF
        mask is the outer diameter
        psf_diam is the inner diameter
        '''
        Mx, My = np.meshgrid(np.arange(self.nx)-self.nx/2., np.arange(self.nx)-self.nx/2.)
        r_pxl = _msqrt(Mx**2 + My**2)
        bk_inner = psf_diam
        bk_outer = mask
        hcyl = np.array(self.nz*[np.logical_and(r_pxl>=bk_inner, r_pxl<bk_outer+1)])
        incyl = np.array(self.nz*[r_pxl< bk_outer])
        background = np.mean(self.PSF[hcyl])
        self.PSF[np.logical_not(incyl)] = background

        return background


    def retrievePF(self, p_diam, p_mask, nIt):
        A = self.PF.plane # initial pupil function:plane
        background = self.background_reset(mask = p_mask, psf_diam = p_diam)
        print( "   background = ", background)
        PSF_sample = self.PSF
        complex_PF = self.PF.psf2pf(PSF_sample, self.zs, background, A, nIt)
        print(self.zs)
        Pupil_final = _PupilFunction(complex_PF)
        self.pf_complex = Pupil_final.complex
        self.pf_phase = unwrap_phase(Pupil_final.phase)
        self.pf_ampli = Pupil_final.amplitude


    def get_phase(self, crop = True):
        '''
        return the (unwrapped pupil phase)
        '''
        if crop:
            hx = int(self.nx//2)
            #cropped_phase = self.pf_phase[hx - self.PF.k_pxl-1:hx+self.PF.k_pxl+1, hx-self.PF.k_pxl-1:hx+self.PF.k_pxl+1]
            cropped_phase = self.pf_phase[hx - self.PF.k_pxl:hx+self.PF.k_pxl, hx-self.PF.k_pxl:hx+self.PF.k_pxl]
            return cropped_phase
        else:
            return self.pf_phase


    def get_ampli(self, crop = True):
        if crop:
            hx = int(self.nx//2)
            #cropped_ampli= self.pf_ampli[hx - self.PF.k_pxl-1:hx+self.PF.k_pxl+1, hx-self.PF.k_pxl-1:hx+self.PF.k_pxl+1]
            cropped_ampli= self.pf_ampli[hx - self.PF.k_pxl:hx+self.PF.k_pxl, hx-self.PF.k_pxl:hx+self.PF.k_pxl]
            return cropped_ampli
        else:
            return self.pf_ampli

    def get_config(self):
        # return the configuration of the class.
        conf_dict = {'NA': self.NA, 'nfrac': self.nfrac, 'objf': self.objf/1000, 'wavelength': self.lcenter*1000, 'pxl':self.pxl*1000 , 'nwave':self.n_wave, 'wstep':self.d_wave*1000, 'zstep':self.dz }
        return conf_dict

    def strehl_ratio(self):
        # this is very raw. Should save the indices for pixels inside the pupil. 
        # by definition:
        phase = self.get_phase()
        ampli = self.get_ampli()
        ephase = np.exp(1j*phase)*np.sign(ampli)
        avg_ephase = ephase.sum()/self.NK
        strehl = np.abs(avg_ephase)**2
        # by approximation:
        avg_phase = phase.sum()/self.NK
        var_phase = (np.sign(ampli)*(phase-avg_phase)**2).sum()/self.NK
        strehl_appro = np.exp(-var_phase)

        # count in amplitude effect:
        strehl_ampli = np.abs((ampli*ephase).sum()/ampli.sum())**2

        return strehl, strehl_appro, strehl_ampli


    def shutDown(self):
        '''
        what should I fill here?
        '''
        pass

class _PupilFunction(object):
    '''
    a pupil function that keeps track when either complex or amplitude/phase
    representation is changed.
    '''
    def __init__(self, cmplx):
        self.complex = cmplx

    @property
    def complex(self):
        return self._complex

    @complex.setter
    def complex(self, new):
        self._complex = new
        self._amplitude = abs(new)
        self._phase = np.angle(new)

    @property
    def amplitude(self):
        return self._amplitude

    @amplitude.setter
    def amplitude(self, new):
        self._amplitude = new
        self._complex = new * np.exp(1j*self._phase)

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, new):
        self._phase = new
        self._complex = self._amplitude * np.exp(1j*new)

