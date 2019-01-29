#!/usr/bin/python
# based on Ryan's inControl package with some minor modification.
# This should be wrapped into a cleaner class.
# Contains a couple of redundant functions which are device-based. Should be removed.
# Last update: 01/16/2019 by Dan

import numpy as np
from scipy import fftpack as _fftpack
from scipy import ndimage
import pyfftw


class Pupil(object):

    '''
    Simulates the behaviour of a microscope based on Fourier optics.

    Parameters
    ----------
    nx: int
        The side length of the pupil function or microscope image in pixels.
    dx: float
        The pixel size in the image plane. unit:
    l: float
        Light wavelength in micrometer.
    n: float
        The refractive index of immersion and sample medium. Mismatching media
        are currently not supported.
    NA: float
        The numerical aperture of the microscope objective.
    f: float
        The objective focal length in micrometer.
    '''

    def __init__(self, nx, dx, l = 0.520, n=1.33, NA=1.27, f=3333.33, wavelengths=10, wave_step=0.005):

        dx = float(dx)
        self.dx = dx
        l = float(l)
        n = float(n)
        NA = float(NA)
        f = float(f)
        self.nx = nx # number of pixels
        self.ny = nx

        self.l = float(l) # wavelength
        self.n = float(n) # refractive index
        self.f = float(f)
        self.NA = NA

        self.dk = 1./(nx*dx)
        self.numWavelengths = wavelengths
        self.d_wl = wave_step
        self.phase_construction()

        # Pupil function pixel grid:
        # Axial Fourier space coordinate. This must be updated when n, l, k are updated.

        self.r = self.k/self.k_max # Should be dimension-less


        # Plane wave:


    def update(self, NA = None, wl = None, focal = None):
        if NA is not None:
            self.NA = NA
        if wl is not None:
            self.l = wl
        if focal is not None:
            self.f = focal


    def phase_construction(self):
        '''
        k-space sampling
        '''
        self.s_max = self.f*self.NA # The maximum radius of pupil function, but it appears no where 
        self.k_max = self.NA/self.l # The radius of the pupil in the k space 
        self.k_pxl = int(self.k_max/self.dk)
        print("The pixel radius of pupil:", self.k_pxl)

        nx = self.nx
        Mx,My = np.mgrid[-nx/2.:nx/2.,-nx/2.:nx/2.]+0.5
        kx = self.dk*Mx
        ky = self.dk*My
        self.k = np.sqrt(kx**2+ky**2) # This is in the unit of 1/x # this is a 2-D array 
        self.plane = np.ones((nx,nx))+1j*np.zeros((nx,nx))
        out_pupil = self.k>self.k_max
        self.k[out_pupil] = 0
        self.plane[out_pupil] = 0 # Outside the pupil: set to zero

        self.kz = np.real(np.sqrt((self.n/self.l)**2-self.k**2))
        self.kz[out_pupil] = 0


        self.kzs = np.zeros((self.numWavelengths,nx,nx))
        ls = np.linspace(self.l-self.d_wl,self.l+self.d_wl,self.numWavelengths)
        print("wavelength:", ls)
        for i in range(0,self.kzs.shape[0]):
            self.kzs[i] = np.real(np.sqrt((self.n/ls[i])**2-self.k**2))
            self.kzs[i,out_pupil] = 0
        # Scaled pupil function radial coordinate:
        print("k_space parameters:", self.k_max, self.k.max(), self.kz.max(), self.kzs.max())

        self.out = out_pupil

    def pf2psf(self, PF, zs, intensity=True, verbose=False, use_pyfftw=False):
        """
        Computes the point spread function for a given pupil function.

        Parameters
        ----------
        PF: array
            The complex pupil function.
        zs: number or iteratable
            The axial position or a list of axial positions which should be computed. Focus is at z=0.
        intensity: bool
            Specifies if the intensity or the complex field should be returned.

        Returns
        -------
        PSF: array or memmap
            The complex PSF. If the memory is to small, a memmap will be
            returned instead of an array.
        """
        ny, nx = self.ny, self.nx
        if np.isscalar(zs):
            zs = [zs]
        print("The z positions:", zs)
        nz = len(zs)
        kz = self.kz

    # The normalization for ifft2:
        N = np.sqrt(self.nx*self.ny)

        # Preallocating memory for PSF:
        if intensity:
            PSF = np.zeros((nz,nx,nx))
        else:
            PSF = np.zeros((nz,nx,nx))+1j*np.zeros((nz,nx,nx))
        for i in range(nz):
            if verbose: print('Calculating PSF slice for z={0}um.'.format(zs[i]))
            U = np.zeros((nx,nx)) + 1j*np.zeros((nx, nx))
            for j in range(0,self.kzs.shape[0]):
                if use_pyfftw:
                     aligned = pyfftw.n_byte_align(_fftpack.ifftshift(np.exp(2*np.pi*1j*self.kzs[j]*zs[i])*PF,16))
                     U = U + N*pyfftw.interfaces.numpy_fft.ifft2(aligned)
                else:
                     U = U + N*_fftpack.ifft2(np.exp(2*np.pi*1j*self.kzs[j]*zs[i])*PF)
            U = U/self.numWavelengths
            _slice_ = _fftpack.ifftshift(U) # move the x0 to the center
            if intensity:
                _slice_ = np.abs(_slice_)**2
            PSF[i] = _slice_
        if nz == 1:
            PSF = PSF[0]
        return PSF




    def psf2pf(self, PSF, zs, mu, A, nIterations=3, use_pyfftw=False, resetAmp=False,
               symmeterize=False):

        '''
        Retrieves the complex pupil function from an intensity-only
        PSF stack by relative entropy minimization. The algorithm is
        based on Kner et al., 2010, doi:10.1117/12.840943, which in turn
        is based on Deming, 2007, J Opt Soc Am A, Vol 24, No 11, p.3666.

        Parameters
        ---------
        PSF: 3D numpy.array
            An intensity PSF stack. PSF.shape has to be
            (nz, psf_tools.nx, psf_tools.nx), where nz is the arbitrary
            number of z slices.
        dz: float
            The distance between two PSF slices.
        mu: float
            The noise level of the PSF.
        A: 2D numpy.array
            The initial guess for the complex pupil function with shape
            (psf_tools.nx, psf_tools.nx).
        '''
        # z spacing:
        nz = PSF.shape[0]
        PSA = np.sqrt(PSF) # The amplitude part of psf
        # Noise level:
        mu = float(mu)
        kz = self.kz
        k = self.k # The lateral self.k
        k_max = self.k_max

        # Z position of slices:
        N = np.sqrt(self.nx*self.ny)

        if use_pyfftw:
            pyfftw.interfaces.cache.enable()

        for ii in range(nIterations):
            # Withing the iteration, A should be masked 
            print( 'Iteration',ii+1)
            U = self.pf2psf(A, zs, intensity=False)
            # Calculated PSF intensity with noise:
            Uconj = np.conj(U)
            #weave.blitz(expr2)
            Ic =  mu + U * Uconj # should I have sqrt here instead of 
            artifact_amp = np.sqrt(Ic)
            print("min", np.min(PSF))
            minFunc = np.mean(PSF*np.log(PSF/Ic))
            print( 'Relative entropy per pixel:', minFunc)
            redChiSq = np.mean((PSF-Ic)**2)
            print( 'Reduced Chi square:', redChiSq)

            # Comparing measured with calculated PSF by entropy minimization:
            Ue = (PSF/Ic)*U # All are 3 d arrays
            #Ue = PSA*U/artifact_amp
            #weave.blitz(expr1)
            # New PF guess:
            A = np.zeros_like(Ue) + 1j*np.zeros_like(Ue) # temporarily set A as a 3-D array 
            for i in range(len(zs)):
                if use_pyfftw:
                    Ue_aligned = pyfftw.n_byte_align(Ue[i],16)
                    fted_ue = _fftpack.fftshift(pyfftw.interfaces.numpy_fft.fft2(Ue_aligned))
                else:
                    fted_ue = _fftpack.fft2(_fftpack.fftshift(Ue[i])) # Transform in x-y plane
                    #fted_ue = _fftpack.fft2(Ue[i]) # Transform in x-y plane

                for j in range(0,self.kzs.shape[0]): # what does this mean? # A correction for multi-wavelength
                    A[i] = A[i] + fted_ue*np.exp(-2*np.pi*1j*self.kzs[j]*zs[i])/N
                A[i] = A[i]/self.numWavelengths

            # OK, Now we are done with all the z-planes.
            A = np.mean(A,axis=0) # Convert A from 3D to 2D; 
            #mean(abs(A))*np.exp(1j*np.angle(A))
            A[self.out] = 0 # set everything outside k_max as 0 
            if resetAmp:
                amp = ndimage.gaussian_filter(np.abs(A),5)
                A = amp*np.nan_to_num(A/np.abs(A))
                A[self.out] = 0 # set everything out of pupil as zero

            if symmeterize:
                if ii>(nIterations/2):
                    A = 0.5*(A+np.flipud(A)) # This is to symmetrize across z-direction
                #counts = sum(abs(A))/self.pupilnpxl
                #A = counts*np.exp(1j*angle(A))
        return A


