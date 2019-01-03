'''
This is the ui interface for phase retrieval.
Created by Dan on 05/04/2017 (Happy Youth's Day!)
'''

from PyQt5 import QtWidgets, QtCore
import sys
import os
import numpy as np
from libtim import zern
from PR_core import Core
import PR_design

class UI(object):
    '''
    Update log and brief instructions.
    '''
    def __init__(self, core):
        '''
        initialize the UI.
        core: the core functions which the UI calls
        design_path: the UI design.
        '''
        self._core= core
        self._app = QtWidgets.QApplication(sys.argv)
        self._window = QtWidgets.QMainWindow()
        self._window.closeEvent = self.shutDown

        self._ui = PR_design.Ui_Form()
        self._ui.setupUi(self._window)

        #self._app = QtWidgets.QWidget()

        # The connection group of the buttons and texts
        self._ui.pushButton_retrieve.clicked.connect(self.retrievePF)
        self._ui.pushButton_loadpsf.clicked.connect(self.load_PSF)
        self._ui.pushButton_ampli.clicked.connect(self.display_ampli)
        self._ui.pushButton_pffit.clicked.connect(self.fit_zernike)
        self._ui.pushButton_savepupil.clicked.connect(self.savePupil)
        self._ui.lineEdit_NA.returnPressed.connect(self.set_NA)
        self._ui.lineEdit_nfrac.returnPressed.connect(self.set_nfrac)
        self._ui.lineEdit_zstep.returnPressed.connect(self.set_dz)
        self._ui.lineEdit_wlc.returnPressed.connect(self.set_wavelength)

        #cs = self._ui.mpl_pupil.figure.axes[0].matshow(self._core.pf_ampli)
        # initialize some parameters
        self.set_wavelength()
        self.set_NA()
        self.set_wstep()
        self.set_nfrac()
        self.set_pxl()
        self.set_objf()
        self.set_nwave()


        self._window.show()
        self._app.exec_()

    def load_PSF(self):
        '''
        load a psf function (.npy) from the selected folder
        '''
        self.set_dz()
        filename = QtWidgets.QFileDialog.getOpenFileName(None, 'Open psf:', '', '*.*')[0]
        print("Filename:", filename)
        self._ui.lineEdit_loadpsf.setText(filename)
        self.file_path = os.path.dirname(filename)
        self._core.load_psf(filename)
        self._core.set_zrange(self.dz)
        self._core.pupil_Simulation(self.nwave,self.wstep)
        self.display_psf(n_cut = self._core.cz )


    def retrievePF(self):
        # retrieve PF from psf
        print("function connected!")
        self._core.set_zrange(self.dz)
        mask_size = int(self._ui.lineEdit_mask.text())
        psf_rad = int(self._ui.lineEdit_prad.text())
        nIt = self._ui.spinBox_nIt.value()
        self._core.retrievePF(psf_rad, mask_size, nIt)
        self.display_psf(n_cut = self._core.cz)
        self.display_pupil()

    # ------Below are a couple of setting functions ------
    def set_nwave(self):
        self.nwave = int(self._ui.lineEdit_nwl.text())

    def set_objf(self, obj_f = None):
        # set objective focal length 
        if obj_f is None:
            obj_f = float(self._ui.lineEdit_objfl.text())
        self.obj_f = obj_f
        self._core.objf = obj_f*1000

    def set_pxl(self, pxl_size = None):
        if pxl_size is None:
            pxl_size = float(self._ui.lineEdit_pxl.text())
        self.pxl_size = pxl_size
        self._core.pxl = pxl_size*0.001

    def set_NA(self, NA_input = None):
        if NA_input is None:
            NA_input = float(self._ui.lineEdit_NA.text())
        self.NA = NA_input
        self._core.NA = NA_input

    def set_nfrac(self, nfrac = None):
        if nfrac is None:
            nfrac = float(self._ui.lineEdit_nfrac.text())
        self._core.nfrac = nfrac

    def set_dz(self, dz_input = None):
        if dz_input is None:
           dz_input = float(self._ui.lineEdit_zstep.text())
        self.dz = dz_input # this is not stored in the core program.

    def set_wavelength(self,wavelength = None):
        if wavelength is None:
            wavelength = float(self._ui.lineEdit_wlc.text())
        self.wavelength = wavelength
        self._core.lcenter = wavelength*0.001


    def set_wstep(self, wstep = None):
        if wstep is None:
            wstep = float(self._ui.lineEdit_wlstep.text())
        self.wstep = wstep*0.001


    # ------Below are a couple of execution and displaying functions ------------
    def fit_zernike(self):
        '''
        fit the retrieved pupil to zernike.
        '''
        self.nmodes = self._ui.spinBox_nmode.value()
        k_max = self._core.PF.k_pxl
        phase = self._core.pf_phase
        D_phase = int(phase.shape[0]//2)
        pf_crop = phase[D_phase - k_max:D_phase+k_max, D_phase-k_max:D_phase + k_max]
        z_fit = zern.fit_zernike(pf_crop, rad = k_max, nmodes = self.nmodes)[0]/(2*np.pi)
        print(z_fit)



    def display_psf(self, n_cut, dimension = 0, log_scale = True):
        '''
        display the psf across a plane.
        dimension = 0: lateral, 1: xz plane, 2: yz plane
        '''
        if dimension == 0:
            psf_slice = self._core.PSF[n_cut]
        elif dimension ==1:
            psf_slice = self._core.PSF[:, n_cut, :]
        else:
            psf_slice = self._core.PSF[:,:, n_cut]
        self._ui.mpl_psf.figure.axes[0].matshow(np.log(psf_slice), cmap = 'Greys_r')
        self._ui.mpl_psf.figure.axes[0].set_axis_off()
        self._ui.mpl_psf.draw()


    def display_ampli(self):
        '''
        display the amplitude of the pupil.
        '''
        cs = self._ui.mpl_pupil.figure.axes[0].matshow(self._core.pf_ampli)
        if len(self._ui.mpl_pupil.figure.axes) ==1:
            self._ui.mpl_pupil.figure.colorbar(cs, orientation = 'vertical', pad = 0.05)
        else:
            cb = self._ui.mpl_pupil.figure.axes[1]
            cb.cla()
            self._ui.mpl_pupil.figure.colorbar(cs, cax = cb)
        self._ui.mpl_pupil.figure.axes[0].set_axis_off()
        self._ui.mpl_pupil.draw()

    def display_pupil(self):
        '''
        display the pupil function.
        '''
        cs = self._ui.mpl_pupil.figure.axes[0].matshow(self._core.pf_phase)
        if len(self._ui.mpl_pupil.figure.axes) ==1:
            self._ui.mpl_pupil.figure.colorbar(cs, orientation = 'vertical', pad = 0.05)
        else:
            cb = self._ui.mpl_pupil.figure.axes[1]
            self._ui.mpl_pupil.figure.colorbar(cs, cax = cb)

        self._ui.mpl_pupil.figure.axes[0].set_axis_off()
        self._ui.mpl_pupil.draw()


    def savePupil(self):
        '''
        save the pupil function
        '''
        psf_export = np.stack((self._core.pf_phase, self._core.pf_ampli))
        basename = self._ui.lineEdit_pupilfname.text()
        full_name = self.file_path + '/' + basename
        print("save to destination:", full_name)
        np.save(full_name, psf_export)



    def shutDown(self, event):
        '''
        shut down the UI
        '''
        self._core.shutDown()
        self._app.quit()

# ------------------------Test of the module------------
def main():
    pr_core = Core()
    PR_UI = UI(pr_core)

if __name__ == '__main__':
    main()
