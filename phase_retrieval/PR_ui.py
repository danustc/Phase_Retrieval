'''
This is the ui interface for phase retrieval.
Created by Dan on 05/04/2017 (Happy Youth's Day!)
'''

from PyQt5 import QtWidgets
import sys
import os
import numpy as np
import yaml
from libtim import zern
from PR_core import Core
import PR_design
from functools import partial

class UI(object):
    '''
    Update log and brief instructions.
    '''
    def __init__(self, core, conf_path = None):
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

        # The connection group of the buttons and texts
        self._ui.pushButton_loadconf.clicked.connect(partial(self.loadConf, None))
        self._ui.pushButton_exconf.clicked.connect(self.exportConf)
        self._ui.pushButton_retrieve.clicked.connect(self.retrievePF)
        self._ui.pushButton_loadpsf.clicked.connect(self.load_PSF)
        self._ui.pushButton_pffit.clicked.connect(self.fit_zernike)
        self._ui.pushButton_savepupil.clicked.connect(self.savePupil)
        self._ui.pushButton_savefit.clicked.connect(self.saveFit)
        self._ui.pushButton_rm4.clicked.connect(self.rm4)

        self._ui.lineEdit_NA.returnPressed.connect(self.set_NA)
        self._ui.lineEdit_nfrac.returnPressed.connect(self.set_nfrac)
        self._ui.lineEdit_zstep.returnPressed.connect(self.set_dz)
        self._ui.lineEdit_wlc.returnPressed.connect(self.set_wavelength)
        self._ui.lineEdit_wlstep.returnPressed.connect(self.set_wstep)
        self._ui.lineEdit_nwl.returnPressed.connect(self.set_nwave)
        self._ui.lineEdit_pxl.returnPressed.connect(self.set_pxl)
        self._ui.lineEdit_objfl.returnPressed.connect(self.set_objf)
        self._ui.checkBox_crop.stateChanged.connect(self.set_crop)
        self.ax_fit = self._ui.mpl_zernike.figure.axes[0]

        # initialize some parameters
        self.has_PSF = False
        self.set_crop()
        if conf_path is None:
            self.conf_folder = '' # set to current folder
            self.configuration()
        else:
            self.conf_folder = os.path.dirname(os.path.abspath(conf_path)) + '/'
            self.loadConf(conf_path)



        self._window.show()
        self._app.exec_()

    def configuration(self, conf_dict = None):
        if conf_dict is None:
            self.set_wavelength()
            self.set_nwave()
            self.set_wstep()
            #self._core.pupil_Simulation(self.nwave,self.wstep)
            self.set_dz()
            self.set_NA()
            self.set_nfrac()
            self.set_pxl()
            self.set_objf()
            self.z_fit = None
            self.cleaned_phase = None
        else:
            self.set_NA(conf_dict['NA'])
            self.set_nfrac(conf_dict['nfrac'])
            self.set_nwave(conf_dict['nwave'])
            self.set_objf(conf_dict['objf'])
            self.set_pxl(conf_dict['pxl'])
            self.set_dz(conf_dict['zstep'])
            self.set_wavelength(conf_dict['wavelength'])
            self.set_wstep(conf_dict['wstep'])

    def loadConf(self, conf_source = None):
        '''
        load configuration
        '''
        if conf_source is None:
            conf_source = QtWidgets.QFileDialog.getOpenFileName(None, 'Open Configuration file:', '', "configuration files(*.txt *.yaml)")[0]
        print("Configuration:", conf_source)
        self._ui.lineEdit_conf.setText(conf_source)
        with open(conf_source, 'r') as fi:
            conf_dict = yaml.load(fi)
            self.configuration(conf_dict)

    def exportConf(self):
        '''
        export configuration
        '''
        # Read the file name from current 
        conf_dest = self._ui.lineEdit_conf.text()
        if conf_dest == '':
            exp_destination = QtWidgets.QFileDialog.getOpenFileName(None, 'Open psf:', '', '*.*')[0]
        else:
            print(conf_dest)
            exp_destination = self.conf_folder + conf_dest
        conf_dict = self._core.get_config()
        with open(exp_destination, 'w') as fo:
            yaml.dump(conf_dict, fo)



    def load_PSF(self):
        '''
        load a psf function (.npy) from the selected folder
        '''
        filename = QtWidgets.QFileDialog.getOpenFileName(None, 'Open psf:', '', '*.*')[0]
        print("Filename:", filename)
        self._ui.lineEdit_loadpsf.setText(filename)
        self.file_path = os.path.dirname(filename)
        if self._core.load_psf(filename):
            self.display_psf(n_cut = int(self._core.nz//2))
            self.has_PSF = True


    def retrievePF(self):
        # retrieve PF from psf
        if self.has_PSF:
            self._core.set_zrange()
            self._core.pupil_Simulation()
            mask_size = int(self._ui.lineEdit_mask.text())
            psf_rad = int(self._ui.lineEdit_prad.text())
            nIt = self._ui.spinBox_nIt.value()
            #nIt = 1
            self._core.retrievePF(psf_rad, mask_size, nIt)
            raw_phase= self._core.get_phase(crop = True)
            ampli=self._core.get_ampli(crop = True)
            #strehl_raw = self.strehl_ratio(raw_phase, ampli)
            #print("Raw strehl ratio:", strehl_raw)
            #self.display_psf(n_cut = int(self._core.nz//2))
            self.display_phase()
            self.display_ampli()
            print("Strehl ratio:", self._core.strehl_ratio())
        else:
            print("There is no PSF for phase retrieval.")


    # ------Below are a couple of setting functions ------
    def set_crop(self):
        self.crop = self._ui.checkBox_crop.isChecked()

    def set_nwave(self, nwave = None):
        '''
        directly update the core parameters
        '''
        if nwave is None:
            nwave = int(self._ui.lineEdit_nwl.text())
        else:
            self._ui.lineEdit_nwl.setText(str(nwave))
        self._core.n_wave = nwave

    def set_objf(self, obj_f = None):
        '''
        # set objective focal length, update the core value only.
        '''
        if obj_f is None:
            obj_f = float(self._ui.lineEdit_objfl.text())
        else:
            self._ui.lineEdit_objfl.setText(str(obj_f))
        self._core.objf = obj_f*1000

    def set_pxl(self, pxl_size = None):
        '''
        update the core value
        '''
        if pxl_size is None:
            pxl_size = float(self._ui.lineEdit_pxl.text())
        else:
            self._ui.lineEdit_pxl.setText(str(pxl_size))
        self._core.dx = pxl_size*0.001

    def set_NA(self, NA_input = None):
        if NA_input is None:
            NA_input = float(self._ui.lineEdit_NA.text())
        else:
            self._ui.lineEdit_NA.setText(str(NA_input))
        self._core.NA = NA_input

    def set_nfrac(self, nfrac = None):
        if nfrac is None:
            nfrac = float(self._ui.lineEdit_nfrac.text())
        else:
            self._ui.lineEdit_nfrac.setText(str(nfrac))

        self._core.nfrac = nfrac

    def set_dz(self, dz_input = None):
        if dz_input is None:
           dz_input = float(self._ui.lineEdit_zstep.text())
        else:
           self._ui.lineEdit_zstep.setText(str(dz_input))
        self._core.dz = dz_input

    def set_wavelength(self,wavelength = None):
        if wavelength is None:
            wavelength = float(self._ui.lineEdit_wlc.text())
        else:
            self._ui.lineEdit_wlc.setText(str(wavelength))
        self._core.lcenter = wavelength*0.001 # convert to microns


    def set_wstep(self, wstep = None):
        if wstep is None:
            wstep = float(self._ui.lineEdit_wlstep.text())
        else:
            self._ui.lineEdit_wlstep.setText(str(wstep))
        self._core.d_wave  = wstep*0.001 # convert to microns



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
        z_fit = zern.fit_zernike(pf_crop, rad = k_max, nmodes = self.nmodes)[0]
        print(z_fit)

        self.z_fit = z_fit
        self.display_fit(rm4 = False)

    def rm4(self):
        '''
        remove 1-4 modes of zernike
        '''
        if self.z_fit is None:
            print("The pupil function has not been fit to Zernike modes. Please fit first!")
        else:
            self.z_fit[:4] = 0.
            k_max = self._core.PF.k_pxl
            print("k_max:", k_max)
            cleaned_phase = zern.calc_zernike(self.z_fit, rad = k_max)
            print(cleaned_phase.shape)
            ampli=self._core.get_ampli(True)
            print(ampli.shape)
            strehl_clean = self.strehl_ratio(cleaned_phase, ampli)
            print("removed the first 4 modes. Strehl ratio:", strehl_clean)
            self.display_phase(cleaned_phase)
            self.display_fit(rm4 = True)
            self.cleaned_phase = cleaned_phase

    def display_pupil(self):
        self.display_ampli()
        self.display_phase()


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
        cs = self._ui.mpl_ampli.figure.axes[0].matshow(self._core.get_ampli(self.crop))
        if len(self._ui.mpl_ampli.figure.axes) ==1:
            self._ui.mpl_ampli.figure.colorbar(cs, orientation = 'vertical', pad = 0.05)
        else:
            cb = self._ui.mpl_ampli.figure.axes[1]
            cb.cla()
            self._ui.mpl_ampli.figure.colorbar(cs, cax = cb)
        self._ui.mpl_ampli.figure.axes[0].set_axis_off()
        self._ui.mpl_ampli.draw()
        #self._ui.pushButton_ampli.setText(QtCore.QCoreApplication.translate("Form", "Phase"))

    def display_phase(self, phase = None):
        '''
        display the pupil function.
        '''
        if phase is None:
            phase = self._core.get_phase(self.crop)
        cs = self._ui.mpl_phase.figure.axes[0].matshow(phase)
        if len(self._ui.mpl_phase.figure.axes) ==1:
            self._ui.mpl_phase.figure.colorbar(cs, orientation = 'vertical', pad = 0.05)
        else:
            cb = self._ui.mpl_phase.figure.axes[1]
            self._ui.mpl_phase.figure.colorbar(cs, cax = cb)

        self._ui.mpl_phase.figure.axes[0].set_axis_off()
        self._ui.mpl_phase.draw()



    def display_fit(self, rm4):
        '''
        display zernike modes.
        '''
        if rm4:
            nmodes = np.arange(5,self.nmodes+1)
            zfit = self.z_fit[4:]
        else:
            nmodes = np.arange(self.nmodes)+1
            zfit = self.z_fit

        print(nmodes, zfit)
        self.ax_fit.cla()
        self.ax_fit.bar(nmodes, zfit)
        self._ui.mpl_zernike.draw()

    def savePupil(self):
        '''
        save the pupil function
        '''
        try:
            psf_export = np.stack((self._core.pf_phase, self._core.pf_ampli))
            basename = self._ui.lineEdit_pupilfname.text()
            full_name = self.file_path + '/' + basename
            print("Save to the destination:", full_name)
            np.save(full_name, psf_export)
        except AttributeError:
            print('There is no retrieved pupil function.')

    def strehl_ratio(self, phase, ampli = None):
        e_phase = np.abs(np.sign(phase))*(np.cos(phase) + 1j * np.sin(phase))
        NK = self._core.NK
        if ampli is None:
            pass
        #strehl = np.abs(e_phase.sum()/NK)**2
        #c_up = np.abs(self.pf_complex.sum())**2
        #c_down = (self.pf_ampli**2).sum()*self.NK
        #strehl = c_up/c_down
        else:
            pf_complex = ampli*e_phase
            c_up = pf_complex.sum()
            c_down = ampli.sum()
            strehl = np.abs(c_up/c_down)**2
        # Phase-only calculation
        return strehl


    def saveFit(self):
        try:
            basename = self._ui.lineEdit_pupilfname.text()
            full_name = self.file_path + '/' + basename + '_zfit'
            np.save(full_name, self.z_fit)
        except AttributeError:
            print("There is no zernike coefficients.")


    def shutDown(self, event):
        '''
        shut down the UI
        '''
        self._core.shutDown()
        self._app.quit()

# ------------------------Test of the module------------
def main():
    pr_core = Core()
    if len(sys.argv) == 1:
        PR_UI = UI(pr_core)
    elif len(sys.argv) > 1:
        # configuration file is received
        print('configuration path:', sys.argv[1])
        PR_UI = UI(pr_core, conf_path = sys.argv[1])

if __name__ == '__main__':
    main()
