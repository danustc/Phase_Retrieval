# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'PR_design.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1024, 700)
        self.centralwidget = QtWidgets.QWidget(Form)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(30, 90, 160, 551))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout_syspara = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_syspara.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_syspara.setObjectName("verticalLayout_syspara")
        self.label_NA = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_NA.setObjectName("label_NA")
        self.verticalLayout_syspara.addWidget(self.label_NA)
        self.lineEdit_NA = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit_NA.setObjectName("lineEdit_NA")
        self.verticalLayout_syspara.addWidget(self.lineEdit_NA)
        self.label_nfrac = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_nfrac.setObjectName("label_nfrac")
        self.verticalLayout_syspara.addWidget(self.label_nfrac)
        self.lineEdit_nfrac = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit_nfrac.setObjectName("lineEdit_nfrac")
        self.verticalLayout_syspara.addWidget(self.lineEdit_nfrac)
        self.label_objfl = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_objfl.setObjectName("label_objfl")
        self.verticalLayout_syspara.addWidget(self.label_objfl)
        self.lineEdit_objfl = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit_objfl.setObjectName("lineEdit_objfl")
        self.verticalLayout_syspara.addWidget(self.lineEdit_objfl)
        self.label_wlc = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_wlc.setObjectName("label_wlc")
        self.verticalLayout_syspara.addWidget(self.label_wlc)
        self.lineEdit_wlc = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit_wlc.setObjectName("lineEdit_wlc")
        self.verticalLayout_syspara.addWidget(self.lineEdit_wlc)
        self.label_nwl = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_nwl.setObjectName("label_nwl")
        self.verticalLayout_syspara.addWidget(self.label_nwl)
        self.lineEdit_nwl = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit_nwl.setObjectName("lineEdit_nwl")
        self.verticalLayout_syspara.addWidget(self.lineEdit_nwl)
        self.label_wlstep = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_wlstep.setObjectName("label_wlstep")
        self.verticalLayout_syspara.addWidget(self.label_wlstep)
        self.lineEdit_wlstep = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit_wlstep.setObjectName("lineEdit_wlstep")
        self.verticalLayout_syspara.addWidget(self.lineEdit_wlstep)
        self.label_pxl = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_pxl.setObjectName("label_pxl")
        self.verticalLayout_syspara.addWidget(self.label_pxl)
        self.lineEdit_pxl = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit_pxl.setObjectName("lineEdit_pxl")
        self.verticalLayout_syspara.addWidget(self.lineEdit_pxl)
        self.label_dz = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_dz.setObjectName("label_dz")
        self.verticalLayout_syspara.addWidget(self.label_dz)
        self.lineEdit_zstep = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit_zstep.setObjectName("lineEdit_zstep")
        self.verticalLayout_syspara.addWidget(self.lineEdit_zstep)
        self.label_mask = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_mask.setObjectName("label_mask")
        self.verticalLayout_syspara.addWidget(self.label_mask)
        self.lineEdit_mask = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit_mask.setObjectName("lineEdit_mask")
        self.verticalLayout_syspara.addWidget(self.lineEdit_mask)
        self.label_nIt = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_nIt.setObjectName("label_nIt")
        self.verticalLayout_syspara.addWidget(self.label_nIt)
        self.spinBox_nIt = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.spinBox_nIt.setMinimum(3)
        self.spinBox_nIt.setProperty("value", 5)
        self.spinBox_nIt.setObjectName("spinBox_nIt")
        self.verticalLayout_syspara.addWidget(self.spinBox_nIt)
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(200, 90, 750, 421))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout_display = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout_display.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_display.setObjectName("gridLayout_display")
        self.label_psfview = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_psfview.setObjectName("label_psfview")
        self.gridLayout_display.addWidget(self.label_psfview, 0, 0, 1, 1)
        self.label_pupilview = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_pupilview.setObjectName("label_pupilview")
        self.gridLayout_display.addWidget(self.label_pupilview, 0, 1, 1, 1)
        self.mpl_pupil = MatplotlibWidget(self.gridLayoutWidget)
        self.mpl_pupil.setObjectName("mpl_pupil")
        self.gridLayout_display.addWidget(self.mpl_pupil, 1, 1, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(3)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_yzview = QtWidgets.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_yzview.sizePolicy().hasHeightForWidth())
        self.pushButton_yzview.setSizePolicy(sizePolicy)
        self.pushButton_yzview.setObjectName("pushButton_yzview")
        self.horizontalLayout.addWidget(self.pushButton_yzview)
        self.pushButton_xyview = QtWidgets.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_xyview.sizePolicy().hasHeightForWidth())
        self.pushButton_xyview.setSizePolicy(sizePolicy)
        self.pushButton_xyview.setObjectName("pushButton_xyview")
        self.horizontalLayout.addWidget(self.pushButton_xyview)
        self.pushButton_xzview = QtWidgets.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_xzview.sizePolicy().hasHeightForWidth())
        self.pushButton_xzview.setSizePolicy(sizePolicy)
        self.pushButton_xzview.setObjectName("pushButton_xzview")
        self.horizontalLayout.addWidget(self.pushButton_xzview)
        self.pushButton_lineview = QtWidgets.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_lineview.sizePolicy().hasHeightForWidth())
        self.pushButton_lineview.setSizePolicy(sizePolicy)
        self.pushButton_lineview.setObjectName("pushButton_lineview")
        self.horizontalLayout.addWidget(self.pushButton_lineview)
        self.gridLayout_display.addLayout(self.horizontalLayout, 2, 0, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setSpacing(3)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.checkBox_rm4 = QtWidgets.QCheckBox(self.gridLayoutWidget)
        self.checkBox_rm4.setObjectName("checkBox_rm4")
        self.horizontalLayout_3.addWidget(self.checkBox_rm4)
        self.pushButton_ampli = QtWidgets.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_ampli.sizePolicy().hasHeightForWidth())
        self.pushButton_ampli.setSizePolicy(sizePolicy)
        self.pushButton_ampli.setObjectName("pushButton_ampli")
        self.horizontalLayout_3.addWidget(self.pushButton_ampli)
        self.spinBox_nmode = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.spinBox_nmode.setMinimum(5)
        self.spinBox_nmode.setProperty("value", 12)
        self.spinBox_nmode.setObjectName("spinBox_nmode")
        self.horizontalLayout_3.addWidget(self.spinBox_nmode)
        self.pushButton_pffit = QtWidgets.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_pffit.sizePolicy().hasHeightForWidth())
        self.pushButton_pffit.setSizePolicy(sizePolicy)
        self.pushButton_pffit.setObjectName("pushButton_pffit")
        self.horizontalLayout_3.addWidget(self.pushButton_pffit)
        self.gridLayout_display.addLayout(self.horizontalLayout_3, 2, 1, 1, 1)
        self.mpl_psf = MatplotlibWidget(self.gridLayoutWidget)
        self.mpl_psf.setObjectName("mpl_psf")
        self.gridLayout_display.addWidget(self.mpl_psf, 1, 0, 1, 1)
        self.horizontalLayoutWidget_4 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_4.setGeometry(QtCore.QRect(200, 520, 741, 54))
        self.horizontalLayoutWidget_4.setObjectName("horizontalLayoutWidget_4")
        self.horizontalLayout_pfsave = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_4)
        self.horizontalLayout_pfsave.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_pfsave.setObjectName("horizontalLayout_pfsave")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_pupilfname = QtWidgets.QLabel(self.horizontalLayoutWidget_4)
        self.label_pupilfname.setObjectName("label_pupilfname")
        self.verticalLayout.addWidget(self.label_pupilfname)
        self.lineEdit_pupilfname = QtWidgets.QLineEdit(self.horizontalLayoutWidget_4)
        self.lineEdit_pupilfname.setObjectName("lineEdit_pupilfname")
        self.verticalLayout.addWidget(self.lineEdit_pupilfname)
        self.horizontalLayout_pfsave.addLayout(self.verticalLayout)
        self.pushButton_pfbrowse = QtWidgets.QPushButton(self.horizontalLayoutWidget_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_pfbrowse.sizePolicy().hasHeightForWidth())
        self.pushButton_pfbrowse.setSizePolicy(sizePolicy)
        self.pushButton_pfbrowse.setObjectName("pushButton_pfbrowse")
        self.horizontalLayout_pfsave.addWidget(self.pushButton_pfbrowse)
        self.pushButton_savepupil = QtWidgets.QPushButton(self.horizontalLayoutWidget_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_savepupil.sizePolicy().hasHeightForWidth())
        self.pushButton_savepupil.setSizePolicy(sizePolicy)
        self.pushButton_savepupil.setObjectName("pushButton_savepupil")
        self.horizontalLayout_pfsave.addWidget(self.pushButton_savepupil)
        self.pushButton_savefit = QtWidgets.QPushButton(self.horizontalLayoutWidget_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_savefit.sizePolicy().hasHeightForWidth())
        self.pushButton_savefit.setSizePolicy(sizePolicy)
        self.pushButton_savefit.setObjectName("pushButton_savefit")
        self.horizontalLayout_pfsave.addWidget(self.pushButton_savefit)
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(30, 29, 501, 51))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout_load = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout_load.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_load.setObjectName("horizontalLayout_load")
        self.pushButton_loadpsf = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_loadpsf.sizePolicy().hasHeightForWidth())
        self.pushButton_loadpsf.setSizePolicy(sizePolicy)
        self.pushButton_loadpsf.setObjectName("pushButton_loadpsf")
        self.horizontalLayout_load.addWidget(self.pushButton_loadpsf)
        self.lineEdit_loadpsf = QtWidgets.QLineEdit(self.horizontalLayoutWidget)
        self.lineEdit_loadpsf.setObjectName("lineEdit_loadpsf")
        self.horizontalLayout_load.addWidget(self.lineEdit_loadpsf)
        self.pushButton_retrieve = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_retrieve.sizePolicy().hasHeightForWidth())
        self.pushButton_retrieve.setSizePolicy(sizePolicy)
        self.pushButton_retrieve.setObjectName("pushButton_retrieve")
        self.horizontalLayout_load.addWidget(self.pushButton_retrieve)
        Form.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Form)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1024, 22))
        self.menubar.setObjectName("menubar")
        Form.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Form)
        self.statusbar.setObjectName("statusbar")
        Form.setStatusBar(self.statusbar)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Phase Retrieval"))
        self.label_NA.setText(_translate("Form", "Numerical aperture"))
        self.lineEdit_NA.setText(_translate("Form", "1.0"))
        self.label_nfrac.setText(_translate("Form", "Refractive index"))
        self.lineEdit_nfrac.setText(_translate("Form", "1.33"))
        self.label_objfl.setText(_translate("Form", "Focal length (mm)"))
        self.lineEdit_objfl.setText(_translate("Form", "9.0"))
        self.label_wlc.setText(_translate("Form", "Wavelength (nm)"))
        self.lineEdit_wlc.setText(_translate("Form", "515"))
        self.label_nwl.setText(_translate("Form", "# wavelengths"))
        self.lineEdit_nwl.setText(_translate("Form", "3"))
        self.label_wlstep.setText(_translate("Form", "Wavelength steps (nm)"))
        self.lineEdit_wlstep.setText(_translate("Form", "5.0"))
        self.label_pxl.setText(_translate("Form", "Pixel size (nm)"))
        self.lineEdit_pxl.setText(_translate("Form", "103"))
        self.label_dz.setText(_translate("Form", "z step (micron)"))
        self.lineEdit_zstep.setText(_translate("Form", "0.40"))
        self.label_mask.setText(_translate("Form", "Mask size (px)"))
        self.lineEdit_mask.setText(_translate("Form", "40"))
        self.label_nIt.setText(_translate("Form", "# of iterations"))
        self.label_psfview.setText(_translate("Form", "PSF view"))
        self.label_pupilview.setText(_translate("Form", "Retrieved pupil"))
        self.pushButton_yzview.setText(_translate("Form", "x-z"))
        self.pushButton_xyview.setText(_translate("Form", "x-y"))
        self.pushButton_xzview.setText(_translate("Form", "x-z"))
        self.pushButton_lineview.setText(_translate("Form", "line"))
        self.checkBox_rm4.setText(_translate("Form", "remove 1-4"))
        self.pushButton_ampli.setText(_translate("Form", "Amplitude"))
        self.pushButton_pffit.setText(_translate("Form", "Fit"))
        self.label_pupilfname.setText(_translate("Form", "Pupil file name"))
        self.pushButton_pfbrowse.setText(_translate("Form", "Browse"))
        self.pushButton_savepupil.setText(_translate("Form", "Save pupil"))
        self.pushButton_savefit.setText(_translate("Form", "Save z-fit"))
        self.pushButton_loadpsf.setText(_translate("Form", "Load psf..."))
        self.pushButton_retrieve.setText(_translate("Form", "Retrieve!"))

from matplotlibwidget import MatplotlibWidget
