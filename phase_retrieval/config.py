'''
This is the configuration class for the Phase_retrieval package.
'''

import yaml

class PR_config(object):
    def __init__(self):
        self.NA = 1.00
        self.objf = 5000
        self.pxl = None
        self.nfrac = None
        self.dz = None
        self.wavelength = None
        self.nwave = None
        self.wstep = None

    def load_config(self, source):
        with open(source, 'r') as fi:
            conf_dict = yaml.load(fi)

        self.set_NA(conf_dict['NA'])
        self.set_nwave(conf_dict['nwave'])
        self.set_objf(conf_dict['obj_f'])
        self.set_pxl(conf_dict['pxl'])
        self.set_dz(conf_dict['dz'])
        self.set_wavelength(conf_dict['wavelength'])
        self.set_wstep(conf_dict['wstep'])


    
