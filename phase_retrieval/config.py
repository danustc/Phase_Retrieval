'''
This is the configuration class for the Phase_retrieval package.
'''

import yaml

class PR_config(object):
    def __init__(self):
        self.NA = None
        self.objf = None
        self.pxl = None
        self.nfrac = None
        self.dz = None
        self.wavelength = None
        self.nwave = None
        self.wstep = None

    def load_config(self, path):
        for key, item in yaml.load(path):
            pass
