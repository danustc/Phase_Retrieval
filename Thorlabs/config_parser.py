'''
A parser program for Thorlab's deformable mirror configuration files.
'''
import glob
import yaml
from lxml import etree, objectify
import numpy as np

def read_and_build(path):
    '''
    load an xml file and return a list.
    '''
    with open(path, 'r') as stream:
        try:
            conf_list = yaml.load(stream)
            return conf_list.encode() # convert it into bytes 
        except yaml.YAMLError as exc:
            print(exc)

# ----------OK This is a playground for inheritance!----------------- 


class xml_config(object):
    def __init__(self):
        print("yes!")
        self.main = None


    def load_config(self, path):
        conf_list = read_and_build(path)
        self.main = objectify.fromstring(conf_list)

    def get_segments(self):
        '''
        read the segments from self.main class
        '''
        segments = self.main.getchildren()[2]
        print()





class Thorlabs_config(xml_config):
    def __init__(self):
        super().__init__()




def class_converter(conf_list):
    '''
    load a configuration list
    '''




def main():
    TC = Thorlabs_config()
    conf_list = glob.glob()


if __name__ == '__main__':
    main()
