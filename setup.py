'''
The setup file for the package phase_retrieval
'''
from setuptools import setup, find_packages

setup(name = 'phase_retrieval',
        version = '0.1',
        description = 'A small downstream analysis package',
        url = 'https://github.com/danustc/Phase_retrieval',
        author = 'danustc',
        author_email = 'Dan.Xie@ucsf.edu',
        license = 'UCSF',
        packages = find_packages(),
        zip_safe = False
        )
