
from setuptools import setup 
  
setup( 
    name='bigantr_lem', 
    version='0.1', 
    description='BigantrLEM: Landscape Evolution Model implementing BIGANTR theory', 
    author='Greg Tucker', 
    author_email='gtucker@colorado.edu', 
    packages=['bigantr_lem'], 
    install_requires=[ 
        'numpy', 
        'landlab', 
    ], 
)
