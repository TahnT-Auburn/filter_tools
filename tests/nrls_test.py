#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from filter_tools.estimators import Estimators
from python_utilities.parsers_class import Parsers

##### Vehicle Mass Estimation using NRLS #####

if __name__ == '__main__':
    
    data = scipy.io.loadmat('data/u5a_p_sg_ct1.mat')
