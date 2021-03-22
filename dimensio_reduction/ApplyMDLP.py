import numpy as np
import pandas as pd
from utils import GetStudyData
import os
import matplotlib.pyplot as plt

normlizeFlag = True
allFeatures = ['Thumbs_dist_Intence', 'Thumbs_proxy_Intence', 'Index_dist_Intence', 'Index_proxy_Intence',
                   'Middle_dist_Intence', 'Middle_proxy_Intence', 'Ring_dist_Intence', 'Ring_proxy_Intence',
                   'Pinky_dist_Intence', 'Pinky_proxy_Intence', 'Palm_arch_Intence', 'Palm_Center_Intence']
allFeatures = ['Thumbs_proxy_Intence', 'Index_proxy_Intence', 'Middle_proxy_Intence', 'Ring_proxy_Intence',
                   'Pinky_proxy_Intence']

folder = r'C:\Projects\Thesis\Feature-Analysis- matlab\Resize Feature\\'

files = os.listdir(folder)
files_xls = [f for f in files if f[-5:] == '.xlsx']
#files_xls = [f for f in files if f[-10:] == 'right.xlsx']
groupedFeature = pd.DataFrame()
names = []
data = np.zeros((np.shape(files_xls)[0], np.shape(allFeatures)[0], 39))
for ind, file in enumerate(files_xls):
    print(file)
    currentFile = folder + file
    currentFeatures = pd.read_excel(currentFile)
    if normlizeFlag:  # Use the ratio
        relevantCols = (currentFeatures.loc[:, allFeatures] - currentFeatures.loc[0, allFeatures]) / \
                        currentFeatures.loc[0, allFeatures]
    else:  # Use the actual values
        relevantCols = currentFeatures.loc[:, allFeatures]

    # Create the 3D matrix with:
    # i- subject, j- ROI and z - time
    data[ind] = relevantCols.T


    indName = file[:-5]
    names.append(indName[:])