import numpy as np
import pandas as pd
from utils import GetStudyData
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def GetGravityData(all_features, normlize_flag=True, hand = ''):
    folder = r'C:\Projects\Thesis\Feature-Analysis- matlab\Resize Feature\\'

    files = os.listdir(folder)
    if not hand or hand == 'both':
        files_xls = [f for f in files if f[-5:] == '.xlsx']
    elif hand == 'right':
        files_xls = [f for f in files if f[-10:] == 'right.xlsx']
    elif hand == 'left':
        files_xls = [f for f in files if f[-9:] == 'left.xlsx']

    grouped_feature = pd.DataFrame()
    names = []
    subject_ids = []
    data = np.zeros((np.shape(files_xls)[0], len(all_features), 19))

    inc_c = np.empty((19, 19, np.shape(files_xls)[0]))
    for ind, file in enumerate(files_xls):
        print(file)
        current_file = folder + file
        current_features = pd.read_excel(current_file)
        if normlize_flag:  # Use the ratio
            relevant_cols = (current_features.loc[0:18, all_features] - current_features.loc[0, all_features]) / \
                            current_features.loc[0, all_features]
        else:  # Use the actual values
            relevant_cols = current_features.loc[:, all_features]
        # transformDataFrame = pd.DataFrame(relevant_cols).T

        for dat_i, dat in relevant_cols.iterrows():
            data[ind, :, dat_i] = dat.values

        ind_name = file[:-5]
        names.append(ind_name[:])
        subject_ids.append(ind_name.split('_')[0] + '_' + ind_name.split('_')[1] + '_')

    unique_subject_ids = np.unique(subject_ids)

    return data, names, unique_subject_ids

def Calc_One_Directional_Laplacian_Pyramid(data):

    t_size = np.shape(data)[2]
    s_size = np.shape(data)[0]
    g_0 = np.empty((s_size, s_size, t_size))
    sigma_0 = 1000
    for t_ind in range(t_size):
        for ind in range(s_size):
            for jnd in range(s_size):

                g_0[ind, jnd, t_ind] = np.exp(-np.sqrt(np.linalg.norm(data[ind, :, t_ind]-data[jnd, :, t_ind]))/sigma_0)
    x=1

    ep_factor = 4

if __name__ == "__main__":
    normlizeFlag = True
    # allFeatures = GetHandFingersDataRelateToCenter()

    all_features = ['Thumbs_proxy_Intence', 'Index_proxy_Intence', 'Middle_proxy_Intence', 'Ring_proxy_Intence',
                   'Pinky_proxy_Intence']
                    #['Thumbs_dist_Intence', 'Thumbs_proxy_Intence', 'Index_dist_Intence', 'Index_proxy_Intence',
                   #'Middle_dist_Intence', 'Middle_proxy_Intence', 'Ring_dist_Intence', 'Ring_proxy_Intence',
                   #'Pinky_dist_Intence', 'Pinky_proxy_Intence', 'Palm_arch_Intence', 'Palm_Center_Intence']

    hand_to_extract = 'right' # either 'right', 'left' or 'both'
    # [groupedFeature, names, subject_id] = GetStudyData(allFeatures, normlizeFlag)
    [grouped_feature, names, subject_ids] = GetGravityData(all_features, normlize_flag=True, hand='right')
    Calc_One_Directional_Laplacian_Pyramid(grouped_feature)