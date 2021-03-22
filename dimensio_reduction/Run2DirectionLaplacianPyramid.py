import numpy as np
import pandas as pd
import os
from common.utils import get_labels_using_gravity_ratio, seperate_subjects_by_reaction
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


def get_gravity_data(all_features, normlize_flag=True, hand=''):
    folder = r'C:\Projects\Hand_IRT_Auto_Ecxtraction\Feature-Analysis- matlab\Resize Feature\\'

    files = os.listdir(folder)
    if not hand or hand == 'both':
        files_xls = [f for f in files if f[-5:] == '.xlsx']
    elif hand == 'right':
        files_xls = [f for f in files if f[-10:] == 'right.xlsx']
    elif hand == 'left':
        files_xls = [f for f in files if f[-9:] == 'left.xlsx']

    names = []
    subject_ids = []
    mean_data = np.empty((np.shape(files_xls)[0], 19))
    data_cube = np.empty((np.shape(files_xls)[0], len(all_features), 19))
    for ind, file in enumerate(files_xls):
        print(file)
        current_file = folder + file
        current_features = pd.read_excel(current_file)
        if normlize_flag:  # Use the ratio
            relevant_cols = (current_features.loc[0:18, all_features] - current_features.loc[0, all_features]) / \
                            current_features.loc[0, all_features]
            mean_data[ind, :] = relevant_cols.mean(axis=1)
        else:  # Use the actual values
            relevant_cols = current_features.loc[0:18, all_features]
            mean_data[ind, :] = np.mean(
                (current_features.loc[0:18, all_features] - current_features.loc[0, all_features]) /
                current_features.loc[0, all_features], axis=1)

        for dat_i, dat in relevant_cols.iterrows():
            data_cube[ind, :, dat_i] = dat.values

        ind_name = file[:-5]
        names.append(ind_name[:])
        subject_ids.append(ind_name.split('_')[0] + '_' + ind_name.split('_')[1] + '_')

    unique_subject_ids = np.unique(subject_ids)

    return data_cube, names, unique_subject_ids, mean_data

def flat_x_direction(data):

    shape = np.shape(data)

    flat_data = np.empty((shape[0], shape[1]*shape[2]))
    for t_i in range(shape[2]):
        current_time = data[:, :, t_i]
        flat_data[:, t_i*shape[1]:t_i*shape[1] + shape[1]] = current_time
    return flat_data

def flat_y_direction(data):

    shape = np.shape(data)

    flat_data = np.empty((0,shape[1]), float)#np.empty((shape[0]*shape[2], shape[1]))
    for t_i in range(shape[2]):
        current_time = data[:, :, t_i]
        flat_data = np.append(flat_data, current_time, axis = 0)

    flat_data_out = flat_data.T
    return flat_data_out

def find_epsilon(dist_data, alpha = 0.5):
    """
    % This is the huristics used to find epsilon
    (see this paper: Heterogeneous Datasets Representation and Learning using Diffusion Maps and Laplacian Pyramids.N Rabin, RR Coifman, SDM, 189 - 199)

    % For input send:
        dist = squareform(pdist(YourInputData));
    % For other
        metrics: dist = squareform(pdist(YourInputData, 'cosine'));
    """
    shape = np.shape(dist_data)
    I_mat = np.identity(shape[0])
    max_val = np.max(dist_data)
    mins = np.min(dist_data + I_mat*max_val, axis=1)

    eps = np.max(mins)

    return eps
if __name__ == "__main__":
    normalize_flag = True
    # allFeatures = GetHandFingersDataRelateToCenter()

    all_features = ['Thumbs_proxy_Intence', 'Index_proxy_Intence', 'Middle_proxy_Intence', 'Ring_proxy_Intence',
                    'Pinky_proxy_Intence']
    # ['Thumbs_dist_Intence', 'Thumbs_proxy_Intence', 'Index_dist_Intence', 'Index_proxy_Intence',
    # 'Middle_dist_Intence', 'Middle_proxy_Intence', 'Ring_dist_Intence', 'Ring_proxy_Intence',
    # 'Pinky_dist_Intence', 'Pinky_proxy_Intence', 'Palm_arch_Intence', 'Palm_Center_Intence']
    hand = 'right'  # either 'right', 'left' or 'both'
    hand_to_extract = 'right'
    [date_cube, names, subject_ids, mean_data] = get_gravity_data(all_features, normalize_flag, hand)

    # Find label
    labels = get_labels_using_gravity_ratio(all_features, hand)

    # Split data into train and test
    data_train, data_test, label_train, label_test = train_test_split(date_cube, labels, test_size=0.25,
                                                                      random_state=0)
    data_train_flat_x = flat_x_direction(data_train)
    pDist_train_flat_x = pdist(data_train_flat_x)
    square_dist_train_x = squareform(pDist_train_flat_x)
    epsilon_train_x = find_epsilon(square_dist_train_x, alpha = 0.5)

    data_train_flat_y = flat_y_direction(data_train)
    pDist_train_flat_y = pdist(data_train_flat_y)
    square_dist_train_y = squareform(pDist_train_flat_y)

    data_test_flat_x = flat_x_direction(data_test)
    data_test_flat_y = flat_y_direction(data_test)


    x = 1
