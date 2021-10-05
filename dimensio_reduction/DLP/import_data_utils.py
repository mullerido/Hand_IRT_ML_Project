import numpy as np
import os
import pandas as pd
from DLP_Utils import running_mean


def get_gravity_data(all_features, normlize_flag=True, smooth_flag=True, hand='', t_samples=19):
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
    mean_data = np.empty((np.shape(files_xls)[0], t_samples))
    data_cube = np.empty((np.shape(files_xls)[0], len(all_features), t_samples))
    data_cube_s = np.empty((np.shape(files_xls)[0], len(all_features), t_samples))
    for ind, file in enumerate(files_xls):
        print(file)
        current_file = folder + file
        current_features = pd.read_excel(current_file)

        relevant_cols = current_features.loc[0:t_samples - 1, all_features]

        for dat_i, dat in relevant_cols.iterrows():
            data_cube[ind, :, dat_i] = dat.values

        data_cube_s[ind, :, :] = np.copy(data_cube[ind, :, :])
        if smooth_flag:
            for roi_i in range(np.shape(all_features)[0]):
                data_cube_s[ind, roi_i, :] = running_mean(data_cube_s[ind, roi_i, :], 5)

                # data_cube[ind, roi_i, 1:] = np.convolve(data_cube[ind, roi_i, 1:], box,
                #                                        mode='same') / 5  # mode='same') / 5

        normalized_data = np.copy(data_cube_s[ind, :, :])
        for roi_i in range(np.shape(all_features)[0]):  # Use the ratio
            normalized_data[roi_i, :] = np.divide(normalized_data[roi_i, :] - normalized_data[roi_i, 0],
                                                  normalized_data[roi_i, 0])
        mean_data[ind, :] = normalized_data.mean(axis=0)

        if normlize_flag:
            data_cube_s[ind, :, :] = normalized_data
        '''
            # relevant_cols = (current_features.loc[0:t_samples - 1, all_features] - current_features.loc[
            #    0, all_features]) / \
            #                current_features.loc[0, all_features]
            mean_data[ind, :] = data_cube[ind, :, :].mean(axis=1)
        else:  # Use the actual values
            mean_data[ind, :] = np.mean(data_cube[ind, :, :] - data_cube[ind, :, 0] / data_cube[ind, :, 0], axis=1)
            # mean_data[ind, :] = np.mean(
            #    (current_features.loc[0:t_samples - 1, all_features] - current_features.loc[0, all_features]) /
            #    current_features.loc[0, all_features], axis=1)
        '''

        ind_name = file[:-5]
        names.append(ind_name[:])
        subject_ids.append(ind_name.split('_')[0] + '_' + ind_name.split('_')[1] + '_')

    unique_subject_ids = np.unique(subject_ids)
    # data_cube = data_cube_s.copy()
    return data_cube_s, names, unique_subject_ids, mean_data


def get_gravity_data_paired(all_features, normalize_flag, smooth_flag, t_samples):
    folder = r'C:\Projects\Hand_IRT_Auto_Ecxtraction\Feature-Analysis- matlab\Resize Feature\\'
    files = os.listdir(folder)
    files_xls = [f for f in files if f[-10:] == 'right.xlsx']

    all_subject_ids = []
    mean_data_r = np.empty((np.shape(files_xls)[0], t_samples))
    mean_data_l = np.empty((np.shape(files_xls)[0], t_samples))
    data_cube_r = np.empty((np.shape(files_xls)[0], len(all_features), t_samples))
    data_cube_l = np.empty((np.shape(files_xls)[0], len(all_features), t_samples))

    for ind, file in enumerate(files_xls):
        s_name = file[:-11]
        all_subject_ids.append(s_name)
        print(s_name)

        current_file_r = folder + file
        data_cube_r[ind, :, :], mean_data_r[ind, :] = get_patient_data(current_file_r, all_features,
                                                                       normalize_flag, smooth_flag, t_samples)

        current_file_l = folder + s_name + '_left.xlsx'
        data_cube_l[ind, :, :], mean_data_l[ind, :] = get_patient_data(current_file_l, all_features,
                                                                       normalize_flag, smooth_flag, t_samples)

    return data_cube_r, mean_data_r, data_cube_l, mean_data_l, all_subject_ids


def get_patient_data(current_file, all_features, normalize_flag, smooth_flag, t_samples):
    current_features = pd.read_excel(current_file)

    relevant_cols = current_features.loc[0:t_samples - 1, all_features]
    data_cube = np.empty((1, len(all_features), t_samples))
    data_cube_smooth = np.empty((1, len(all_features), t_samples))
    mean_data = np.empty((1, t_samples))

    for dat_i, dat in relevant_cols.iterrows():
        data_cube[0, :, dat_i] = dat.values

    data_cube_smooth[0, :, :] = np.copy(data_cube[0, :, :])
    if smooth_flag:
        for roi_i in range(np.shape(all_features)[0]):
            data_cube_smooth[0, roi_i, :] = running_mean(data_cube_smooth[0, roi_i, :], 5)

            # data_cube[ind, roi_i, 1:] = np.convolve(data_cube[ind, roi_i, 1:], box,
            #                                        mode='same') / 5  # mode='same') / 5
    if not normalize_flag:
        return data_cube_smooth, mean_data
    else:
        normalized_data = np.copy(data_cube_smooth[0, :, :])
        for roi_i in range(np.shape(all_features)[0]):  # Use the ratio
            normalized_data[roi_i, :] = np.divide(normalized_data[roi_i, :] - normalized_data[roi_i, 0],
                                                  normalized_data[roi_i, 0])
        mean_data[0, :] = normalized_data.mean(axis=0)

        return normalized_data, mean_data
