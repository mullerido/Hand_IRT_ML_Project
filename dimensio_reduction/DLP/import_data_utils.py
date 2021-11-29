import numpy as np
import os
import pandas as pd
from DLP_Utils import running_mean
import json


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)


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
        subject_ids.append(ind_name.split('_')[0] + '_' + ind_name.split('_')[1])  # + '_')

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
        normalized_data = np.copy(data_cube_smooth[0, :, :])
        mean_data[0, :] = normalized_data.mean(axis=0)
        return normalized_data, mean_data
    else:
        normalized_data = np.copy(data_cube_smooth[0, :, :])
        for roi_i in range(np.shape(all_features)[0]):  # Use the ratio
            normalized_data[roi_i, :] = np.divide(normalized_data[roi_i, :] - normalized_data[roi_i, 0],
                                                  normalized_data[roi_i, 0])
        mean_data[0, :] = normalized_data.mean(axis=0)

        return normalized_data, mean_data


def save_all_gravity_data(save_path, normalize_flag=True, smooth_flag=True, t_samples=39):
    all_features = ['Thumbs_dist_Intence', 'Thumbs_proxy_Intence', 'Ring_dist_Intence', 'Ring_proxy_Intence',
                    'Middle_dist_Intence', 'Middle_proxy_Intence', 'Index_dist_Intence', 'Index_proxy_Intence',
                    'Pinky_dist_Intence', 'Pinky_proxy_Intence', 'Palm_Center_Intence', 'Palm_arch_Intence']

    [data_cube_r, mean_data_r, data_cube_l, mean_data_l, all_subject_ids] = get_gravity_data_paired(all_features,
                                                                                                    normalize_flag,
                                                                                                    smooth_flag,
                                                                                                    t_samples)

    data_dict = {'data_cube_r': np.reshape(data_cube_r, (29 * 12 * 39, 1)),
                 'mean_data_r': np.reshape(mean_data_r, (29 * 39, 1)),
                 'data_cube_l': np.reshape(data_cube_l, (29 * 12 * 39, 1)),
                 'mean_data_l': np.reshape(mean_data_l, (29 * 39, 1)),
                 'all_subject_ids': all_subject_ids,
                 'all_features': all_features,
                 'normalize_flag': normalize_flag,
                 'smooth_flag': smooth_flag,
                 }

    with open(save_path, "w") as write_file:
        json.dump(data_dict, write_file, cls=NumpyArrayEncoder)

def get_relevant_data_from_file(file, t_samples, relevant_features):
    # Parse the data
    with open(file, 'r', encoding="utf-8") as f:
        data_json = json.load(f)

    all_features = data_json['all_features']
    relevant_features_ids = indices = np.where(np.in1d(all_features, relevant_features))[0]

    data_cube_r_all = np.reshape(data_json['data_cube_r'], (29, 12, 39))
    data_cube_r = data_cube_r_all[:, relevant_features_ids, 0:t_samples]
    mean_data_r = np.empty((np.shape(data_cube_r)[0], t_samples))
    for s_id in range(np.shape(data_cube_r)[0]):
        mean_data_r[s_id, :] = np.mean(data_cube_r[s_id, :, :], axis=0)

    data_cube_l_all = np.reshape(data_json['data_cube_l'], (29, 12, 39))
    data_cube_l = data_cube_l_all[:, relevant_features_ids, 0:t_samples]
    mean_data_l = np.empty((np.shape(data_cube_r)[0], t_samples))
    for s_id in range(np.shape(data_cube_r)[0]):
        mean_data_l[s_id, :] = np.mean(data_cube_l[s_id, :, :], axis=0)

    all_subject_ids = data_json['all_subject_ids']

    normalize_flag = data_json['normalize_flag']
    smooth_flag = data_json['smooth_flag']

    return data_cube_r, mean_data_r, data_cube_l, mean_data_l, all_subject_ids