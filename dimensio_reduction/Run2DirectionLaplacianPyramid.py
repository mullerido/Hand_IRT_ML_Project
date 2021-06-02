import numpy as np
import pandas as pd
import os
from common.utils import get_labels_using_gravity_ratio, seperate_subjects_by_reaction
from scipy.spatial.distance import pdist, squareform, cdist
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

    flat_data = np.empty((shape[0], shape[1] * shape[2]))
    for t_i in range(shape[2]):
        current_time = data[:, :, t_i]
        flat_data[:, t_i * shape[1]:t_i * shape[1] + shape[1]] = current_time
    return flat_data


def flat_y_direction(data):
    shape = np.shape(data)

    flat_data = np.empty((0, shape[1]), float)  # np.empty((shape[0]*shape[2], shape[1]))
    for t_i in range(shape[2]):
        current_time = data[:, :, t_i]
        flat_data = np.append(flat_data, current_time, axis=0)

    flat_data_out = flat_data.T
    return flat_data_out


def find_epsilon(dist_data, alpha=0.5):
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
    mins = np.min(dist_data + I_mat * max_val, axis=1)

    eps = np.max(mins) * alpha

    return eps


def run_ALP_model_train_itterations(ALP_model, data_end, num_itter=10):
    size_of_end = np.shape(data_end)[0] * np.shape(data_end)[1] * np.shape(data_end)[2]
    data_end_flat_x = flat_x_direction(data_end)
    data_end_flat_initial = np.reshape(data_end_flat_x, [size_of_end, 1])
    initial_data_end = np.copy(data_end)
    for itter in range(num_itter):

        sigma_x = ALP_model['sigma_x'][0, itter]
        data_end_flat_x = flat_x_direction(data_end)
        [kernel_x, approx_flat_x] = run_ALP_1D(ALP_model['square_dist_x'], sigma_x, data_end_flat_x)
        ALP_model['Kernel_x'][:, :, itter] = kernel_x
        ALP_model['end_approx_x'][itter] = cube_data_from_x_flat(approx_flat_x, np.shape(data_end)[0],
                                                                 np.shape(data_end)[1], np.shape(data_end)[2])

        data_end_flat_y = flat_y_direction(data_end)
        sigma_y = ALP_model['sigma_y'][0, itter]
        [kernel_y, approx_flat_y] = run_ALP_1D(ALP_model['square_dist_y'], sigma_y, data_end_flat_y)
        ALP_model['Kernel_y'][:, :, itter] = kernel_y
        ALP_model['end_approx_y'][itter] = cube_data_from_y_flat(approx_flat_y, np.shape(data_end)[0],
                                                                 np.shape(data_end)[1], np.shape(data_end)[2])
        ALP_model['end_approx'][itter] = 0.5 * ALP_model['end_approx_x'][itter] + 0.5 * ALP_model['end_approx_y'][itter]

        if itter == 0:
            ALP_model['end_multiscale'][itter] = ALP_model['end_approx'][itter]

        else:
            ALP_model['end_multiscale'][itter] = ALP_model['end_multiscale'][itter - 1] + ALP_model['end_approx'][itter]

        end_multiscale_flat_x = flat_x_direction(ALP_model['end_multiscale'][itter])
        end_multiscale_flat = np.reshape(end_multiscale_flat_x, [size_of_end, 1])

        err_vec_b = data_end_flat_initial - end_multiscale_flat
        err_vec_b = np.power(err_vec_b, 2)
        ALP_model['error_itter'][0, itter] = np.sum(err_vec_b) / size_of_end
        ALP_model['root_error_itter'][0, itter] = np.sqrt(ALP_model['error_itter'][0, itter])

        ALP_model['dist'][itter] = initial_data_end - ALP_model['end_multiscale'][itter]

        data_end = ALP_model['dist'][itter]
        ALP_model['sigma_x'][0, itter + 1] = ALP_model['sigma_x'][0, itter] / 2
        ALP_model['sigma_y'][0, itter + 1] = ALP_model['sigma_y'][0, itter] / 2

    return ALP_model


def run_ALP_1D(square_dist, sigma, data_end_flat):
    kernel = np.exp(-(np.power(square_dist, 2) / np.power(sigma, 2)))
    # zero diagonal
    for ind in range(len(kernel)):
        kernel[ind, ind] = 0
    # ave kernel
    for ind in range(len(kernel)):
        s = np.sum(kernel[ind, :])
        if s > 1e-6:
            kernel[ind, :] = kernel[ind, :] / s
        else:
            kernel[ind, :] = np.zeros((1, len(kernel)))

    approx_flat = np.zeros((np.shape(data_end_flat)[0], np.shape(data_end_flat)[1]))
    for row_ind in range(np.shape(data_end_flat)[0]):
        for col_ind in range(np.shape(data_end_flat)[0]):
            approx_flat[row_ind, :] = approx_flat[row_ind, :] + kernel[row_ind, col_ind] * data_end_flat[col_ind, :]

    return kernel, approx_flat


def cube_data_from_x_flat(flat_x, x, y, z):
    cube_data = np.zeros((x, y, z))
    j = 1
    for t_ind in range(z):
        t_values = flat_x[:, (j - 1) * y: j * y]
        cube_data[:, :, t_ind] = t_values
        j += 1

    return cube_data


def cube_data_from_y_flat(flat_y, x, y, z):
    cube_data = np.zeros((x, y, z))
    flat_y = flat_y.T
    i = 1
    for t_ind in range(z):
        t_values = flat_y[(i - 1) * x:i * x, :]
        cube_data[:, :, t_ind] = t_values
        i += 1
    return cube_data


def convergence_ROIs(data_cube):
    data_2D = np.zeros([np.shape(data_cube)[0], np.shape(data_cube)[2]])
    for s_ind in range(np.shape(data_cube)[0]):
        for t_ind in range(np.shape(data_cube)[2]):
            data_2D[s_ind, t_ind] = np.mean(data_cube[s_ind, :, t_ind])

    return data_2D


def weighed_dist(data, labels, w=1):
    """
        Calculate the distance matrix of data and apply weight based on the label
    """
    # dist_list = pdist(data, 'euclidean')
    # dist_sqare = squareform(dist_list)
    dist_sqare = cdist(data, data, 'euclidean')

    unique_labels = np.unique(labels)
    for c_l in unique_labels:
        same_label_ids = np.where(labels == c_l)
        for r in same_label_ids[0]:
            for c in same_label_ids[0]:
                dist_sqare[r, c] = dist_sqare[r, c] * w

    return dist_sqare


def run_ALP_model_test_itterations(ALP_model, data_test_start, data_test_end_true, data_train_end, min_ind):
    end_test_model = {'multiscale': [dict() for x in range(min_ind + 1)],
                      'approx': [dict() for x in range(min_ind + 1)],
                      'mse_error': np.zeros((1, min_ind + 1)),
                      'root_mse_error': np.zeros((1, min_ind + 1))}  # REMOVE THE (-1) WHEN ITS NOT THE 0 INDEX!!!
    data_test_end = data_train_end
    # k_i = 0
    if np.ndim(data_test_start) == 3:
        x_flat_s = np.shape(data_test_start)[0]
        y_flat_s = np.shape(data_test_start)[1] * np.shape(data_test_start)[2]
        x_flat_e = np.shape(data_test_end_true)[0]
        y_flat_e = np.shape(data_test_end_true)[1] * np.shape(data_test_end_true)[2]
        data_test_end_true_flat = np.reshape(data_test_end_true, [x_flat_e, y_flat_e])
    elif np.ndim(data_test_start) == 2:
        x_flat_s = 1
        y_flat_s = np.shape(data_test_start)[0] * np.shape(data_test_start)[1]
        x_flat_e = 1
        y_flat_e = np.shape(data_test_end_true)[0] * np.shape(data_test_end_true)[1]
        data_test_end_true_flat = np.reshape(data_test_end_true, [x_flat_e, y_flat_e])

    for k_i in range(min_ind + 1):

        date_test_start_flat_x = np.reshape(data_test_start, [x_flat_s, y_flat_s], order='F')

        p_dist = cdist(ALP_model['flat_x'], date_test_start_flat_x, 'euclidean')

        kernel_x_test = np.exp(-(np.power(p_dist, 2) / np.power(ALP_model['sigma_x'][0, k_i], 2)))
        for ind in range(np.shape(kernel_x_test)[1]):
            s = np.sum(kernel_x_test[:, ind])
            if s > 1e-6:
                kernel_x_test[:, ind] = kernel_x_test[:, ind] / s
            else:
                kernel_x_test[:, ind] = np.zeros((1, len(kernel_x_test)))

        data_test_end_flat_x = flat_x_direction(data_test_end)
        end_flat_for_kernel_X = np.zeros([x_flat_s,
                                          np.shape(data_test_end)[1] * np.shape(data_test_end)[2]])

        for subject_ind in range(np.shape(kernel_x_test)[1]):
            for row_ind in range(np.shape(kernel_x_test)[0]):
                end_flat_for_kernel_X = end_flat_for_kernel_X + \
                                        kernel_x_test[row_ind, subject_ind] * data_test_end_flat_x[row_ind, :]

        end_test_model['approx'][k_i] = cube_data_from_x_flat(end_flat_for_kernel_X, x_flat_s,
                                                              np.shape(data_train_end)[1], np.shape(data_train_end)[2])

        if k_i == 0:
            end_test_model['multiscale'][k_i] = end_test_model['approx'][k_i]
        else:
            end_test_model['multiscale'][k_i] = end_test_model['multiscale'][k_i - 1] + end_test_model['approx'][k_i]

        data_test_end = ALP_model['dist'][k_i]

        end_test_multiscale_flat = np.reshape(end_test_model['multiscale'][k_i], [x_flat_e, y_flat_e], order='F')
        error_vec_test = data_test_end_true_flat - end_test_multiscale_flat
        error_vec_test_2 = np.power(error_vec_test, 2)
        end_test_model['mse_error'][0, k_i] = np.sum(error_vec_test_2) / y_flat_e
        end_test_model['root_mse_error'][0, k_i] = np.sqrt(end_test_model['mse_error'][0, k_i])

    end_test_multiscale_model = end_test_model['multiscale'][min_ind]
    return end_test_multiscale_model


def plot_comparison_per_subject_per_ROI(end_model, start_data, data_test_end_true_cube, ROI_S):
    for s_ind in range(np.shape(end_model)[0]):
        fig, ax = plt.subplots(ROI_S)
        fig.suptitle('Test Group Prediction Comparison', fontsize=20)

        for f_ind in range(ROI_S):
            trueData = np.concatenate((start_data[s_ind, :], data_test_end_true_cube[s_ind, f_ind, :]))
            predictedData = np.concatenate((start_data[s_ind, :], end_model[s_ind, f_ind, :]))

            ax[f_ind].plot(trueData, color='forestgreen')
            ax[f_ind].plot(predictedData, color='indianred')
            ax[f_ind].plot(start_data[s_ind, :])

        fig.legend(['True', 'Predict', 'Start'], loc='lower right')
        plt.show()


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

    # Split the data automatically
    # Find label
    labels = get_labels_using_gravity_ratio(all_features, hand)

    # Split data into train and test
    # [data_train, data_test, label_train, label_test] = train_test_split(date_cube, labels, test_size=0.2,
    #                                                                   random_state=0)
    # Split data manually
    # '''

    train_inds = np.arange(28)
    test_ind = 28
    data_train = date_cube[train_inds, :, :]
    label_train = labels.iloc[train_inds, :]
    data_test = date_cube[test_ind, :, :]
    label_test = labels.iloc[[test_ind]]
    # '''

    data_train_start = data_train[:, :, 0:6]  # Take the first 6 samples
    data_train_end = data_train[:, :, 6:]

    data_train_start_flat_x = flat_x_direction(data_train_start)
    square_dist_train_start_x = weighed_dist(data_train_start_flat_x, label_train, 0.1)
    epsilon_train_start_x = find_epsilon(square_dist_train_start_x, alpha=0.5)

    data_train_flat_start_y = flat_y_direction(data_train_start)
    # pDist_train_flat_start_y = pdist(data_train_flat_start_y, 'euclidean')
    # square_dist_train_start_y = squareform(pDist_train_flat_start_y)
    square_dist_train_start_y = cdist(data_train_flat_start_y, data_train_flat_start_y, 'euclidean')
    epsilon_train_start_y = find_epsilon(square_dist_train_start_y, alpha=0.5)

    num_itterations = 10

    itter_epsilon_train_start_x = np.zeros((1, num_itterations + 1))
    itter_epsilon_train_start_x[0, 0] = epsilon_train_start_x
    itter_epsilon_train_start_y = np.zeros((1, num_itterations + 1))
    itter_epsilon_train_start_y[0, 0] = epsilon_train_start_y

    ALP_model = {'flat_x': data_train_start_flat_x,
                 'flat_y': data_train_flat_start_y,
                 'square_dist_x': square_dist_train_start_x,
                 'square_dist_y': square_dist_train_start_y,
                 'sigma_x': itter_epsilon_train_start_x,
                 'sigma_y': itter_epsilon_train_start_y,
                 'Kernel_x': np.zeros(
                     (np.shape(square_dist_train_start_x)[0], np.shape(square_dist_train_start_x)[1], num_itterations)),
                 'Kernel_y': np.zeros(
                     (np.shape(square_dist_train_start_y)[0], np.shape(square_dist_train_start_y)[1], num_itterations)),
                 'end_approx_x': [dict() for x in range(num_itterations)],
                 'end_approx_y': [dict() for x in range(num_itterations)],
                 'end_approx': [dict() for x in range(num_itterations)],
                 'end_multiscale': [dict() for x in range(num_itterations)],
                 'dist': [dict() for x in range(num_itterations)],
                 'error_itter': np.zeros((1, num_itterations)),
                 'root_error_itter': np.zeros((1, num_itterations))}

    # Run the model on training cases
    ALP_model = run_ALP_model_train_itterations(ALP_model, data_train_end, num_itterations)
    min_ind = np.argmin(ALP_model['error_itter'])
    if np.ndim(data_test) == 3:
        data_test_start = data_test[:, :, 0:6]  # Take the first 6 samples
        data_test_end_true = data_test[:, :, 6:]
        data_test_end_true_cube = data_test_end_true
    elif np.ndim(data_test) == 2:
        data_test_start = data_test[:, 0:6]  # data_test[0, :, 0:5]  # Take the first 6 samples
        data_test_end_true = data_test[:, 6:]  # data_test[0, :, 6:]
        data_test_end_true_cube = np.reshape(data_test_end_true,
                                             [1, np.shape(data_test_end_true)[0], np.shape(data_test_end_true)[1]])

    # Run the model on test cases- for prediction
    end_model = run_ALP_model_test_itterations(ALP_model, data_test_start, data_test_end_true, data_train_end, min_ind)

    # Average ROIs per patient- from cube 2 2D
    data_train_2D = convergence_ROIs(data_train)

    # Prepare plot data(mean and STD) per time per label group
    group_plot_data = {}
    unique_labels = np.unique(labels)
    for l_ind in unique_labels:
        match_lables_ids = np.where((label_train == l_ind))
        match_lables_data = data_train_2D[match_lables_ids[0], :]
        match_lables_mean = match_lables_data.mean(axis=0)
        match_lables_std = match_lables_data.std(axis=0)

        group_plot_data[str(l_ind)] = np.array([match_lables_mean, match_lables_std]).T

    # @@@@@ Run over test patient and plot them inside the area of the group_plot_data
    x = np.linspace(0, len(match_lables_std) - 1, len(match_lables_std))
    for s_ind in range(np.shape(end_model)[0]):
        s_label = label_test.iloc[s_ind, 0]
        label_group_data = group_plot_data[str(s_label)]
        fig, ax = plt.subplots()
        plt.suptitle('Test Group Prediction Comparison- ' + label_test.index[s_ind], fontsize=20)
        ax.fill_between(x, label_group_data[:, 0] - label_group_data[:, 1],
                        label_group_data[:, 0] + label_group_data[:, 1], color='C2', alpha=0.2)

        if np.ndim(data_test) == 3:
            s_real_test = data_test[s_ind, :, :].mean(axis=0)
            s_model_test = np.concatenate(
                (data_test_start[s_ind, :, :].mean(axis=0), end_model[s_ind, :, :].mean(axis=0)))
        else:
            s_real_test = data_test.mean(axis=0)
            s_real_test_start = s_real_test[0:6]
            s_model_test = np.concatenate((s_real_test_start, end_model[s_ind, :, :].mean(axis=0)))

        ax.plot(x, s_real_test, color='g')
        ax.plot(x, s_model_test, color='r')
        ax.legend(['Real', 'Predict', 'group'], loc='lower right')

        plt.show()

    plot_comparison_per_subject_per_ROI(end_model, data_test_start, data_test_end_true_cube, np.shape(all_features)[0])

x = 1
