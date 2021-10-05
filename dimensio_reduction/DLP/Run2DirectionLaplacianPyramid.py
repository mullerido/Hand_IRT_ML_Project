import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common.utils import get_labels_using_gravity_ratio, seperate_subjects_by_reaction
import matplotlib.pyplot as plt

from dimensio_reduction.DLP.DLP_Utils import get_ave_of_k_nearest_data
from dimensio_reduction.DLP.DLP_plot import plot_test_vs_group, plot_test_progress_vs_k_nearest, \
    plot_comparison_per_subject_per_ROI
from dimensio_reduction.DLP.import_data_utils import get_gravity_data
from dimensio_reduction.DLP.run_DLP_model import run_ALP_model_train, run_ALP_model_test_iterations

if __name__ == "__main__":
    normalize_flag = False
    # allFeatures = GetHandFingersDataRelateToCenter()
    per_patient_result = []
    all_features = ['Thumbs_proxy_Intence', 'Index_proxy_Intence', 'Middle_proxy_Intence', 'Ring_proxy_Intence',
                    'Pinky_proxy_Intence']
    t_samples = 19  # Should be between 1 and 39- where 19 is for the gravitation phase and 39 for the entire session
    save_folder = r'C:\Users\ido.DM\Desktop\Temp\Random Test-train split\\'  # 'C:\Users\ido.DM\Google Drive\Thesis\Project\Results\Two-D Laplacian\One Vs all\\'
    all_dist_method = [
        'euclidean']  # , 'correlation', 'cosine', 'minkowski']  # 'minkowski'  # 'cosine'#'euclidean'/'correlation'
    # ['Thumbs_dist_Intence', 'Thumbs_proxy_Intence', 'Index_dist_Intence', 'Index_proxy_Intence',
    # 'Middle_dist_Intence', 'Middle_proxy_Intence', 'Ring_dist_Intence', 'Ring_proxy_Intence',
    # 'Pinky_dist_Intence', 'Pinky_proxy_Intence', 'Palm_arch_Intence', 'Palm_Center_Intence']
    hand = 'right'  # either 'right', 'left' or 'both'
    hand_to_extract = 'right'
    smooth_flag = True
    [date_cube, names, subject_ids, mean_data] = get_gravity_data(all_features, normalize_flag, smooth_flag, hand,
                                                                  t_samples)

    n_ROIs = len(all_features)
    n_time = np.shape(date_cube)[2]
    n_start = 6  # int(np.round(n_time/3))
    num_iterations = 10
    dim_ratio = [0.5, 0.5]
    alpha = 0.5
    # Split the data automatically
    # Find label
    labels = get_labels_using_gravity_ratio(mean_data, names, all_features, hand, t_samples)
    s_range = np.arange(np.shape(date_cube)[0])
    # rng = np.random.RandomState(2000)

    # use for one Vs. all
    loopRange = range(np.shape(mean_data)[0])
    # use for random 80$-20% split
    # loopRange = range(100)
    for loop_ind in loopRange:
        loop_ind = 2
        print(loop_ind / 100)
        # '''
        save_prefix = str(loop_ind)
        # Split data into train and test
        [data_train, data_test, label_train, label_test] = train_test_split(date_cube, labels, test_size=0.2,
                                                                            random_state=loop_ind)

        '''
        # Split data manually
        save_prefix = ''
        test_ind = loop_ind
        train_inds = s_range[np.arange(len(s_range)) != test_ind]

        data_train = date_cube[train_inds, :, :]
        label_train = labels.iloc[train_inds, :]
        data_test = date_cube[test_ind, :, :].reshape(1, np.shape(date_cube)[1], np.shape(date_cube)[2])
        label_test = labels.iloc[[test_ind]]
        '''
        n_train = np.shape(data_train)[0]
        n_test = np.shape(data_test)[0]

        data_train_start = data_train[:, :, 0:n_start]  # Take the first 6 samples
        data_train_end = data_train[:, :, n_start:]

        for idx, dist_method in enumerate(all_dist_method):

            ALP_model = run_ALP_model_train(data_train_start, data_train_end, label_train, num_iterations, alpha,
                                            dist_method, dim_ratio)
            min_ind = np.argmin(ALP_model['root_error_itter'])

            data_test_start = data_test[:, :, 0:n_start]  # Take the first 6 samples
            data_test_end_true = data_test[:, :, n_start:]
            data_test_end_true_cube = data_test_end_true

            # Run the model on test cases- for prediction
            end_model, complete_end_model = run_ALP_model_test_iterations(ALP_model, data_test_start,
                                                                           data_test_end_true,
                                                                           data_train_end, min_ind)

            ave_nearest = get_ave_of_k_nearest_data(data_train, data_test, n_start)

            for s_test_ind in range(n_test):
                # Distances between real test and the model prediction
                ave_data_test_end_true = data_test_end_true[s_test_ind, :, :].mean(axis=0)
                ave_end_model = end_model[s_test_ind, :, :].mean(axis=0)
                distances_to_model = np.abs(ave_end_model - ave_data_test_end_true)
                ave_distances_to_model = distances_to_model.mean(axis=0)
                last_distance_to_model = distances_to_model[-1]
                # Distances between real test and K- nearest
                distances_to_k_nearest = np.abs(ave_nearest[s_test_ind, n_start:] - ave_data_test_end_true)
                ave_distances_to_k_nearest = distances_to_k_nearest.mean(axis=0)
                last_distance_to_k_nearest = distances_to_k_nearest[-1]

                per_patient_result.append([save_prefix, label_test.index[s_test_ind], dist_method,
                                           ave_distances_to_model, last_distance_to_model,
                                           ave_distances_to_k_nearest, last_distance_to_k_nearest])

            if dist_method == 'euclidean':
                plot_test_vs_group(end_model, data_train, data_test, data_test_start, n_start, labels, label_test,
                                   label_train, save_folder, save_prefix)

                # plot_test_vs_k_nearest(end_model, data_train, data_test, label_test, n_start, save_folder, min_ind,
                # save_prefix)

                plot_test_progress_vs_k_nearest(complete_end_model, data_test, ave_nearest, label_test, n_start,
                                                save_folder, save_prefix, range(min_ind + 1))

                plot_comparison_per_subject_per_ROI(end_model, data_test_start, data_test_end_true,
                                                    n_ROIs, label_test, n_start, save_folder, save_prefix)

                plt.close('all')
            # except:
            #    x = 1

    analysis_headers = ['n_run', 's_name', 'type', 'ave_dist_model', 'last_dist_model', 'ave_dist_k_nearest',
                        'last_dist_k_nearest']

    analysis_df = pd.DataFrame(data=per_patient_result, columns=analysis_headers).set_index('s_name')
    analysis_df.to_excel(r'C:\Users\ido.DM\Google Drive\Thesis\Project\Results\Two-D Laplacian\temp.xlsx')
