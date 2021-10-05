import numpy as np
from sklearn.model_selection import train_test_split
from common.utils import get_labels_using_gravity_ratio, seperate_subjects_by_reaction
from dimensio_reduction.DLP.DLP_Utils import convergence_ROIs
from dimensio_reduction.DLP.import_data_utils import get_gravity_data, get_gravity_data_paired
from dimensio_reduction.DLP.run_DLP_model import run_ALP_model_train, run_ALP_model_test_iterations
from dimensio_reduction.DLP.DLP_plot import plot_hand_before_after, plot_lines, plot_subject_lines, plot_lines_subplots
import matplotlib.pyplot as plt
import pandas as pd
from dimensio_reduction.ApplyPCA import run_pca_on_df
from dimensio_reduction import plotPCA
import random
from sklearn.manifold import TSNE
from dimensio_reduction.Apply_tSNE import vizualise_tsne_and_get_distance, run_tSNE_per_input
from sklearn.decomposition import PCA  # Use to perform the PCA transform

if __name__ == "__main__":
    normalize_flag = True
    # allFeatures = GetHandFingersDataRelateToCenter()
    per_patient_result = []
    all_features = ['Palm_Center_Intence', 'Thumbs_proxy_Intence', 'Index_proxy_Intence', 'Middle_proxy_Intence',
                    'Ring_proxy_Intence',
                    'Pinky_proxy_Intence']
    t_samples = 39  # Should be between 1 and 39- where 19 is for the gravitation phase and 39 for the entire session
    save_folder = r'G:\My Drive\Thesis\Project\Results\Two-D Laplacian\per subject- single hand prediction- n more then 23\\'  # r'C:\Users\ido.DM\Desktop\Temp\Random Test-train split\both hands\\'  # 'C:\Users\ido.DM\Google Drive\Thesis\Project\Results\Two-D Laplacian\One Vs all\\'

    all_dist_method = [
        'euclidean']  # , 'correlation', 'cosine', 'minkowski']  # 'minkowski'  # 'cosine'#'euclidean'/'correlation'
    # ['Thumbs_dist_Intence', 'Thumbs_proxy_Intence', 'Index_dist_Intence', 'Index_proxy_Intence',
    # 'Middle_dist_Intence', 'Middle_proxy_Intence', 'Ring_dist_Intence', 'Ring_proxy_Intence',
    # 'Pinky_dist_Intence', 'Pinky_proxy_Intence', 'Palm_arch_Intence', 'Palm_Center_Intence']
    smooth_flag = True
    # [date_cube, names, subject_ids, mean_data] = get_gravity_data_paired(all_features, normalize_flag, smooth_flag, hand_to_extract,
    #                                                              t_samples)
    [data_cube_r, mean_data_r, data_cube_l, mean_data_l, all_subject_ids] = get_gravity_data_paired(all_features,
                                                                                                    normalize_flag,
                                                                                                    smooth_flag,
                                                                                                    t_samples)

    labels = get_labels_using_gravity_ratio(mean_data_r, all_subject_ids)  # , all_features, hand, t_samples)
    s_range = np.arange(np.shape(data_cube_r)[0])
    n_time = np.shape(data_cube_r)[2]
    n_ROIs = len(all_features)
    dim_ratio_for_smooth = [1, 0]
    dim_ratio = [0.5, 0.5]
    alpha = 500
    dist_method = 'cityblock'  # 'euclidean'
    loop_ind = 2
    loopRange = range(100)
    run_both_hands_flag = False
    if run_both_hands_flag:
        for loop_ind in loopRange:
            print(loop_ind / 100)
            # '''
            save_prefix = str(0)

            save_prefix = str(loop_ind)
            # '''
            # Split data into train and test
            [data_train_r, data_test_r, label_train, label_test] = train_test_split(data_cube_r, labels, test_size=0.2,
                                                                                    random_state=loop_ind)
            '''
            # Split data manually
            save_prefix = ''
            test_ind = loop_ind
            train_inds = s_range[np.arange(len(s_range)) != test_ind]
        
            data_train_r = data_cube_r[train_inds, :, :]
            label_train = labels.iloc[train_inds, :]
            data_test_r = data_cube_r[test_ind, :, :].reshape(1, np.shape(data_cube_r)[1], np.shape(data_cube_r)[2])
            label_test = labels.iloc[[test_ind]]
            '''

            # Get left hand for train subjects
            data_train_left = np.empty((np.shape(data_train_r)[0], len(all_features), t_samples))
            data_train_mean_left = np.empty((np.shape(data_train_r)[0], t_samples))
            data_train_mean_right = np.empty((np.shape(data_train_r)[0], t_samples))
            for l_ind, label_val in enumerate(label_train.index):
                match_lables_ids = np.where((labels.index == label_val))
                data_train_left[l_ind, :, :] = data_cube_l[match_lables_ids, :, :]
                data_train_mean_left[l_ind, :] = mean_data_l[match_lables_ids, :]
                data_train_mean_right[l_ind, :] = mean_data_r[match_lables_ids, :]

            # Get left hand for test subjects
            data_test_left = np.empty((np.shape(data_test_r)[0], len(all_features), t_samples))
            data_test_mean_left = np.empty((np.shape(data_test_r)[0], t_samples))
            data_test_mean_right = np.empty((np.shape(data_train_r)[0], t_samples))
            for l_ind, label_val in enumerate(label_test.index):
                match_lables_ids = np.where((labels.index == label_val))
                data_test_left[l_ind, :, :] = data_cube_l[match_lables_ids, :, :]
                data_test_mean_left[l_ind, :] = mean_data_l[match_lables_ids, :]
                data_test_mean_right[l_ind, :] = mean_data_r[match_lables_ids, :]

            # Run the model on the train data- given the right hand and the match left hand
            ALP_model = run_ALP_model_train(data_train_r, data_train_left, label_train, 10, alpha,
                                            dist_method, dim_ratio)

            min_ind = np.argmin(ALP_model['root_error_itter'])

            end_model, complete_end_model = run_ALP_model_test_iterations(ALP_model, data_test_r,
                                                                          data_test_left,
                                                                          data_train_left, min_ind)

            predicted_mean = convergence_ROIs(end_model)
            data_test_mean_left = convergence_ROIs(data_test_left)
            data_test_mean_r = convergence_ROIs(data_test_r)
            for s_ind in range(np.shape(end_model)[0]):
                plot_subject_lines(label_test.index[s_ind], real_right_hand=data_test_mean_r[s_ind, :],
                                   real_left_hand=data_test_mean_left[s_ind, :],
                                   predicted_left_hand=predicted_mean[s_ind, :])
                plt.savefig(save_folder + label_test.index[s_ind] + str(loop_ind) + '.png')

            plt.close('all')

    else:
        dim_ratio_for_smooth = [1, 0]
        dim_ratio = [0.5, 0.5]
        alpha = 5

        per_time_ave = np.zeros((s_range[-1] + 1, n_time))
        per_roi_ave = np.zeros((s_range[-1] + 1, n_ROIs))
        per_rand_time_ave = np.zeros((s_range[-1] + 1, n_time))
        per_rand_roi_ave = np.zeros((s_range[-1] + 1, n_ROIs))

        test_pred_ave_diff_all = np.zeros((s_range[-1] + 1, 1))
        test_pred_ave_diff_rand_all = np.zeros((s_range[-1] + 1, s_range[-1]))
        # The following code is the way we wanted to predict the hand of single subject based on his own hand.

        # Smooth the data and predict for every subject separately
        for s_ind, s_name in enumerate(all_subject_ids):
            # Use the investigated subject right hand to predict the laft hand
            train_hand = np.reshape(data_cube_r[s_ind, :, :], [n_ROIs, n_time], order='F').T
            pred_hand = np.reshape(data_cube_l[s_ind, :, :], [n_ROIs, n_time], order='F').T
            ALP_model_smooth = run_ALP_model_train(train_hand, train_hand, labels, 10, alpha,
                                                   dist_method, dim_ratio_for_smooth)

            min_ind = np.argmax([0, np.argmin(ALP_model_smooth['root_error_itter']) - 1])

            train_hand_smooth = ALP_model_smooth['end_multiscale'][min_ind]

            # plot_lines(train_hand=train_hand.mean(axis=1), smooth_hand_1=train_hand_smooth.mean(axis=1),
            #           smooth_hand_2=ALP_model_smooth['end_multiscale'][1].mean(axis=1))
            # plot_hand_before_after(train_hand, train_hand_smooth, 'original', 'smooth')

            ALP_model = run_ALP_model_train(train_hand_smooth, pred_hand, labels, 10, alpha,
                                            dist_method, dim_ratio)

            min_ind = np.argmin(ALP_model['root_error_itter'])

            pred_hand_model = ALP_model['end_multiscale'][min_ind]

            per_time_ave[s_ind, :] = np.mean(pred_hand_model, axis=1)
            per_roi_ave[s_ind, :] = np.mean(pred_hand_model, axis=0)

            train_mean = mean_data_r[s_ind, :]
            test_mean = mean_data_l[s_ind, :]

            test_pred_ave_diff = np.mean(test_mean - per_time_ave[s_ind, :])
            # use to plot line for just one subject
            #plot_lines(real_right_hand=train_hand.mean(axis=1), smooth_right_hand=train_hand_smooth.mean(axis=1),
            #           real_left_hand=pred_hand.mean(axis=1), predicted_left_hand=pred_hand_model.mean(axis=1))

            # use to plot subjecy prediction in comparison with prediction of other subjects
            fig = plt.figure(figsize=[10, 8], constrained_layout=True)
            fig_ax = plt.subplot2grid((3, 4), (0, 0), colspan=3)

            #gs = fig.add_gridspec(3, 4)
            #fig_ax = fig.add_subplot(gs[0, :])
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()

            plot_lines_subplots(fig=fig_ax, name=s_name, sreal_right_hand=train_hand.mean(axis=1),
                                smooth_right_hand=train_hand_smooth.mean(axis=1),
                                real_left_hand=pred_hand.mean(axis=1), predicted_left_hand=pred_hand_model.mean(axis=1))
            fig_ax.text(0.2, 0.85, 'Ave_diff_in_pred = ' + str("{:.5f}".format(test_pred_ave_diff)))
            # PERFORM PERMUTATION BASED ON ALL OTHER SUBJECT AND COPARE WHEN USING THE SAME SUBJECT HAND
            random_ids = [x for x in all_subject_ids if x != s_name]
            ranodom_to_plot = random.sample(random_ids, 8)
            ax_ind = 0
            test_pred_ave_diff_rand = []
            for s_perm_ind_t, s_perm_name in enumerate(random_ids):
                # Use the investigated subject right hand to predict the laft hand
                s_perm_ind = [idx for idx, s in enumerate(all_subject_ids) if s_perm_name in s][0]
                train_rand_hand = np.reshape(data_cube_r[s_perm_ind, :, :], [n_ROIs, n_time], order='F').T
                # pred_rand_hand = np.reshape(data_cube_l[s_perm_ind, :, :], [n_ROIs, n_time], order='F').T
                ALP_model_smooth = run_ALP_model_train(train_rand_hand, train_rand_hand, labels, 10, alpha,
                                                       dist_method, dim_ratio_for_smooth)

                min_ind = np.argmax([0, np.argmin(ALP_model_smooth['root_error_itter']) - 1])

                rand_train_hand_smooth = ALP_model_smooth['end_multiscale'][min_ind]

                ALP_model = run_ALP_model_train(rand_train_hand_smooth, pred_hand, labels, 10, alpha,
                                                dist_method, dim_ratio)

                min_ind = np.argmin(ALP_model['root_error_itter'])

                pred_rand_hand_model = ALP_model['end_multiscale'][min_ind]

                per_rand_time_ave[s_perm_ind_t, :] = np.mean(pred_rand_hand_model, axis=1)
                per_rand_roi_ave[s_perm_ind_t, :] = np.mean(pred_rand_hand_model, axis=0)

                train_rand_mean = mean_data_r[s_perm_ind, :]
                test_rand_mean = mean_data_l[s_perm_ind, :]
                test_pred_ave_diff_rand.append(np.mean(test_rand_mean - per_rand_time_ave[s_perm_ind, :]))

                if s_perm_name in ranodom_to_plot:
                    #plot suplots of random subjects
                    ax_x = np.remainder(ax_ind, 2)+1
                    ax_y = int(np.floor(np.divide(ax_ind, 2)))
                    ax_ind += 1
                    fig_ax = plt.subplot2grid((3, 4), (ax_x, ax_y))
                    #fig_ax1 = fig.add_subplot(gs[ax_x, ax_y])
                    plot_lines_subplots(fig=fig_ax, name=s_perm_name,
                                        real_right_hand=train_rand_hand.mean(axis=1),
                                        smooth_right_hand=rand_train_hand_smooth.mean(axis=1),
                                        real_left_hand=pred_hand.mean(axis=1),
                                        predicted_left_hand=pred_rand_hand_model.mean(axis=1))
                    fig_ax.get_legend().remove()
                    fig_ax.text(0.2, 0.85, str("{:.5f}".format(test_pred_ave_diff_rand[-1])))
            # plot_hand_before_after(pred_hand, pred_hand_model, 'original', 'predicted')
            test_pred_ave_diff_rand_all[s_ind, :] = test_pred_ave_diff_rand
            test_pred_ave_diff_all[s_ind, 0] = test_pred_ave_diff

            #plt.savefig(save_folder + s_name + '.png')
            plt.close('all')

        '''
        path = save_folder + 'diffs_comparison.xlsx'
        writer = pd.ExcelWriter(path, engine='openpyxl')
        pd.DataFrame(test_pred_ave_diff_rand_all).to_excel(writer, sheet_name='rand')
        pd.DataFrame(test_pred_ave_diff_all).to_excel(writer, sheet_name='self')
        writer.save()
        '''
        per_time_ave_df = pd.DataFrame(data=per_time_ave)
        per_time_ave_df.index = all_subject_ids


        # Run PCA- per time ave dist
        [principal_components_Df, principal_components] = run_pca_on_df(per_time_ave_df, 3)
        principal_components_Df.index = all_subject_ids
        # Plot the visualization of the two PCs
        charectaristcs_PD = pd.read_excel(r"G:\My Drive\Thesis\Data\Characteristics.xlsx")
        charectaristcs_PD = charectaristcs_PD.set_index('Ext_name_short')
        exData = charectaristcs_PD.loc[:, [col for col in charectaristcs_PD.columns if 'PP' in col]]
        minVal = min(exData.values) * 0.8
        maxVal = max(exData.values)
        r = 100 * (exData.values - minVal) / (maxVal - minVal)
        rDF = pd.DataFrame(data=r, index=exData.index)

        plotPCA.plotPCA2D(principal_components_Df, all_subject_ids, '', exData)

        # Run tSNE
        perplexity_v = 5
        learning_rate_v = 150
        rCol = []
        for ind in range(len(all_subject_ids)):
            rCol.append('C' + str(ind))

        fashion_tsne = TSNE(n_components=2, perplexity=perplexity_v, early_exaggeration=12.0,
                            learning_rate=learning_rate_v, n_iter=1000, n_iter_without_progress=300,
                            min_grad_norm=1e-07, metric='euclidean', init='random', verbose=0,
                            random_state=None, method='barnes_hut', angle=0.5).fit_transform(per_time_ave_df.values)

        fig, ax = plt.subplots()
        for i, s in enumerate(all_subject_ids):
            xa = fashion_tsne[i, 0]
            ya = fashion_tsne[i, 1]

            relevantR = rDF[[s in t for t in rDF.index]]

            ax.scatter(xa, ya, c=rCol[i], linewidths=0.5, edgecolors='r', s=relevantR.values * 6,
                       alpha=0.5)  # marker=markers[i],
            plt.text(xa, ya, s.split('_')[1], horizontalalignment='center', verticalalignment='center')

    x=1


