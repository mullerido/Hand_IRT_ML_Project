import numpy as np
from sklearn.model_selection import train_test_split
from common.utils import get_labels_using_gravity_ratio, seperate_subjects_by_reaction
from dimensio_reduction.DLP.DLP_Utils import convergence_ROIs
from dimensio_reduction.DLP.import_data_utils import get_gravity_data, get_gravity_data_paired, save_all_gravity_data,\
    get_relevant_data_from_file
from dimensio_reduction.DLP.run_DLP_model import run_ALP_model_train, run_ALP_model_test_iterations
from dimensio_reduction.DLP.DLP_plot import plot_hand_before_after, plot_lines, plot_subject_lines, plot_lines_subplots
import matplotlib.pyplot as plt
import pandas as pd
from dimensio_reduction.ApplyPCA import run_pca_on_df, split_data_in_pca_dim
from dimensio_reduction import plotPCA
from dimensio_reduction.plotUtils import plot_subplots_bars, plot_areas_by_groups_pca_dim_single_fid
import random
from sklearn.manifold import TSNE
import json
from json import JSONEncoder
from dimensio_reduction.Apply_tSNE import vizualise_tsne_and_get_distance, run_tSNE_per_input
from sklearn.decomposition import PCA  # Use to perform the PCA transform


if __name__ == "__main__":

    #save_all_gravity_data(r'C:\Projects\Hand_IRT_Auto_Ecxtraction\Feature-Analysis- matlab\Not_Normalized_smooth_data.json',
    #                      normalize_flag=False, smooth_flag=True, t_samples=39)

    t_samples = 19
    all_dist_method = [
        'euclidean']  # , 'correlation', 'cosine', 'minkowski']  # 'minkowski'  # 'cosine'#'euclidean'/'correlation'
    per_patient_result = []
    relevant_features = ['Thumbs_dist_Intence', 'Ring_dist_Intence', 'Middle_dist_Intence', 'Index_dist_Intence',
                    'Pinky_dist_Intence', 'Palm_Center_Intence']
    save_folder = r'G:\My Drive\Thesis\Project\Results\Two-D Laplacian\temp\\'
    import_file = r'C:\Projects\Hand_IRT_Auto_Ecxtraction\Feature-Analysis- matlab\Normalized_smooth_data.json'
    [data_cube_r, mean_data_r, data_cube_l, mean_data_l, all_subject_ids] = get_relevant_data_from_file(import_file,
                                                                                                        t_samples, relevant_features)

    #[data_cube_r, mean_data_r, data_cube_l, mean_data_l, all_subject_ids] = get_gravity_data_paired(relevant_features,
    #                                                                                                True,
    #                                                                                                True,
    #                                                                                                t_samples)

    labels = get_labels_using_gravity_ratio(mean_data_r, all_subject_ids, False)  # , relevant_features, hand, t_samples)
    s_range = np.arange(np.shape(data_cube_r)[0])
    n_time = np.shape(data_cube_r)[2]
    n_ROIs = len(relevant_features)
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
            data_train_left = np.empty((np.shape(data_train_r)[0], len(relevant_features), t_samples))
            data_train_mean_left = np.empty((np.shape(data_train_r)[0], t_samples))
            data_train_mean_right = np.empty((np.shape(data_train_r)[0], t_samples))
            for l_ind, label_val in enumerate(label_train.index):
                match_lables_ids = np.where((labels.index == label_val))
                data_train_left[l_ind, :, :] = data_cube_l[match_lables_ids, :, :]
                data_train_mean_left[l_ind, :] = mean_data_l[match_lables_ids, :]
                data_train_mean_right[l_ind, :] = mean_data_r[match_lables_ids, :]

            # Get left hand for test subjects
            data_test_left = np.empty((np.shape(data_test_r)[0], len(relevant_features), t_samples))
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
        train_time_ave = np.zeros((s_range[-1] + 1, n_time))
        train_roi_ave = np.zeros((s_range[-1] + 1, n_ROIs))

        test_pred_diff_times = np.zeros((s_range[-1] + 1, n_time))
        test_pred_diff_roi = np.zeros((s_range[-1] + 1, n_ROIs))
        test_pred_ave_diff_all = np.zeros((s_range[-1] + 1, 1))
        test_pred_ave_diff_rand_all = np.zeros((s_range[-1] + 1, s_range[-1]))
        # The following code is the way we wanted to predict the hand of single subject based on his own hand.

        # Smooth the data and predict for every subject separately
        for s_ind, s_name in enumerate(all_subject_ids):
            # Use the investigated subject right hand to predict the laft hand
            train_hand = np.reshape(data_cube_r[s_ind, :, :], [n_ROIs, n_time], order='F').T
            pred_hand = np.reshape(data_cube_l[s_ind, :, :], [n_ROIs, n_time], order='F').T

            # Create data fame n=times with the s_name as index and the label #
            s_df = pd.DataFrame(columns=['label'])
            for t_s in range(np.shape(train_hand)[0]):
                s_df = s_df.append(pd.DataFrame(data=[labels.loc[s_name, 'label']], columns=['label']))
                s_df = s_df.rename(index = {0: s_name})

            ALP_model_smooth = run_ALP_model_train(train_hand, train_hand, s_df, 10, alpha,
                                                   dist_method, dim_ratio_for_smooth)

            min_ind = np.argmax([0, np.argmin(ALP_model_smooth['root_error_itter']) - 1])

            train_hand_smooth = ALP_model_smooth['end_multiscale'][min_ind]

            train_time_ave[s_ind, :] = np.mean(train_hand_smooth, axis=1)
            train_roi_ave[s_ind, :] = np.mean(train_hand_smooth, axis=0)
            # plot_lines(train_hand=train_hand.mean(axis=1), smooth_hand_1=train_hand_smooth.mean(axis=1),
            #           smooth_hand_2=ALP_model_smooth['end_multiscale'][1].mean(axis=1))
            # plot_hand_before_after(train_hand, train_hand_smooth, 'original', 'smooth')

            ALP_model = run_ALP_model_train(train_hand_smooth, pred_hand, s_df, 10, alpha,
                                            dist_method, dim_ratio)

            min_ind = np.argmin(ALP_model['root_error_itter'])

            pred_hand_model = ALP_model['end_multiscale'][min_ind]

            per_time_ave[s_ind, :] = np.mean(pred_hand_model, axis=1)
            per_roi_ave[s_ind, :] = np.mean(pred_hand_model, axis=0)

            train_mean = mean_data_r[s_ind, :]
            test_mean = mean_data_l[s_ind, :]

            test_mean_time = np.mean(pred_hand, axis=1)
            test_mean_roi = np.mean(pred_hand, axis=0)

            test_pred_diff_times[s_ind, :] = test_mean_time - per_time_ave[s_ind, :]
            test_pred_diff_roi[s_ind, :] = test_mean_roi - per_roi_ave[s_ind, :]

            test_pred_ave_diff = np.mean(test_mean - per_time_ave[s_ind, :])
            # use to plot line for just one subject
            # plot_lines(real_right_hand=train_hand.mean(axis=1), smooth_right_hand=train_hand_smooth.mean(axis=1),
            #           real_left_hand=pred_hand.mean(axis=1), predicted_left_hand=pred_hand_model.mean(axis=1))

            # use to plot subjecy prediction in comparison with prediction of other subjects
            fig = plt.figure(figsize=[10, 8], constrained_layout=True)
            fig_ax = plt.subplot2grid((3, 4), (0, 0), colspan=3)

            # gs = fig.add_gridspec(3, 4)
            # fig_ax = fig.add_subplot(gs[0, :])
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()

            plot_lines_subplots(fig=fig_ax, name=s_name, sreal_right_hand=train_hand.mean(axis=1),
                                smooth_right_hand=train_hand_smooth.mean(axis=1),
                                real_left_hand=pred_hand.mean(axis=1), predicted_left_hand=pred_hand_model.mean(axis=1))
            fig_ax.text(0.2, 0.85, 'Ave_diff_in_pred = ' + str("{:.5f}".format(test_pred_ave_diff)))
            # PERFORM PERMUTATION BASED ON ALL OTHER SUBJECT AND COMPARE WHEN USING THE SAME SUBJECT HAND
            random_ids = [x for x in all_subject_ids if x != s_name]
            ranodom_to_plot = random.sample(random_ids, 8)
            ax_ind = 0
            test_pred_ave_diff_rand = []

        '''
            for s_perm_ind_t, s_perm_name in enumerate(random_ids):
                # Use the investigated subject right hand to predict the laft hand
                s_perm_ind = [idx for idx, s in enumerate(all_subject_ids) if s_perm_name in s][0]
                train_rand_hand = np.reshape(data_cube_r[s_perm_ind, :, :], [n_ROIs, n_time], order='F').T
                # pred_rand_hand = np.reshape(data_cube_l[s_perm_ind, :, :], [n_ROIs, n_time], order='F').T
                # Create data fame n=times with the s_name as index and the label #
                s_df = pd.DataFrame(columns=['label'])
                for t_s in range(np.shape(train_rand_hand)[0]):
                    s_df = s_df.append(pd.DataFrame(data=[labels.loc[s_perm_name, 'label']], columns=['label']))
                    s_df = s_df.rename(index={0: s_perm_name})

                ALP_model_smooth = run_ALP_model_train(train_rand_hand, train_rand_hand, s_df, 10, alpha,
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

                # Plot comarison in sub-plot for random subject
                if s_perm_name in ranodom_to_plot:
                    # plot suplots of random subjects
                    ax_x = np.remainder(ax_ind, 2) + 1
                    ax_y = int(np.floor(np.divide(ax_ind, 2)))
                    ax_ind += 1
                    fig_ax = plt.subplot2grid((3, 4), (ax_x, ax_y))
                    # fig_ax1 = fig.add_subplot(gs[ax_x, ax_y])
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

            plt.savefig(save_folder + s_name + '.png')
            plt.close('all')
            

        path = save_folder + 'diffs_comparison.xlsx'
        writer = pd.ExcelWriter(path, engine='openpyxl')
        pd.DataFrame(test_pred_ave_diff_rand_all).to_excel(writer, sheet_name='rand')
        pd.DataFrame(test_pred_ave_diff_all).to_excel(writer, sheet_name='self')
        writer.save()

        '''
        '''
        # Analysis error in times
        per_time_ave_df = pd.DataFrame(data=test_pred_diff_times)  #per_time_ave
        per_time_ave_df.index = all_subject_ids

        # Run PCA- per time ave dist
        [principal_components_Df, principal_components, eigen_vectors, eigen_values] = run_pca_on_df(per_time_ave_df, 2)
        principal_components_Df.index = all_subject_ids
        # Plot the visualization of the two PCs
        charectaristcs_PD = pd.read_excel(r"G:\My Drive\Thesis\Data\Characteristics.xlsx")
        charectaristcs_PD = charectaristcs_PD.set_index('Ext_name_short')
        exData = charectaristcs_PD.loc[:, [col for col in charectaristcs_PD.columns if 'PP' in col]]
        minVal = min(exData.values) * 0.8
        maxVal = max(exData.values)
        r = 100 * (exData.values - minVal) / (maxVal - minVal)
        rDF = pd.DataFrame(data=r, index=exData.index)

        save_path = save_folder + '2-D PCA_time_ave' + '.png'
        title = 'PCA Time Ave'
        #plotPCA.plotPCA2D(principal_components_Df, all_subject_ids, '', exData, title, save_path)
        #save_path = save_folder + '3-D PCA_roi_ave' + '.png'
        #plotPCA.plotPCA3D(principal_components_Df, all_subject_ids, '', exData, title, save_path)

        plot_subplots_bars(eigen_vectors, title='Feature coefficients', x_label='Feature number', y_label='Coefficient value')
        x = np.linspace(0, np.shape(eigen_vectors)[1]-1, np.shape(eigen_vectors)[1])
        barWidth = 0.5
        fig, ax = plt.subplots(np.shape(eigen_vectors)[0], 1, figsize=(10, 8))
        ax[0].title.set_text('Time errors coefficients for the first PC')
        ax[1].title.set_text('Time errors coefficients for the Second PC')

        for i in range(np.shape(eigen_vectors)[0]):
            ax[i].bar(x, eigen_vectors[i, :], width=barWidth, color=(0.1, 0.1, 0.1, 0.1), edgecolor='blue')
            ax[i].plot([-0.5, x[-1]+0.5], [0, 0], color=(0.9, 0.9, 0.9), linewidth=1)
            ax[i].set_ylabel('Coefficient value')
        ax[1].set_xlabel('Time index')

        [groups, Legend] = split_data_in_pca_dim(principal_components, all_subject_ids, type='rectangle', x1_thresh=-0.03, y1_thresh=-0.03, x2_thresh=0.014,
                              y2_thresh=0.015)

        mean_data_r_df = pd.DataFrame(data=mean_data_r)  #per_time_ave
        mean_data_r_df.index = all_subject_ids
        plot_areas_by_groups_pca_dim_single_fid(mean_data_r_df, all_subject_ids, groups, Legend, True)

        mean_data_l_df = pd.DataFrame(data=mean_data_l)  # per_time_ave
        mean_data_l_df.index = all_subject_ids
        plot_areas_by_groups_pca_dim_single_fid(mean_data_l_df, all_subject_ids, groups, Legend, True)
        '''
#################################################################################
        # Analysis error in times
        per_roi_ave_df = pd.DataFrame(data=test_pred_diff_roi)  # per_time_ave
        per_roi_ave_df.index = all_subject_ids

        # Run PCA- per time ave dist
        [principal_components_Df, principal_components, eigen_vectors, eigen_values] = run_pca_on_df(per_roi_ave_df, 2)
        principal_components_Df.index = all_subject_ids

        save_path = save_folder + '2-D PCA_roi_ave' + '.png'
        title = 'PCA ROI Ave'
        plotPCA.plotPCA2D(principal_components_Df, all_subject_ids, '', exData, title, save_path)
        # save_path = save_folder + '3-D PCA_roi_ave' + '.png'
        # plotPCA.plotPCA3D(principal_components_Df, all_subject_ids, '', exData, title, save_path)

        plot_subplots_bars(eigen_vectors, title='Feature coefficients', x_label='Feature number',
                           y_label='Coefficient value')
        x = np.linspace(0, np.shape(eigen_vectors)[1] - 1, np.shape(eigen_vectors)[1])
        barWidth = 0.5
        fig, ax = plt.subplots(np.shape(eigen_vectors)[0], 1, figsize=(10, 8))
        ax[0].title.set_text('ROI errors coefficients for the first PC')
        ax[1].title.set_text('ROI errors coefficients for the Second PC')

        for i in range(np.shape(eigen_vectors)[0]):
            ax[i].bar(x, eigen_vectors[i, :], width=barWidth, color=(0.1, 0.1, 0.1, 0.1), edgecolor='blue')
            ax[i].plot([-0.5, x[-1] + 0.5], [0, 0], color=(0.9, 0.9, 0.9), linewidth=1)
            ax[i].set_ylabel('Coefficient value')
        ax[1].set_xlabel('Time index')

        [groups, Legend] = split_data_in_pca_dim(principal_components, all_subject_ids, type='rectangle',
                                                 x1_thresh=-0.03, y1_thresh=-0.02, x2_thresh=0.02,
                                                 y2_thresh=0.02)

        mean_data_r_df = pd.DataFrame(data=mean_data_r)  # per_time_ave
        mean_data_r_df.index = all_subject_ids
        plot_areas_by_groups_pca_dim_single_fid(mean_data_r_df, all_subject_ids, groups, Legend, True)

        mean_data_l_df = pd.DataFrame(data=mean_data_l)  # per_time_ave
        mean_data_l_df.index = all_subject_ids
        plot_areas_by_groups_pca_dim_single_fid(mean_data_l_df, all_subject_ids, groups, Legend, True)

########################################################################
        # Run tSNE
        perplexity_v = 5
        learning_rate_v = 150
        rCol = []
        for ind in range(len(all_subject_ids)):
            rCol.append('C' + str(ind))

        fashion_tsne = TSNE(n_components=3, perplexity=perplexity_v, early_exaggeration=12.0,
                            learning_rate=learning_rate_v, n_iter=1000, n_iter_without_progress=300,
                            min_grad_norm=1e-07, metric='euclidean', init='random', verbose=0,
                            random_state=None, method='barnes_hut', angle=0.5).fit_transform(per_time_ave_df.values)

        #fig, ax = plt.subplots()
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        #ax = plt.axes(projection='3d')
        for i, s in enumerate(all_subject_ids):
            xa = fashion_tsne[i, 0]
            ya = fashion_tsne[i, 1]
            za = fashion_tsne[i, 2]
            relevantR = rDF.loc[s, 0]
            ax.scatter(xa, ya, za, c=rCol[i], linewidth=0.5, s=relevantR*5, alpha= 0.6)

            ax.text(xa, ya, za, s.split('_')[1])#, horizontalalignment='center', verticalalignment='center')
        ax.title.set_text('tSNE- roi Ave')
        save_path = save_folder + '3D tSNE_roi_ave' + '.png'
        plt.savefig(save_folder)


    x = 1
