import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import seaborn as sns
from dimensio_reduction.DLP.DLP_Utils import convergence_ROIs


def plot_comparison_per_subject_per_ROI(end_model, start_data, data_test_end_true_cube, ROI_S, label_test, n_start,
                                        save_folder, save_prefix=''):
    for s_ind in range(np.shape(end_model)[0]):
        fig, ax = plt.subplots(ROI_S)
        fig.suptitle('Test Group Prediction Comparison', fontsize=20)

        for f_ind in range(ROI_S):
            trueData = np.concatenate((start_data[s_ind, f_ind, :], data_test_end_true_cube[s_ind, f_ind, :]))
            predictedData = np.concatenate((start_data[s_ind, f_ind, :], end_model[s_ind, f_ind, :]))

            ax[f_ind].plot(trueData, color='forestgreen')
            ax[f_ind].plot(predictedData, color='indianred')
            ax[f_ind].plot(start_data[s_ind, f_ind, :])
            y_lim = ax[f_ind].get_ylim()
            ax[f_ind].plot([n_start - 1, n_start - 1], y_lim, linestyle='--', linewidth=0.8, color='lightgray')

        fig.legend(['True', 'Predict', 'Start'], loc='lower right')
        plt.savefig(save_folder + 'ROI sub-plots\\' + save_prefix + '_' + label_test.index[s_ind] + '.png')


def plot_test_vs_group(end_model, data_train, data_test, data_test_start, n_start, labels, label_test, label_train,
                       save_folder, save_prefix=''):
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
        plt.suptitle('Label Grouped Vs. Prediction- ' + label_test.index[s_ind], fontsize=20)
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

        ax.plot(x, s_real_test, linestyle='-', linewidth=1, color='cornflowerblue')
        ax.plot(x, s_model_test, linestyle='--', linewidth=1.2, color='royalblue')
        ax.legend(['Real', 'Predict', 'group'], loc='lower right')
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        ax.plot([n_start - 1, n_start - 1], y_lim, linestyle='--', linewidth=0.8, color='lightgray')
        plt.text(np.mean(x_lim) * 0.7, y_lim[1] * 0.9, 'group= ' + str(label_test.iloc[s_ind, 0]), fontsize=16,
                 verticalalignment='top')

        plt.savefig(save_folder + 'Label Grouped Comparison\\' + save_prefix + '_' + label_test.index[s_ind] + '.png')


def plot_test_vs_k_nearest(end_model, data_train, data_test, label_test, n_start, save_folder, save_prefix=''):
    # Average ROIs per patient- from cube 2 2D
    data_train_2D = convergence_ROIs(data_train)
    data_test_2D = convergence_ROIs(data_test)
    # if np.ndim(data_test) == 3:
    end_model_2D = convergence_ROIs(end_model)
    # elif np.ndim(data_test) == 2:
    #    end_model_2D = convergence_ROIs(end_model.reshape((5, 13), order='F'))

    x = np.linspace(0, np.shape(data_test_2D)[1] - 1, np.shape(data_test_2D)[1])
    for plt_ind in range(np.shape(data_test_2D)[0]):
        dist_to_test = cdist(data_test_2D[plt_ind, :].reshape(1, -1), data_train_2D, 'euclidean')

        minarr = np.argpartition(dist_to_test, 3, axis=1)[0][:3]

        closest_data = data_train_2D[minarr, :].mean(axis=0)

        fig, ax = plt.subplots()
        plt.suptitle('k-nearest Vs. Prediction- ' + label_test.index[plt_ind], fontsize=20)

        ax.plot(x, closest_data, linestyle='-', linewidth=1, color='tomato')
        # ax.plot(x[:6], closest_data[:6], linestyle='-', color='C1')
        # ax.plot(x[5:], closest_data[5:], linestyle='--', color='C1')

        ax.plot(x, data_test_2D[plt_ind, :], linestyle='-', linewidth=1, color='cornflowerblue')
        ax.plot(x[5:], np.append(data_test_2D[plt_ind, 5], end_model_2D[plt_ind, :]), linestyle='--', linewidth=1.2,
                color='royalblue')
        fig.legend(['k-nearest', 'test-true', 'test-predict'], loc='lower right')
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        ax.plot([n_start - 1, n_start - 1], y_lim, linestyle='--', linewidth=0.8, color='lightgray')
        plt.text(np.mean(x_lim) * 0.7, y_lim[1] * 0.9, 'group= ' + str(label_test.iloc[plt_ind, 0]), fontsize=16,
                 verticalalignment='top')

        plt.savefig(save_folder + r'k nearest comparison\\' + save_prefix + '_' + label_test.index[plt_ind] + '.png')
        # ax.plot(x[:6], data_test_2D[plt_ind, :6], linestyle='-', color='C0')
        # ax.plot(x[5:], data_test_2D[plt_ind, 5:], linestyle='-', color='C0')


def plot_test_progress_vs_k_nearest(end_model_plot, data_test, closest_data, label_test, n_start, min_ind,
                                    save_folder, save_prefix, plot_range):
    # Average ROIs per patient- from cube 2 2D
    data_test_2D = convergence_ROIs(data_test)
    x = np.linspace(0, np.shape(data_test_2D)[1] - 1, np.shape(data_test_2D)[1])
    colors = ['CadetBlue', 'SteelBlue', 'LightBlue', 'DeepSkyBlue', 'cornflowerblue', 'midnightblue', 'mediumblue',
              'blue', 'slateblue', 'DarkSlateBlue', ]
    for plt_ind in range(np.shape(data_test_2D)[0]):
        current_end_model = end_model_plot['s_model'][plt_ind]

        # if np.ndim(data_test) == 3:
        # elif np.ndim(data_test) == 2:
        #    end_model_2D = convergence_ROIs(end_model.reshape((5, 13), order='F'))
        fig, ax = plt.subplots()
        plt.suptitle('k-nearest Vs. Prediction- ' + label_test.index[plt_ind], fontsize=20)
        ax.plot(x, closest_data[plt_ind, :], linestyle='-', linewidth=1, color='tomato')
        ax.plot(x, data_test_2D[plt_ind, :], linestyle='-', linewidth=1, color='cornflowerblue')
        fig_legend = ['k-nearest', 'test-true']
        for model_in in plot_range:
            end_model_2D = convergence_ROIs(current_end_model['multiscale'][model_in])

            if model_in == min_ind:
                line_style = '-.'
                c = 'ForestGreen'
            else:
                line_style = ':'
                c = colors[model_in]

            ax.plot(x[5:], np.append(data_test_2D[plt_ind, 5], end_model_2D), linestyle=line_style, linewidth=1.2,
                    color=c)
            fig_legend.append('k_ind-' + str(model_in))

        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        ax.plot([n_start - 1, n_start - 1], y_lim, linestyle='--', linewidth=0.8, color='lightgray')
        plt.text(np.mean(x_lim) * 0.7, y_lim[1] * 0.9, 'group= ' + str(label_test.iloc[plt_ind, 0]), fontsize=16,
                 verticalalignment='top')
        fig.legend(fig_legend, loc='lower right')
        plt.savefig(save_folder + r'k nearest comparison\\' + save_prefix + '_' + label_test.index[plt_ind] + '.png')


def plot_hand_before_after(before, after, before_title='before', after_title='after'):
    fig, ax = plt.subplots(1, 2)
    x = np.linspace(0, np.shape(before)[0] - 1, np.shape(before)[0])
    plt.suptitle('Before Vs. After', fontsize=20)
    n_roi = np.shape(before)[1]
    rgb_values = sns.color_palette("muted", n_roi)

    for roi in range(n_roi):
        ax[0].plot(x, before[:, roi], linestyle='-', linewidth=0.8, color=rgb_values[roi])
        ax[1].plot(x, after[:, roi], linestyle='-', linewidth=0.8, color=rgb_values[roi])
    y_lim = [np.min([ax[0].get_ylim()[0], ax[1].get_ylim()[0]]), np.max([ax[0].get_ylim()[1], ax[1].get_ylim()[1]])]
    ax[0].set_ylim(y_lim)
    ax[1].set_ylim(y_lim)
    ax[0].set_title(before_title)
    ax[1].set_title(after_title)


def plot_lines(**kwargs):
    all_line_styles = ['-', '--', '-.', '-.', ':']
    colors = plt.cm.rainbow(np.linspace(0, 1, 2))
    fig, ax = plt.subplots()
    legend = []
    style = 0
    for s, v in kwargs.items():
        if np.ndim(v) == 1:

            x = np.linspace(0, np.shape(v)[0] - 1, np.shape(v)[0])
            ax.plot(x, v, c=colors[int(np.floor(style / 2)), :], linestyle=all_line_styles[int(np.mod(style, 2))],
                    linewidth=0.8)
            legend.append(s + '_' + str(0))
        elif np.ndim(v) == 2:
            colors = plt.cm.rainbow(np.linspace(0, 1, np.shape(v)[1]))

            for col_i in range(np.shape(v)[1]):
                x = np.linspace(0, np.shape(v)[0] - 1, np.shape(v)[0])
                ax.plot(x, v[:, col_i], c=colors[col_i, :], linestyle=all_line_styles[style], linewidth=0.8)
                legend.append(s + '_' + str(col_i))

        elif np.ndim(v) == 3:
            colors = plt.cm.rainbow(np.linspace(0, 1, np.shape(v)[1]))

            for subject in np.shape(v)[0]:

                current_s = v[subject, :, :]

                for col_i in range(np.shape(v)[1]):
                    x = np.linspace(0, np.shape(v)[0] - 1, np.shape(v)[0])
                    ax.plot(x, v[:, col_i], c=colors[col_i, :], linestyle=all_line_styles[style], linewidth=0.8)
                    legend.append(s + '_' + str(col_i))

        style += 1
    ax.set_title('Comparison')
    ax.set_ylim([-0.3, 1])
    box = ax.get_position()
    ax.set_position([box.x0, 0.2 + box.y0, box.width, box.height * 0.8])
    ax.legend(legend, loc='center', bbox_to_anchor=(0.5, -0.25))

    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.legend(legend, loc='center left', bbox_to_anchor=(0.5, 0.5))


def plot_lines_subplots(**kwargs):
    all_line_styles = ['-', '--', '-.', '-.', ':']
    colors = plt.cm.rainbow(np.linspace(0, 1, 2))
    # fig, ax = plt.subplots()
    legend = []
    style = 0
    for s, v in kwargs.items():

        if s == 'fig':
            ax = v
            continue
        elif s == 'name':
            name = v
            continue
        elif np.ndim(v) == 1:

            x = np.linspace(0, np.shape(v)[0] - 1, np.shape(v)[0])
            ax.plot(x, v, c=colors[int(np.floor(style / 2)), :], linestyle=all_line_styles[int(np.mod(style, 2))],
                    linewidth=0.8)
            legend.append(s)
        elif np.ndim(v) == 2:
            colors = plt.cm.rainbow(np.linspace(0, 1, np.shape(v)[1]))

            for col_i in range(np.shape(v)[1]):
                x = np.linspace(0, np.shape(v)[0] - 1, np.shape(v)[0])
                ax.plot(x, v[:, col_i], c=colors[col_i, :], linestyle=all_line_styles[style], linewidth=0.8)
                legend.append(s + '_' + str(col_i))

        elif np.ndim(v) == 3:
            colors = plt.cm.rainbow(np.linspace(0, 1, np.shape(v)[1]))

            for subject in np.shape(v)[0]:

                current_s = v[subject, :, :]

                for col_i in range(np.shape(v)[1]):
                    x = np.linspace(0, np.shape(v)[0] - 1, np.shape(v)[0])
                    ax.plot(x, v[:, col_i], c=colors[col_i, :], linestyle=all_line_styles[style], linewidth=0.8)
                    legend.append(s + '_' + str(col_i))

        style += 1
    ax.set_title('Comparison- ' + name)
    ax.set_ylim([-0.3, 1])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 1, box.height * 0.8])
    ax.legend(legend, loc='right', ncol=1, bbox_to_anchor=(1.45, 0.5))


def plot_subject_lines(s_name, **kwargs):
    colors = plt.cm.rainbow(np.linspace(0, 1, len(kwargs)))

    fig, ax = plt.subplots()
    legend = []
    style = 0
    for s, v in kwargs.items():
        x = np.linspace(0, np.shape(v)[0] - 1, np.shape(v)[0])
        ax.plot(x, v, c=colors[style, :], linestyle='solid', linewidth=0.8)
        legend.append(s)
        style += 1

    ax.set_title('Comparison-' + s_name)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(legend, loc='center left', bbox_to_anchor=(1, 0.5))
