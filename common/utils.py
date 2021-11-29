import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as PathEffects
from scipy import stats
from scipy.spatial import distance


def get_gravity_data(all_features, normlize_flag=False, hand='', t_samples=19):
    folder = r'C:\Projects\Hand_IRT_Auto_Ecxtraction\Feature-Analysis- matlab\Resize Feature\\'

    files = os.listdir(folder)
    if not hand or hand == 'both':
        files_xls = [f for f in files if f[-5:] == '.xlsx']
    elif hand == 'right':
        files_xls = [f for f in files if f[-10:] == 'right.xlsx']
    elif hand == 'left':
        files_xls = [f for f in files if f[-9:] == 'left.xlsx']

    grouped_feature = pd.DataFrame()
    names = []
    subject_id = []
    data = np.empty((np.shape(files_xls)[0], t_samples))
    for ind, file in enumerate(files_xls):
        print(file)
        currentFile = folder + file
        currentFeatures = pd.read_excel(currentFile)
        if normlize_flag:  # Use the ratio
            relevant_cols = (currentFeatures.loc[0:t_samples - 1, all_features] - currentFeatures.loc[
                0, all_features]) / \
                            currentFeatures.loc[0, all_features]
            data[ind, :] = relevant_cols.mean(axis=1)
        else:  # Use the actual values
            # relevantCols = currentFeatures.filter(regex='Intence')
            relevant_cols = currentFeatures.loc[0:t_samples - 1, all_features]
            data[ind, :] = relevant_cols.mean(
                (currentFeatures.loc[0:t_samples - 1, all_features] - currentFeatures.loc[0, all_features]) /
                currentFeatures.loc[0, all_features], axis=1)

        transformData = relevant_cols.values.ravel('F')
        transformDataFrame = pd.DataFrame(transformData).T
        indName = file[:-5]
        transformDataFrame = transformDataFrame.rename(index={0: indName})
        grouped_feature = grouped_feature.append(transformDataFrame)
        ind_name = file[:-5]
        names.append(ind_name.split('_')[0] + '_' + ind_name.split('_')[1] + '_' + ind_name.split('_')[2][0])
        subject_id.append(ind_name.split('_')[0] + '_' + ind_name.split('_')[1] + '_')
    subject_id = np.unique(subject_id)
    grouped_feature.index = names

    return grouped_feature, names, subject_id, data


def get_study_data(all_features, normalize_flag=False):
    folder = r'C:\Projects\Hand_IRT_Auto_Ecxtraction\Feature-Analysis- matlab\Resize Feature\\'

    files = os.listdir(folder)
    # files_xls = [f for f in files if f[-10:] == 'right.xlsx']
    files_xls = [f for f in files if f[-5:] == '.xlsx']

    groupedFeature = pd.DataFrame()
    names = []
    subject_id = []
    data = np.empty((np.shape(files_xls)[0], 19))
    for ind, file in enumerate(files_xls):
        print(file)
        currentFile = folder + file
        currentFeatures = pd.read_excel(currentFile)
        if normalize_flag:  # Use the ratio
            relevant_cols = (currentFeatures.loc[0:, all_features] - currentFeatures.loc[0, all_features]) / \
                            currentFeatures.loc[0, all_features]
        else:  # Use the actual values
            # relevantCols = currentFeatures.filter(regex='Intence')
            relevant_cols = currentFeatures.loc[0:, all_features]

        data[ind, :] = relevant_cols.mean(axis=1)
        transformData = relevant_cols.values.ravel('F')
        transformDataFrame = pd.DataFrame(transformData).T
        indName = file[:-5]
        transformDataFrame = transformDataFrame.rename(index={0: indName})
        groupedFeature = groupedFeature.append(transformDataFrame)
        ind_name = file[:-5]
        names.append(ind_name.split('_')[0] + '_' + ind_name.split('_')[1] + '_' + ind_name.split('_')[2][0])
        subject_id.append(ind_name.split('_')[0] + '_' + ind_name.split('_')[1] + '_')
    subject_id = np.unique(subject_id)

    return groupedFeature, names, subject_id


def get_hand_fingers_data_relate_to_center(normalize_flag=False, hand=''):
    grouped_feature = pd.DataFrame()
    folder = r'C:\Projects\Hand_IRT_Auto_Ecxtraction\Feature-Analysis- matlab\Resize Feature\\'

    files = os.listdir(folder)
    if not hand or hand == 'both':
        files_xls = [f for f in files if f[-5:] == '.xlsx']
    elif hand == 'right':
        files_xls = [f for f in files if f[-10:] == 'right.xlsx']
    elif hand == 'left':
        files_xls = [f for f in files if f[-9:] == 'left.xlsx']
    subject_id = []
    groupedFeature = pd.DataFrame()
    names = []
    allingers = ['Thumbs_dist_Intence', 'Thumbs_proxy_Intence', 'Index_dist_Intence', 'Index_proxy_Intence',
                 'Middle_dist_Intence', 'Middle_proxy_Intence', 'Ring_dist_Intence', 'Ring_proxy_Intence',
                 'Pinky_dist_Intence', 'Pinky_proxy_Intence']
    allingers = ['Thumbs_dist_Intence', 'Index_dist_Intence',
                 'Middle_dist_Intence', 'Ring_dist_Intence',
                 'Pinky_dist_Intence']

    for file in files_xls:
        print(file)
        currentFile = folder + file
        currentFeatures = pd.read_excel(currentFile)

        # relevantCols = currentFeatures.filter(regex='Intence')
        relevantFingersCols = np.transpose(currentFeatures.loc[:, allingers].values)
        relevantPalmCols = currentFeatures.loc[:, 'Palm_Center_Intence'].values
        subtract_fingers = relevantFingersCols - relevantPalmCols
        # subtract_fingers = pd.DataFrame(subtract_fingers)
        if normalize_flag:
            relevant_cols = np.transpose(
                (np.transpose(subtract_fingers[:, :]) - subtract_fingers[:, 0]) / subtract_fingers[:, 0])

        else:
            relevant_cols = subtract_fingers

        relevant_cols_pd = pd.DataFrame(relevant_cols[:, 0:18])
        transformData = relevant_cols_pd.values.ravel('C')
        transformDataFrame = pd.DataFrame(transformData).T
        ind_name = file[:-5]
        transformDataFrame = transformDataFrame.rename(index={0: ind_name})
        grouped_feature = grouped_feature.append(transformDataFrame)
        names.append(ind_name.split('_')[0] + '_' + ind_name.split('_')[1] + '_' + ind_name.split('_')[2][0])
        subject_id.append(ind_name.split('_')[0] + '_' + ind_name.split('_')[1] + '_')
    subject_id = np.unique(subject_id)

    return grouped_feature, names, subject_id


def seperate_subjects_by_reaction(data, grouped_feature, normalize_flag, subject_id, names, plot_flag=False):
    rCol = []
    for ind in range(len(subject_id)):
        rCol.append('C' + str(ind))

    # Seperate subjects by their reaction
    data_mean = data.mean(axis=1, keepdims=True)
    data_std = data.std(axis=1, keepdims=True)
    data_mean_std = np.array([data_mean[:, 0] - data_std[:, 0], data_mean[:, 0] + data_std[:, 0]]).T

    positive_reaction_ids = np.array([data_mean_std[:, 0] > 0]).T
    negative_reaction_ids = np.array([data_mean_std[:, 1] < 0]).T
    balance_reaction_ids = (positive_reaction_ids == False) & (negative_reaction_ids == False)
    reactions_ids = np.array([negative_reaction_ids[:, 0], balance_reaction_ids[:, 0], positive_reaction_ids[:, 0]]).T
    if len(data_mean_std) > 30:
        positive_reaction_ids = positive_reaction_ids[['r' in s for s in names]]
        negative_reaction_ids = negative_reaction_ids[['r' in s for s in names]]
        balance_reaction_ids = balance_reaction_ids[['r' in s for s in names]]

    # Plot the reaction by groups of reactions
    if plot_flag:
        fig, ax = plt.subplots()
        plt.title('Group Comparison', fontsize=20)
        plt.ylabel('Ratio', fontsize=12)
        plt.xlabel("Index of time", fontsize=12)
        grouped_feature_plot = grouped_feature[['r' in s for s in names]]

        groups = [np.where(negative_reaction_ids), np.where(balance_reaction_ids), np.where(positive_reaction_ids)]
        Legend = ['Negative reaction', 'Balance reaction', 'Possitive reaction']
        for ind, groupInds in enumerate(groups):
            if groupInds:
                tempDF = pd.DataFrame()
                tempDF = grouped_feature_plot.iloc[groupInds[0], :]
                tempDFVal = tempDF.values
                dataByTimes = np.resize(tempDFVal, (tempDFVal.shape[0] * 12, np.shape(data)[1]))

                if not normalize_flag:
                    for i in range(dataByTimes.shape[0]):
                        dataByTimes[i, :] = (dataByTimes[i, :] - dataByTimes[i, 0]) / dataByTimes[i, 0]
                groupMeans = dataByTimes.mean(axis=0)
                groupStd = dataByTimes.std(axis=0)
                ax.plot(groupMeans, color=rCol[ind])
                x = np.linspace(0, len(groupStd) - 1, len(groupStd))
                ax.fill_between(x, groupMeans - groupStd, groupMeans + groupStd, color=rCol[ind], alpha=0.3)

        plt.legend(Legend[0:ind + 1], loc='center left', bbox_to_anchor=(0, 0.9))
        plt.show()

    return reactions_ids


def get_labels_by_hands_similarity(all_features, grouped_feature=[], subject_id=[], names=[]):
    if len(grouped_feature) == 0:
        [grouped_feature, names, subject_id, data] = get_gravity_data(all_features, True)

    plot_flag = False
    labels = seperate_subjects_by_reaction(data, grouped_feature, True, subject_id, names, plot_flag)

    return labels


def get_labels_using_gravity_ratio(data, names, plot_flag=False):  # , all_features, hand='', t_samples=19):
    # Get the result of the gravity phase
    # normalizeFlag = True
    # [groupedFeature, a, b, data] = get_gravity_data(all_features, normlizeFlag, hand, t_samples)

    # Separate subjects by their reaction
    data_mean = data.mean(axis=1, keepdims=True)
    data_std = data.std(axis=1, keepdims=True)
    data_mean_std = np.array([data_mean[:, 0] - data_std[:, 0], data_mean[:, 0] + data_std[:, 0]]).T

    positive_reaction_ids = np.array([data_mean_std[:, 0] > 0]).T
    negative_reaction_ids = np.array([data_mean_std[:, 1] < 0]).T
    balance_reaction_ids = (positive_reaction_ids == False) & (negative_reaction_ids == False)
    reactions_ids = np.array([negative_reaction_ids[:, 0], balance_reaction_ids[:, 0], positive_reaction_ids[:, 0]]).T
    labels = pd.DataFrame(0, index=names, columns=['label'])
    for ind, d in enumerate(reactions_ids):
        labels.iloc[ind, 0] = np.where(d)[0][0]

    # Plot the reaction by groups of reactions
    if plot_flag:
        rCol = []
        for ind in range(len(names)):
            rCol.append('C' + str(ind))

        fig, ax = plt.subplots()
        plt.title('Group Comparison', fontsize=20)
        plt.ylabel('Ratio', fontsize=12)
        plt.xlabel("Index of time", fontsize=12)
        #grouped_feature_plot = grouped_feature[['r' in s for s in names]]

        groups = [np.where(negative_reaction_ids), np.where(balance_reaction_ids), np.where(positive_reaction_ids)]
        Legend = ['Negative reaction', 'Balance reaction', 'Possitive reaction']
        for ind, groupInds in enumerate(groups):
            if groupInds:
                groupValues = data[groupInds[0], :]
                dataByTimes = np.resize(groupValues, (groupValues.shape[0] * 12, np.shape(data)[1]))

                groupMeans = groupValues.mean(axis=0)
                groupStd = groupValues.std(axis=0)
                ax.plot(groupMeans, color=rCol[ind])
                x = np.linspace(0, len(groupStd) - 1, len(groupStd))
                ax.fill_between(x, groupMeans - groupStd, groupMeans + groupStd, color=rCol[ind], alpha=0.3)

        plt.legend(Legend[0:ind + 1], loc='center left', bbox_to_anchor=(0, 0.9))
        plt.show()

    return labels


def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):
        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


def foundTwoHils(array):
    arrayDiff = -np.diff(np.r_[array[1], array[1:]])
    tempMax = []
    maxInd = []

    for rad in np.arange(0, len(array) - 50, 50):
        tempMax.append(arrayDiff[rad:rad + 50].max())
        maxInd.append(arrayDiff[rad:rad + 50].argmax(axis=0) + rad)
    # First Maxima
    tmepMaxSoretd = np.sort(tempMax)
    tempIndSorted = np.argsort(tempMax)
    localMaximasInds = [maxInd[tempIndSorted[-1]]]
    itterFlag = True
    ind = 2
    while itterFlag and ind < len(tempIndSorted):
        if tempIndSorted[-1] - tempIndSorted[-ind] > 2:
            localMaximasInds.append(maxInd[tempIndSorted[-ind]])
            itterFlag = False
        else:
            ind += 1
    localMaximasInds = localMaximasInds[::-1]

    return localMaximasInds


def compare_hand_distances(principal_breast_Df, subject_id, names, exData):

    minVal = min(exData.values) * 0.8
    maxVal = max(exData.values)
    r = 100 * (exData.values - minVal) / (maxVal - minVal)
    rDF = pd.DataFrame(data=r, index=exData.index)

    rCol = []
    for ind in range(len(names)):
        rCol.append('C' + str(ind))

    in_subject_hand_dist = []
    between_all_subject_hand_dist = []
    between_subject_hand_dist = []
    perm = []

    fig, ax = plt.subplots()
    for i, s in enumerate(subject_id):
        is_match = np.where(np.char.find(names, s) == 0)
        xa = principal_breast_Df.iloc[is_match[0][0], 0]
        ya = principal_breast_Df.iloc[is_match[0][0], 1]
        xb = principal_breast_Df.iloc[is_match[0][1], 0]
        yb = principal_breast_Df.iloc[is_match[0][1], 1]

        relevantR = rDF[[s in t + '_' for t in rDF.index]]
        ax.scatter(xa, ya, c=rCol[i], linewidths=0.5, edgecolors='r', s=relevantR.values * 6,
                   alpha=0.5)  # marker=markers[i],
        plt.text(xa, ya, s.split('_')[1], horizontalalignment='center', verticalalignment='center')
        ax.scatter(xb, yb, c=rCol[i], linewidths=0.5, edgecolors='g', s=relevantR.values * 6, alpha=0.5)
        plt.text(xb, yb, s.split('_')[1], horizontalalignment='center', verticalalignment='center')
        in_subject_hand_dist.append(distance.euclidean((xa, ya), (xb, yb)))

        perm.append([s + s])
        for j, s_n in enumerate(subject_id):
            current_match_1 = [s + s_n]
            current_match_2 = [s_n + s]
            if not (any(np.char.find(perm, current_match_1) == 0) or any(np.char.find(perm, current_match_2) == 0)):
                perm.append(current_match_1)
                s_n_all = [s_n + 'r', s_n + 'l']
                for hand_i, hand_m in enumerate(s_n_all):
                    #dist to the right and left hands
                    xc = principal_breast_Df.loc[hand_m, principal_breast_Df.columns[0]]
                    yc = principal_breast_Df.loc[hand_m, principal_breast_Df.columns[1]]
                    between_all_subject_hand_dist.append(distance.euclidean((xa, ya), (xc, yc)))
                    between_all_subject_hand_dist.append(distance.euclidean((xb, yb), (xc, yc)))


        between_subject_hand_dist.append(np.mean(between_all_subject_hand_dist))
    # ax.legend(names, loc='center left', bbox_to_anchor=(1, 0.5), ncol=3)
    # Statistics of the difference between hands
    t, p = stats.ttest_ind(in_subject_hand_dist, between_subject_hand_dist)
    print('Match hands dist = ' + str(np.round(np.average(in_subject_hand_dist), 4)) + ' + ' + str(
        np.round(np.nanstd(in_subject_hand_dist), 4)))
    print('Not match dist = ' + str(np.round(np.average(between_subject_hand_dist), 4)) + ' + ' + str(
        np.round(np.nanstd(between_subject_hand_dist), 4)))
    print('p = ' + str(np.round(p, 4)))

    prc_75 = np.percentile(in_subject_hand_dist, 75)