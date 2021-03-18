import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as PathEffects


def GetGravityData(allFeatures, normlizeFlag=False, hand=''):
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
    subject_id = []
    data = np.empty((np.shape(files_xls)[0], 19))
    for ind, file in enumerate(files_xls):
        print(file)
        currentFile = folder + file
        currentFeatures = pd.read_excel(currentFile)
        if normlizeFlag:  # Use the ratio
            relevant_cols = (currentFeatures.loc[0:18, allFeatures] - currentFeatures.loc[0, allFeatures]) / \
                            currentFeatures.loc[0, allFeatures]
            data[ind, :] = relevant_cols.mean(axis=1)
        else:  # Use the actual values
            # relevantCols = currentFeatures.filter(regex='Intence')
            relevant_cols = currentFeatures.loc[0:18, allFeatures]
            data[ind, :] = relevant_cols.mean(
                (currentFeatures.loc[0:18, allFeatures] - currentFeatures.loc[0, allFeatures]) /
                currentFeatures.loc[0, allFeatures], axis=1)

        transformData = relevant_cols.values.ravel('F')
        transformDataFrame = pd.DataFrame(transformData).T
        indName = file[:-5]
        transformDataFrame = transformDataFrame.rename(index={0: indName})
        grouped_feature = grouped_feature.append(transformDataFrame)
        ind_name = file[:-5]
        names.append(ind_name.split('_')[0] + '_' + ind_name.split('_')[1] + '_' + ind_name.split('_')[2][0])
        subject_id.append(ind_name.split('_')[0] + '_' + ind_name.split('_')[1] + '_')
    subject_id = np.unique(subject_id)

    return grouped_feature, names, subject_id, data


def GetStudyData(allFeatures, normlizeFlag=False):
    folder = r'C:\Projects\Thesis\Feature-Analysis- matlab\Resize Feature\\'

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
        if normlizeFlag:  # Use the ratio
            relevant_cols = (currentFeatures.loc[0:, allFeatures] - currentFeatures.loc[0, allFeatures]) / \
                            currentFeatures.loc[0, allFeatures]
        else:  # Use the actual values
            # relevantCols = currentFeatures.filter(regex='Intence')
            relevant_cols = currentFeatures.loc[0:, allFeatures]

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


# Utility function to visualize the outputs of PCA and t-SNE

def get_hand_fingers_data_relate_to_center(normalize_flag=False, hand=''):
    grouped_feature = pd.DataFrame()
    folder = r'C:\Projects\Thesis\Feature-Analysis- matlab\Resize Feature\\'

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

    for file in files_xls:
        print(file)
        currentFile = folder + file
        currentFeatures = pd.read_excel(currentFile)

        # relevantCols = currentFeatures.filter(regex='Intence')
        relevantFingersCols = np.transpose(currentFeatures.loc[:, allingers].values)
        relevantPalmCols = currentFeatures.loc[:, 'Palm_Center_Intence'].values
        subtract_fingers = relevantFingersCols - relevantPalmCols
        #subtract_fingers = pd.DataFrame(subtract_fingers)
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
