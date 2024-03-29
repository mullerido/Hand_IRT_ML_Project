import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, scale  # use to normalize the data
from sklearn.decomposition import PCA  # Use to perform the PCA transform
import matplotlib.pyplot as plt
from common.utils import get_study_data, get_hand_fingers_data_relate_to_center, get_gravity_data, \
    seperate_subjects_by_reaction, compare_hand_distances
from scipy import stats
import seaborn as sns
from dimensio_reduction import plotPCA
from dimensio_reduction.plotUtils import plot_areas_by_groups_pca_dim_single_fid, \
    plot_areas_by_groups_pca_dim_multiple_fid


def run_pca_on_df(grouped_feature, n_comp=2):
    # Data Visualization
    x = grouped_feature.values
    # x_norm = StandardScaler().fit_transform(x) # normalizing the features
    min_max_scaler = MinMaxScaler()
    x_norm = min_max_scaler.fit_transform(x)
    y = scale(grouped_feature.values)

    # groupedFeature['label'] = 'n'

    # convert the normalized features into a tabular format with the help of DataFrame.
    feat_cols = ['feature' + str(i) for i in range(x.shape[1])]
    normalised_Data = pd.DataFrame(x_norm, columns=feat_cols)
    normalised_Data.tail()

    pca_Data = PCA(n_components=n_comp)
    principal_components = pca_Data.fit_transform(x)
    eigen_vectors = pca_Data.components_
    eigen_values = pca_Data.explained_variance_


    # create a DataFrame that will have the principal component values for all samples
    rCol = []
    for ind in range(n_comp):
        rCol.append('principal component ' + str(ind + 1))
    principal_components_Df = pd.DataFrame(data=principal_components,
                                           columns=rCol)  # ['principal component 1', 'principal component 2'])

    # principal_components_Df.tail()  # Return the last n rows.

    # Show for each principle component how much of the informaion it holds
    print('Explained variation per principal component: {}'.format(pca_Data.explained_variance_ratio_))

    return principal_components_Df, principal_components, eigen_vectors, eigen_values


def vizualise_pca_and_get_distance(exData, principal_breast_Df, subject_id):
    rCol = []
    for ind in range(len(subject_id)):
        rCol.append('C' + str(ind))

    minVal = min(exData.values) * 0.8
    maxVal = max(exData.values)
    r = 100 * (exData.values - minVal) / (maxVal - minVal)
    rDF = pd.DataFrame(data=r, index=exData.index)
    # plt.subplot()
    markers = [".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X",
               "D", "d", "o", "v", "^", "<", ">", "1"]
    rgb_values = sns.color_palette("muted", len(subject_id))

    # Plot both hands together + collect distances
    in_subject_hand_dist = []
    between_subject_hand_dist = []
    perm = []
    fig, ax = plt.subplots()
    for i, s in enumerate(subject_id):
        is_match = np.where(np.char.find(names, s) == 0)
        xa = principal_breast_Df.iloc[is_match[0][0], 0]
        ya = principal_breast_Df.iloc[is_match[0][0], 1]
        xb = principal_breast_Df.iloc[is_match[0][1], 0]
        yb = principal_breast_Df.iloc[is_match[0][1], 1]

        relevantR = rDF[[s in t for t in rDF.index]]
        ax.scatter(xa, ya, c=rCol[i], linewidths=0.5, edgecolors='r', s=relevantR.values * 6,
                   alpha=0.5)  # marker=markers[i],
        plt.text(xa, ya, s.split('_')[1], horizontalalignment='center', verticalalignment='center')
        ax.scatter(xb, yb, c=rCol[i], linewidths=0.5, edgecolors='g', s=relevantR.values * 6, alpha=0.5)
        plt.text(xb, yb, s.split('_')[1], horizontalalignment='center', verticalalignment='center')
        in_subject_hand_dist.append(np.linalg.norm(np.array(xa, ya) - np.array(xb, yb)))
        perm.append([s + s])
        for j, s_n in enumerate(subject_id):
            current_match_1 = [s + s_n]
            current_match_2 = [s_n + s]
            if not (any(np.char.find(perm, current_match_1) == 0) or any(np.char.find(perm, current_match_2) == 0)):
                perm.append(current_match_1)
                for hand_i in range(1):
                    xa = principal_breast_Df.iloc[is_match[0][hand_i], 0]
                    ya = principal_breast_Df.iloc[is_match[0][hand_i], 1]
                    xb = principal_breast_Df.iloc[j, 0]
                    yb = principal_breast_Df.iloc[j, 1]
                    between_subject_hand_dist.append(np.linalg.norm(np.array(xa, ya) - np.array(xb, yb)))
    # ax.legend(names, loc='center left', bbox_to_anchor=(1, 0.5), ncol=3)
    # Statistics of the difference between hands
    t, p = stats.ttest_ind(in_subject_hand_dist, between_subject_hand_dist)
    print('Match hands dist = ' + str(np.round(np.average(in_subject_hand_dist), 4)) + ' + ' + str(
        np.round(np.nanstd(in_subject_hand_dist), 4)))
    print('Not match dist = ' + str(np.round(np.average(between_subject_hand_dist), 4)) + ' + ' + str(
        np.round(np.nanstd(between_subject_hand_dist), 4)))
    print('p = ' + str(np.round(p, 4)))

    return in_subject_hand_dist


def split_data_in_pca_dim(principal_cmponents, names, type='axis', x1_thresh=0, y1_thresh=0, x2_thresh=None,
                          y2_thresh=None):
    # Start plot lines by groups on same fig
    relevant_data = principal_cmponents#[['r' in s for s in names]]
    groups = [[], [], [], []]

    if type == 'axis':
        if not (x1_thresh == None) and not (y1_thresh == None):
            choice = np.logical_and(np.greater_equal(relevant_data[:, 0], x1_thresh),
                                    np.greater_equal(relevant_data[:, 1], y1_thresh))
            groups[0] = np.where(choice)
            choice = np.logical_and(np.greater_equal(relevant_data[:, 0], x1_thresh),
                                    np.less(relevant_data[:, 1], y1_thresh))
            groups[1] = np.where(choice)
            choice = np.logical_and(np.less(relevant_data[:, 0], x1_thresh), np.less(relevant_data[:, 1], y1_thresh))
            groups[2] = np.where(choice)
            choice = np.logical_and(np.less(relevant_data[:, 0], x1_thresh),
                                    np.greater_equal(relevant_data[:, 1], y1_thresh))
            groups[3] = np.where(choice)
            Legend = ['1Qtr', '2Qtr', '3Qtr', '4Qtr']

        elif not (x1_thresh == None) and (y1_thresh == None):
            choice = np.greater_equal(relevant_data[:, 0], x1_thresh)
            groups[0] = np.where(choice)
            choice = np.less(relevant_data[:, 0], x1_thresh)
            groups[1] = np.where(choice)
            Legend = ['X_Possitive', 'X_Negative']

        elif (x1_thresh == None) and not (y1_thresh == None):
            choice = np.greater_equal(relevant_data[:, 1], y1_thresh)
            groups[0] = np.where(choice)
            choice = np.less(relevant_data[:, 1], y1_thresh)
            groups[1] = np.where(choice)
            Legend = ['Y_Possitive', 'Y_Negative']

    elif type == 'rectangle':
        # zone 1- inside rectangle and zone 2 outside the rectangle
        choice_in = np.array([relevant_data[:, 0] >= x1_thresh]).T & \
                    np.array([relevant_data[:, 1] >= y1_thresh]).T & \
                    np.array([relevant_data[:, 0] <= x2_thresh]).T & \
                    np.array([relevant_data[:, 1] <= y2_thresh]).T
        groups[0] = np.where(choice_in)

        choice_out = np.array([not elem for elem in choice_in])
        groups[1] = np.where(choice_out)

        Legend = ['In', 'Out']

    return groups, Legend


if __name__ == "__main__":
    normalize_flag = True
    # allFeatures = GetHandFingersDataRelateToCenter()

    allFeatures = ['Thumbs_dist_Intence', 'Thumbs_proxy_Intence', 'Index_dist_Intence', 'Index_proxy_Intence',
                   'Middle_dist_Intence', 'Middle_proxy_Intence', 'Ring_dist_Intence', 'Ring_proxy_Intence',
                   'Pinky_dist_Intence', 'Pinky_proxy_Intence', 'Palm_arch_Intence', 'Palm_Center_Intence']
    allFeatures = ['Thumbs_dist_Intence', 'Index_dist_Intence',
                   'Middle_dist_Intence', 'Ring_dist_Intence',
                   'Pinky_dist_Intence', 'Palm_Center_Intence']

    hand_to_extract = 'both'  # either 'right', 'left' or 'both'
    # [groupedFeature, names, subject_id] = GetStudyData(allFeatures, normlizeFlag)
    hand = ''
    [grouped_feature, names, subject_id, data] = get_gravity_data(allFeatures, normalize_flag, hand, 19)

    # Get data seperated by the reaction
    plot_flag = False
    # reactions_ids = seperate_subjects_by_reaction(data, subject_id, names, plot_flag)

    reactions_ids = seperate_subjects_by_reaction(data, grouped_feature, normalize_flag, subject_id, names,
                                                  plot_flag=False)

    # Run PCA
    [principal_components_Df, principal_components] = run_pca_on_df(grouped_feature)
    principal_components_Df.index = names
    # Plot the visualization of the two PCs
    charectaristcs_PD = pd.read_excel(r"G:\My Drive\Thesis\Data\Characteristics.xlsx")
    short_name = []
    for index, row in charectaristcs_PD.iterrows():
        short_name.append(row['Ext_name'].split('_')[0] + '_' + row['Ext_name'].split('_')[1])
    charectaristcs_PD.index = short_name  #
    exData = charectaristcs_PD.loc[:, [col for col in charectaristcs_PD.columns if 'PP' in col]]
    plotPCA.plotPCA2D(principal_components_Df, names, hand[0], exData, "Principal Component Analysis of All Inteneces",
                      r'G:\My Drive\Thesis\Project\Results\PCA\27Oct2021\PCA_right_hand_dist_and_center_gravity')

    # Plot areas by groups in PCA dimension
    [groups, Legend] = split_data_in_pca_dim(principal_components, names,
                                             type='rectangle', x1_thresh=-1, y1_thresh=-1, x2_thresh=0.9, y2_thresh=0.5)

    mean_data_df = pd.DataFrame(data)
    mean_data_df.index = names
    plot_areas_by_groups_pca_dim_single_fid(mean_data_df, subject_id, groups, Legend, normalize_flag)

    plot_areas_by_groups_pca_dim_multiple_fid(grouped_feature, subject_id, groups, Legend, normalize_flag)

    '''
    if hand_to_extract == 'both':
        in_subject_hand_dist = vizualise_pca_and_get_distance(exData, principal_components_Df, subject_id)
        prc_75 = np.percentile(in_subject_hand_dist, 75)
    '''
    charectaristcs_PD = pd.read_excel(r"G:\My Drive\Thesis\Data\Characteristics.xlsx")
    charectaristcs_PD = charectaristcs_PD.set_index('Ext_name')
    exData = charectaristcs_PD.loc[:, [col for col in charectaristcs_PD.columns if 'PP' in col]]
    if not hand:
        compare_hand_distances(principal_components_Df, subject_id, names, exData)
