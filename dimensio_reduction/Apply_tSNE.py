import numpy as np
import pandas as pd
from common.utils import fashion_scatter, get_hand_fingers_data_relate_to_center
from sklearn.manifold import TSNE
from common.utils import get_study_data, get_hand_fingers_data_relate_to_center, get_gravity_data, compare_hand_distances
import matplotlib.pyplot as plt
from dimensio_reduction.ApplyPCA import split_data_in_pca_dim
from dimensio_reduction.plotUtils import plot_areas_by_groups_pca_dim_single_fid, plot_areas_by_groups_pca_dim_multiple_fid

from scipy import stats
import seaborn as sns
#from ApplyPCA import run_pca_on_df

'''
    Load training data:
    Rows = N samples
    Columns = features
'''


def run_tSNE_per_input(data, radius):
    # testVals = np.linspace(5, 30, 30/5)#perplexity
    testVals = np.linspace(50, 1000, 20)
    fig, axs = plt.subplots(5, 4, figsize=(15, 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.5, wspace=.001)
    axs = axs.ravel()
    plt.title('tSNE- Vs perplexity', fontsize=20)

    perplexityVal = 5
    # learning_rate = 150
    for i, learning_rate in enumerate(testVals):

        fashion_tsne = TSNE(n_components=2, perplexity=perplexityVal, early_exaggeration=12.0,
                            learning_rate=learning_rate, n_iter=1000, n_iter_without_progress=300,
                            min_grad_norm=1e-07, metric='euclidean', init='random', verbose=0,
                            random_state=None, method='barnes_hut', angle=0.5).fit_transform(data.values)
        for j, txt in enumerate(subject_id):
            relevantR = rDF[[txt in s for s in radius.index]]
            # axs[i].scatter(dataF.loc[i, 'principal component 1'], dataF.loc[i, 'principal component 2'], c='b', s=relevantR.values*6, alpha=0.3)
            # plt.text(dataF.loc[i, 'principal component 1'], dataF.loc[i, 'principal component 2'], txt)
            axs[i].scatter(fashion_tsne[j, 0], fashion_tsne[j, 1], lw=0, c='b', alpha=0.3, s=relevantR.values)
        axs[i].set_title('learning_rate= ' + str(learning_rate), fontsize=8)

    fashion_scatter(fashion_tsne, np.array([0] * fashion_tsne.shape[0]))


def vizualise_tsne(grouped_feature, exData, names, n_components=2, perplexity_v=5, learning_rate_v=150):
    fashion_tsne = TSNE(n_components, perplexity=perplexity_v, early_exaggeration=12.0,
                        learning_rate=learning_rate_v, n_iter=1000, n_iter_without_progress=300,
                        min_grad_norm=1e-07, metric='euclidean', init='random', verbose=0,
                        random_state=None, method='barnes_hut', angle=0.5).fit_transform(grouped_feature.values)

    rCol = []
    for ind in range(len(grouped_feature)):
        rCol.append('C' + str(ind))

    minVal = min(exData.values) * 0.8
    maxVal = max(exData.values)
    r = 100 * (exData.values - minVal) / (maxVal - minVal)
    rDF = pd.DataFrame(data=r, index=exData.index)

    # Plot both hands together + collect distances
    if n_components == 2:
        fig, ax = plt.subplots()
        for i, s in enumerate(names):
            is_match = np.where(np.char.find(names, s) == 0)
            xa = fashion_tsne[i, 0]
            ya = fashion_tsne[i, 1]

            s_short = s.split('_')[0] + '_' + s.split('_')[1] + '_'
            match_r_ind = np.where([s_short in t + '_' for t in rDF.index])
            relevantR = rDF.iloc[match_r_ind[0][0], 0]
            if np.isnan(relevantR):
                relevantR = minVal/10
            ax.scatter(xa, ya, c=rCol[match_r_ind[0][0]], linewidths=0.5, edgecolors='r', alpha=0.5, s=np.pi*relevantR*6)
                       # marker=markers[i],
            plt.text(xa, ya, s.split('_')[1], horizontalalignment='center', verticalalignment='center')

    elif n_components == 3:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel('Dim - 1', fontsize=8)
        ax.set_ylabel('Dim - 2', fontsize=8)
        ax.set_zlabel('Dim - 3', fontsize=8)

        ax.title.set_text('tSNE Comparison')

        for i, s in enumerate(names):
                xa = fashion_tsne[i, 0]
                ya = fashion_tsne[i, 1]
                za = fashion_tsne[i, 2]

                s_short = s.split('_')[0] + '_' + s.split('_')[1] + '_'
                match_r_ind = np.where([s_short in t + '_' for t in rDF.index])
                relevantR = rDF.iloc[match_r_ind[0][0], 0]

                ax.scatter(xa, ya, za, c=rCol[i], s=np.pi*relevantR,
                           alpha=0.3)

                ax.text(xa, ya, za, s_short.split('_')[1],
                        horizontalalignment='center',
                        verticalalignment='center')

    return fashion_tsne

if __name__ == "__main__":

    RS = 150
    normalize_flag = True

    allFeatures = ['Thumbs_dist_Intence', 'Thumbs_proxy_Intence', 'Index_dist_Intence', 'Index_proxy_Intence',
                   'Middle_dist_Intence', 'Middle_proxy_Intence', 'Ring_dist_Intence', 'Ring_proxy_Intence',
                   'Pinky_dist_Intence', 'Pinky_proxy_Intence', 'Palm_arch_Intence', 'Palm_Center_Intence']
    allFeatures = ['Thumbs_dist_Intence', 'Index_dist_Intence',
                   'Middle_dist_Intence', 'Ring_dist_Intence',
                   'Pinky_dist_Intence', 'Palm_Center_Intence']
    hand_to_extract = 'right'
    # [groupedFeature, names] = GetStudyData(normlizeFlag)
    #[grouped_feature, names, subject_id, data] = get_gravity_data(allFeatures, normalize_flag, hand_to_extract)
    [grouped_feature, names, subject_id] = get_hand_fingers_data_relate_to_center(normalize_flag, hand_to_extract)
    # [principal_breast_Df, principalComponents] = run_pca_on_df(grouped_feature, 20)

    charectaristcs_PD = pd.read_excel(r"G:\My Drive\Thesis\Data\Characteristics.xlsx")
    #charectaristcs_PD = charectaristcs_PD.set_index('Ext_name')
    short_name = []
    for index, row in charectaristcs_PD.iterrows():
        short_name.append(row['Ext_name'].split('_')[0] + '_' + row['Ext_name'].split('_')[1])
    charectaristcs_PD.index = short_name  #
    exData = charectaristcs_PD.loc[:, [col for col in charectaristcs_PD.columns if 'BMI' in col]]
    minVal = min(exData.values) * 0.8
    maxVal = max(exData.values)
    r = 100 * (exData.values - minVal) / (maxVal - minVal)
    rDF = pd.DataFrame(data=r, index=exData.index)

    # run_tSNE_per_input(grouped_feature, rDF)
    fashion_tsne = vizualise_tsne(grouped_feature, exData, names, 2, 5, 150)

    if hand_to_extract == 'both':
        fashion_tsne_df = pd.DataFrame(fashion_tsne)
        fashion_tsne_df.index = names
        compare_hand_distances(fashion_tsne_df, subject_id, names, exData)

    else:
        # Plot areas by grops in PCA dimension
        [groups, Legend] = split_data_in_pca_dim(fashion_tsne, names,
                                                 type='rectangle', x1_thresh=-1, y1_thresh=-1, x2_thresh=0.9,
                                                 y2_thresh=0.5)


def vizualise_tsne_and_get_distance():
    return None