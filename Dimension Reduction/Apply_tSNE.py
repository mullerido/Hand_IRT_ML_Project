import numpy as np
import pandas as pd
from utils import fashion_scatter, get_hand_fingers_data_relate_to_center
from sklearn.manifold import TSNE
from utils import GetStudyData, GetGravityData
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from ApplyPCA import run_pca_on_df
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

def vizualise_tsne_and_get_distance(grouped_feature, exData, subject_id, perplexity_v=5, learning_rate_v=150):

    fashion_tsne = TSNE(n_components=2, perplexity=perplexity_v, early_exaggeration=12.0,
                        learning_rate=learning_rate_v, n_iter=1000, n_iter_without_progress=300,
                        min_grad_norm=1e-07, metric='euclidean', init='random', verbose=0,
                        random_state=None, method='barnes_hut', angle=0.5).fit_transform(grouped_feature.values)

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
        xa = fashion_tsne[is_match[0][0], 0]
        ya = fashion_tsne[is_match[0][0], 1]
        xb = fashion_tsne[is_match[0][1], 0]
        yb = fashion_tsne[is_match[0][1], 1]

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
                    xa = fashion_tsne[is_match[0][hand_i], 0]
                    ya = fashion_tsne[is_match[0][hand_i], 1]
                    xb = fashion_tsne[j, 0]
                    yb = fashion_tsne[j, 1]
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


if __name__ == "__main__":

    RS = 150
    normalize_flag = True
    #allFeatures = ['Thumbs_dist_Intence', 'Thumbs_proxy_Intence', 'Index_dist_Intence', 'Index_proxy_Intence',
    #               'Middle_dist_Intence', 'Middle_proxy_Intence', 'Ring_dist_Intence', 'Ring_proxy_Intence',
    #               'Pinky_dist_Intence', 'Pinky_proxy_Intence', 'Palm_arch_Intence', 'Palm_Center_Intence']
    allFeatures = ['Thumbs_dist_Intence', 'Thumbs_proxy_Intence', 'Index_dist_Intence', 'Index_proxy_Intence',
                   'Middle_dist_Intence', 'Middle_proxy_Intence', 'Ring_dist_Intence', 'Ring_proxy_Intence',
                   'Pinky_dist_Intence', 'Pinky_proxy_Intence', 'Palm_arch_Intence', 'Palm_Center_Intence']
    hand_to_extract = 'both'
    #[groupedFeature, names] = GetStudyData(normlizeFlag)
    #[grouped_feature, names, subject_id, data] = GetGravityData(allFeatures, normalize_flag, hand_to_extract)
    [grouped_feature, names, subject_id] = get_hand_fingers_data_relate_to_center(normalize_flag, hand_to_extract)
    #[principal_breast_Df, principalComponents] = run_pca_on_df(grouped_feature, 20)

    charectaristcs_PD = pd.read_excel(r"C:\Users\ido.DM\Google Drive\Thesis\Data\Characteristics.xlsx")
    charectaristcs_PD = charectaristcs_PD.set_index('Ext_name')
    exData = charectaristcs_PD.loc[:, [col for col in charectaristcs_PD.columns if 'PP' in col]]
    minVal = min(exData.values) * 0.8
    maxVal = max(exData.values)
    r = 100 * (exData.values - minVal) / (maxVal - minVal)
    rDF = pd.DataFrame(data=r, index=exData.index)

    #run_tSNE_per_input(grouped_feature, rDF)

    if hand_to_extract == 'both':
        in_subject_hand_dist = vizualise_tsne_and_get_distance(grouped_feature, exData, subject_id)
        prc_75 = np.percentile(in_subject_hand_dist, 75)


