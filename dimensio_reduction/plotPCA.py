import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plotPCA2D(dataF, names, hand, exData=[], title = "Principal Component Analysis of All Inteneces",
              save_folder=''):

    #if hand:
    #    relevant_data = dataF[[hand in s for s in names]]
    #    relevant_data = relevant_data.reset_index(drop=True)
    #    subject_id = [x for x in names if np.char.find(x, hand) > 0]
    #else:
    relevant_data = dataF
    subject_id = names


    rCol = []
    for ind in range(len(subject_id)):
        rCol.append('C' + str(ind))

    # Scale ExData
    min_val = min(exData.values)*0.8
    max_val = max(exData.values)
    r = 100*(exData.values-min_val)/(max_val-min_val)
    r_df = pd.DataFrame(data=r, index=exData.index)

    plt.figure(figsize=(10, 10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component - 1', fontsize=20)
    plt.ylabel('Principal Component - 2', fontsize=20)
    plt.title(title, fontsize=20)

    for i, txt in enumerate(subject_id):
        #relevant_r = r_df[[txt in s for s in r_df.index]]
        #if not(np.isnan(relevant_r.values)):
        #    plt.scatter(relevant_data.loc[i, 'principal component 1'], relevant_data.loc[i, 'principal component 2'], c=rCol[i], s=relevant_r.values*6, alpha=0.3)
        #    plt.text(relevant_data.loc[i, 'principal component 1'], relevant_data.loc[i, 'principal component 2'], txt.split('_')[1], horizontalalignment='center', verticalalignment='center')
        #else:
        #    plt.scatter(relevant_data.loc[i, 'principal component 1'], relevant_data.loc[i, 'principal component 2'],
        #                c=rCol[i], marker='x')
        #    plt.text(relevant_data.loc[i, 'principal component 1'], relevant_data.loc[i, 'principal component 2'],
        #             txt.split('_')[0])
        short_name = txt.split('_')[0] + '_' + txt.split('_')[1]

        plt.scatter(relevant_data.loc[txt, 'principal component 1'], relevant_data.loc[txt, 'principal component 2'],
                    c=rCol[i], s=r_df.loc[short_name, 0] * 6, alpha=0.3)
        plt.text(relevant_data.loc[txt, 'principal component 1'], relevant_data.loc[txt, 'principal component 2'],
                 txt.split('_')[1], horizontalalignment='center', verticalalignment='center')

    if save_folder:
        plt.savefig(save_folder)

def plotPCA3D(dataF, names, hand, exData=[], title="Principal Component Analysis of All Inteneces", save_folder=''):

    if hand:
        relevant_data = dataF[[hand in s for s in names]]
        relevant_data = relevant_data.reset_index(drop=True)
        subject_id = [x for x in names if np.char.find(x, hand) > 0]
    else:
        relevant_data = dataF
        subject_id = names


    rCol = []
    for ind in range(len(subject_id)):
        rCol.append('C' + str(ind))

    # Scale ExData
    min_val = min(exData.values)*0.8
    max_val = max(exData.values)
    r = 100*(exData.values-min_val)/(max_val-min_val)
    r_df = pd.DataFrame(data=r, index=exData.index)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('Principal Component - 1', fontsize=8)
    ax.set_ylabel('Principal Component - 2', fontsize=8)
    ax.set_zlabel('Principal Component - 3', fontsize=8)

    ax.title.set_text(title)

    for i, idx in enumerate(subject_id):

        ax.scatter(relevant_data.loc[idx, 'principal component 1'], relevant_data.loc[idx, 'principal component 2'],
                   relevant_data.loc[idx, 'principal component 3'], c=rCol[i], s=r_df.loc[idx, 0] * 6, alpha=0.3)
        ax.text(relevant_data.loc[idx, 'principal component 1'], relevant_data.loc[idx, 'principal component 2'],
                relevant_data.loc[idx, 'principal component 3'], idx.split('_')[1], horizontalalignment='center',
                verticalalignment='center')

    if save_folder:
        plt.savefig(save_folder)
