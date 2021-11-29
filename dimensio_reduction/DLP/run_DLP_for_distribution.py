import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dimensio_reduction.DLP.import_data_utils import get_gravity_data, get_relevant_data_from_file
from sklearn.manifold import TSNE
from dimensio_reduction.Apply_tSNE import vizualise_tsne
from dimensio_reduction.plotUtils import plot_areas_by_groups_pca_dim_single_fid
from dimensio_reduction.ApplyPCA import run_pca_on_df, split_data_in_pca_dim
from dimensio_reduction import plotPCA

if __name__ == "__main__":
    normalize_flag = False
    # allFeatures = GetHandFingersDataRelateToCenter()
    per_patient_result = []
    all_features = ['Thumbs_dist_Intence', 'Thumbs_proxy_Intence',
                    'Index_dist_Intence', 'Index_proxy_Intence',
                    'Middle_dist_Intence', 'Middle_proxy_Intence',
                    'Ring_dist_Intence', 'Ring_proxy_Intence',
                    'Pinky_dist_Intence', 'Pinky_proxy_Intence',
                    'Palm_Center_Intence']
    all_features = ['Thumbs_dist_Intence',
                    'Index_dist_Intence',
                    'Middle_dist_Intence',
                    'Ring_dist_Intence',
                    'Pinky_dist_Intence',
                    'Palm_Center_Intence']

    n_feature = np.shape(all_features)[0]

    t_samples = 19  # Should be between 1 and 39- where 19 is for the gravitation phase and 39 for the entire session
    save_folder = r'G:\My Drive\Thesis\Project\Results\tSNE\right hand ditribution\\'  # 'C:\Users\ido.DM\Google Drive\Thesis\Project\Results\Two-D Laplacian\One Vs all\\'
    all_dist_method = [
        'euclidean']  # , 'correlation', 'cosine', 'minkowski']  # 'minkowski'  # 'cosine'#'euclidean'/'correlation'
    # ['Thumbs_dist_Intence', 'Thumbs_proxy_Intence', 'Index_dist_Intence', 'Index_proxy_Intence',
    # 'Middle_dist_Intence', 'Middle_proxy_Intence', 'Ring_dist_Intence', 'Ring_proxy_Intence',
    # 'Pinky_dist_Intence', 'Pinky_proxy_Intence', 'Palm_arch_Intence', 'Palm_Center_Intence']
    hand = 'right'  # either 'right', 'left' or 'both'
    hand_to_extract = 'right'
    smooth_flag = True
    import_file = r'C:\Projects\Hand_IRT_Auto_Ecxtraction\Feature-Analysis- matlab\Normalized_smooth_data.json'

    #[data_cube, mean_data, data_cube_l, mean_data_l, subject_ids] = get_relevant_data_from_file(import_file,
    #                                                                                                    t_samples,
    #                                                                                                    all_features)

    [data_cube, names, subject_ids, mean_data] = get_gravity_data(all_features, normalize_flag, smooth_flag, hand,
                                                                  t_samples)
    mean_data_df = pd.DataFrame(data=mean_data)
    mean_data_df.index = subject_ids
    n_subjects = np.shape(data_cube)[0]

    # Convert to data of diff between dist to center per time

    # 1- Use to compare all to center
    data_cube_distribution = np.zeros((n_subjects, n_feature-1, t_samples))
    data_distribution = np.zeros((n_subjects, t_samples))
    for s_ind in range(n_subjects):
        subject_data = data_cube[s_ind, :, :]
        data_cube_distribution[s_ind, :, :] = subject_data[:-1, :] - subject_data[5, :]
        data_distribution[s_ind, :] = np.mean(data_cube_distribution[s_ind, :, :], 0)
    '''
    #2- Use to compare with both proxy and dist
    n_grad_features = int((n_feature - 1)/2)#should be=5
    grad_data_cube_distribution = np.zeros((n_subjects, n_grad_features, t_samples))
    grad_data_distribution = np.zeros((n_subjects, t_samples))
    for s_ind in range(n_subjects):
        subject_data = data_cube[s_ind, :, :]
        for n_f in range(n_grad_features):
            grad_data_cube_distribution[s_ind, n_f, :] = subject_data[int(n_f*2), :] - \
                                                                subject_data[int(n_f*2)+1, :]
    '''


    data_df = pd.DataFrame(data=data_distribution)
    data_df.index = subject_ids

    # Choose characteristics as the Radius
    charectaristcs_PD = pd.read_excel(r"G:\My Drive\Thesis\Data\Characteristics.xlsx")
    charectaristcs_PD = charectaristcs_PD.set_index('Ext_name_short')
    exData = charectaristcs_PD.loc[:, [col for col in charectaristcs_PD.columns if 'PP' in col]]
    minVal = min(exData.values) * 0.8
    maxVal = max(exData.values)
    r = 100 * (exData.values - minVal) / (maxVal - minVal)
    rDF = pd.DataFrame(data=r, index=exData.index)

    # Run PCA- per time ave dist
    [principal_components_Df, principal_components, eigen_vectors, eigen_values] = run_pca_on_df(data_df, 2)
    principal_components_Df.index = subject_ids

    save_path = save_folder + '2-D PCA_roi_ave' + '.png'
    title = 'PCA ROI Ave'
    plotPCA.plotPCA2D(principal_components_Df, subject_ids, '', exData, title, save_path)

    [groups, Legend] = split_data_in_pca_dim(principal_components, subject_ids,
                                             type='axis', x1_thresh=0,
                                             y1_thresh=None)  # , x2_thresh=0.9, y2_thresh=0.5)
    plot_areas_by_groups_pca_dim_single_fid(mean_data_df, subject_ids, groups, Legend, True)

    # Run tSNE
    perplexity_v = 5
    learning_rate_v = 150
    rCol = []
    for ind in range(len(subject_ids)):
        rCol.append('C' + str(ind))

    fashion_tsne = vizualise_tsne(data_df, exData, subject_ids, n_components=2, perplexity_v=5, learning_rate_v=150)

    [groups, Legend] = split_data_in_pca_dim(fashion_tsne, subject_ids,
                                             type='axis', x1_thresh=200, y1_thresh=-220)#, x2_thresh=0.9, y2_thresh=0.5)

    plot_areas_by_groups_pca_dim_single_fid(data_df, subject_ids, groups, Legend, True)

    '''
    fashion_tsne = TSNE(n_components=3, perplexity=perplexity_v, early_exaggeration=12.0,
                        learning_rate=learning_rate_v, n_iter=1000, n_iter_without_progress=300,
                        min_grad_norm=1e-07, metric='euclidean', init='random', verbose=0,
                        random_state=None, method='barnes_hut', angle=0.5).fit_transform(data_df.values)

    # fig, ax = plt.subplots()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax = plt.axes(projection='3d')
    for i, s in enumerate(subject_ids):
        xa = fashion_tsne[i, 0]
        ya = fashion_tsne[i, 1]
        za = fashion_tsne[i, 2]
        relevantR = rDF.loc[s, 0]
        ax.scatter(xa, ya, za, c=rCol[i], linewidth=0.5, s=relevantR * 5, alpha=0.6)

        ax.text(xa, ya, za, s.split('_')[1])  # , horizontalalignment='center', verticalalignment='center')
    ax.title.set_text('tSNE- roi Ave')
    save_path = save_folder + '3D tSNE_roi_ave' + '.png'
    plt.savefig(save_folder)
    '''