import numpy as np
import pandas as pd
from utils import GetStudyData
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

normlize_flag = True
all_features = ['Thumbs_dist_Intence', 'Thumbs_proxy_Intence', 'Index_dist_Intence', 'Index_proxy_Intence',
                   'Middle_dist_Intence', 'Middle_proxy_Intence', 'Ring_dist_Intence', 'Ring_proxy_Intence',
                   'Pinky_dist_Intence', 'Pinky_proxy_Intence', 'Palm_arch_Intence', 'Palm_Center_Intence']
all_features = ['Thumbs_proxy_Intence', 'Index_proxy_Intence', 'Middle_proxy_Intence', 'Ring_proxy_Intence',
                   'Pinky_proxy_Intence']

folder = r'C:\Projects\Thesis\Feature-Analysis- matlab\Resize Feature\\'

files = os.listdir(folder)
files_xls = [f for f in files if f[-5:] == '.xlsx']
#files_xls = [f for f in files if f[-10:] == 'right.xlsx']
grouped_feature = pd.DataFrame()
names = []
subject_id=[]
inc_c = np.empty((19, 19, np.shape(files_xls)[0]))
data = np.empty((np.shape(files_xls)[0], 19))
for ind, file in enumerate(files_xls):
    print(file)
    current_file = folder + file
    current_features = pd.read_excel(current_file)
    if normlize_flag:  # Use the ratio
        relevant_cols = (current_features.loc[0:18, all_features] - current_features.loc[0, all_features]) / \
                        current_features.loc[0, all_features]
    else:  # Use the actual values
        relevant_cols = current_features.loc[:, all_features]
    #transformDataFrame = pd.DataFrame(relevant_cols).T
    cov_mat = np.cov(np.array(relevant_cols.values))
    inc_c[:, :, ind] = np.linalg.pinv(cov_mat)

    data[ind, :] = relevant_cols.mean(axis=1)

    ind_name = file[:-5]
    names.append(ind_name[:])
    subject_id.append(ind_name.split('_')[0] + '_' + ind_name.split('_')[1] + '_')

subject_id = np.unique(subject_id)

# Seperate subjects by their reaction
data_mean = data.mean(axis=1, keepdims=True)
data_std = data.std(axis=1, keepdims=True)
data_mean_std = np.array([data_mean[:,0] - data_std[:,0], data_mean[:,0] + data_std[:,0]]).T

positive_reaction_ids = np.array([data_mean_std[:, 0] > 0]).T
negative_rection_ids = np.array([data_mean_std[:, 1] < 0]).T
balance_reaction_ids = (positive_reaction_ids == False) & (negative_rection_ids == False)
reactions_ids = np.array([negative_rection_ids[:, 0], balance_reaction_ids[:, 0], positive_reaction_ids[:, 0]]).T
# compute the pairwise distances based on the inverse covariance matrix of each point
d_size = np.shape(data)[0]
Dis = np.empty((d_size, d_size))
for ind in range(d_size):
    for jnd in range(d_size):
        Dis[ind, jnd] = 0.5*np.matmul(np.matmul(np.array([data[ind, :] - data[jnd, :]]),  np.array([inc_c[:, :, jnd]+inc_c[:, :, ind]]))[0],
                                      np.array([data[ind, :]-data[jnd, :]]).T)

ep_factor = 4
ep = 1000
alpha = 1

kernel = np.exp(-Dis/(ep_factor*ep))

# Normalization
sum_alpha = pow(kernel.sum(axis=1, keepdims=True), alpha)
symmetric_sum = np.matmul(sum_alpha.reshape(-1, 1), sum_alpha.reshape(1, -1))
norm_kernel = (kernel/ symmetric_sum)
# second normalization to make it row - stochatic
sum_c = np.sum(norm_kernel, axis=1)
for ind in range(d_size):
    norm_kernel[ind, :] = norm_kernel[ind, :]/sum_c[ind]

u, d_ker, v_ker = np.linalg.svd(norm_kernel)

markers = ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']
#rgb_values = sns.color_palette("muted",  len(subject_id))
rgb_values = []
for ind in range(len(subject_id)):
    rgb_values.append('C'+str(ind))
rgb_values = ['C0', 'C1', 'C2']
in_subject_hand_dist = []
between_subject_hand_dist=[]
perm = []
fig, ax = plt.subplots()
for i, s in enumerate(subject_id):
    is_match = np.where(np.char.find(names, s) == 0)
    xa = d_ker[1] * v_ker[is_match[0][0], 1]
    ya = d_ker[2] * v_ker[is_match[0][0], 2]
    xb = d_ker[1] * v_ker[is_match[0][1], 1]
    yb = d_ker[2] * v_ker[is_match[0][1], 2]
    color_a = np.where(reactions_ids[is_match[0][0], :])[0]
    ax.scatter(xa, ya,  c=rgb_values[color_a[0]], linewidths=0.5, edgecolors='r', alpha=0.5)
    ax.scatter(xb, yb, c=rgb_values[color_a[0]], linewidths=0.5, edgecolors='g', alpha=0.5)
    in_subject_hand_dist.append(np.linalg.norm(np.array(xa, ya) - np.array(xb, yb)))
    perm.append([s + s])
    for j, s_n in enumerate(subject_id):
        current_match_1 = [s + s_n]
        current_match_2 = [s_n + s]
        if not(any(np.char.find(perm, current_match_1) == 0) or any(np.char.find(perm, current_match_2) == 0)):
            perm.append(current_match_1)
            for hand_i in range(1):
                xa = d_ker[1] * v_ker[is_match[0][hand_i], 1]
                ya = d_ker[2] * v_ker[is_match[0][hand_i], 2]
                xb = d_ker[1] * v_ker[j, 1]
                yb = d_ker[2] * v_ker[j, 2]
                between_subject_hand_dist.append(np.linalg.norm(np.array(xa, ya) - np.array(xb, yb)))

t, p = stats.ttest_ind(in_subject_hand_dist, between_subject_hand_dist)
print('in_patient_dist = ' + str(np.round(np.average(in_subject_hand_dist), 4)) + ' + ' + str(np.round(np.nanstd(in_subject_hand_dist), 4)))
print('in_patient_dist = ' + str(np.round(np.average(between_subject_hand_dist), 4)) + ' + ' + str(np.round(np.nanstd(between_subject_hand_dist), 4)))
print('p = ' + str(np.round(p, 4)))

plt.show()