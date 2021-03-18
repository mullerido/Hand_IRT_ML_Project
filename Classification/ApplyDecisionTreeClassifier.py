# Load libraries
import numpy as np
import pandas as pd
import os
from utils import GetStudyData, GetHandFingersDataRelateToCenter, GetGravityData
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
# For Visualizing the Decision Tree
from sklearn.tree import export_graphviz, plot_tree
from six import StringIO
from IPython.display import Image
import pydotplus
import graphviz
import ApplyPCA

def get_all_features_from_xls (all_features = ['Thumbs_dist_Intence', 'Thumbs_proxy_Intence', 'Index_dist_Intence', 'Index_proxy_Intence',
                   'Middle_dist_Intence', 'Middle_proxy_Intence', 'Ring_dist_Intence', 'Ring_proxy_Intence',
                   'Pinky_dist_Intence', 'Pinky_proxy_Intence', 'Palm_arch_Intence', 'Palm_Center_Intence']):
    """
    Get features from xls files as is and parse it to be use with decision tree
    """
#all_features = ['Thumbs_dist_Intence', 'Thumbs_proxy_Intence', 'Index_dist_Intence', 'Index_proxy_Intence',
#                   'Middle_dist_Intence', 'Middle_proxy_Intence', 'Ring_dist_Intence', 'Ring_proxy_Intence',
#                   'Pinky_dist_Intence', 'Pinky_proxy_Intence', 'Palm_arch_Intence', 'Palm_Center_Intence']

    folder = r'C:\Projects\Thesis\Feature-Analysis- matlab\Resize Feature\\'

    files = os.listdir(folder)
    file_name_inc = 'right.xlsx'
    files_xls = [f for f in files if f[-len(file_name_inc):] == file_name_inc]
    #files_xls = [f for f in files if f[-10:] == 'right.xlsx']
    grouped_feature = pd.DataFrame(columns=['Palm_Center_Intence', 'Proxy_dist', 'Dist_dist'])
    names = []
    subject_id=[]
    inc_c = np.empty((19, 19, np.shape(files_xls)[0]))
    data = np.empty((np.shape(files_xls)[0], 19))
    for ind, file in enumerate(files_xls):
        print(file)
        current_file = folder + file
        current_features = pd.read_excel(current_file)
        relevant_cols = current_features.loc[0, all_features]
        relevant_cols = relevant_cols
        ind_name = file[:-5]
        names.append(ind_name[:])
        subject_id.append(ind_name.split('_')[0] + '_' + ind_name.split('_')[1] + '_')

        # Use dist from center as feature
        dist_from_center = relevant_cols[:]-relevant_cols['Palm_Center_Intence']

        # Use dist from center as feature
        dist_from_center = relevant_cols[:] - relevant_cols['Palm_Center_Intence']

        grouped_feature.loc[ind, 'Palm_Center_Intence'] = relevant_cols['Palm_Center_Intence']
        grouped_feature.loc[ind, 'Proxy_dist'] = np.mean(dist_from_center[['proxy' in s for s in dist_from_center.index]])
        grouped_feature.loc[ind, 'Dist_dist'] = np.mean(dist_from_center[['dist' in s for s in dist_from_center.index]])
        #grouped_feature = grouped_feature.append([relevant_cols], ignore_index=False)

    grouped_feature.index = names

    subject_id = np.unique(subject_id)

    return grouped_feature, subject_id, names

def add_charectaristics_to_features(grouped_feature, char_cols_names):
    # Add charectaristics
    charectaristcs_PD = pd.read_excel(r"C:\Users\ido.DM\Google Drive\Thesis\Data\Characteristics.xlsx")
    charectaristcs_PD = charectaristcs_PD.set_index('Ext_name')
    char_indices =charectaristcs_PD.index
    for i, s in enumerate(charectaristcs_PD.index):
        charectaristcs_PD.rename(index={s: s.split('_')[0] + '_' + s.split('_')[1]})

    for ind, var in enumerate(char_cols_names):
        exData= charectaristcs_PD.loc[:, [col for col in charectaristcs_PD.columns if var in col]]
        grouped_feature[var] = np.nan
        for i, s in enumerate(charectaristcs_PD.index):
            grouped_feature.loc[[row for row in grouped_feature.index if s[:-2] in row], var]= exData.loc[s, var]

    return grouped_feature

def get_labels_using_gravity_ratio(all_features):
    # Get the result of the gravity phase
    normlizeFlag = True
    [groupedFeature, names, subject_id, data] = GetGravityData(all_features, normlizeFlag)

    # Seperate subjects by their reaction
    data_mean = data.mean(axis=1, keepdims=True)
    data_std = data.std(axis=1, keepdims=True)
    data_mean_std = np.array([data_mean[:, 0] - data_std[:, 0], data_mean[:, 0] + data_std[:, 0]]).T

    positive_reaction_ids = np.array([data_mean_std[:, 0] > 0]).T
    negative_rection_ids = np.array([data_mean_std[:, 1] < 0]).T
    balance_reaction_ids = (positive_reaction_ids == False) & (negative_rection_ids == False)
    reactions_ids = np.array([negative_rection_ids[:, 0], balance_reaction_ids[:, 0], positive_reaction_ids[:, 0]]).T
    labels = pd.DataFrame(0, index=groupedFeature.index, columns=['label'])
    for ind, d in enumerate(reactions_ids):
        labels.iloc[ind, 0] = np.where(d)[0][0]

    return labels

def get_labels_by_hands_similiraty(all_features):

    [groupedFeature, names, subject_id, data] = ApplyPCA.GetGravityData(all_features, True)

    plot_flag = False
    reactions_ids = ApplyPCA.seperate_subjects_by_reaction(groupedFeature, subject_id, names, plot_flag)

    return labels

def run_decision_tree (grouped_feature, criterion="gini", max_depth=3 ):
    # Run over the data iteritevly
    num_itter = range(30)
    accuracy =[]

    for itter in num_itter:
        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(grouped_feature, labels, test_size=0.25,
                                                            random_state=itter)  # 70% training and 30% test

        # Create Decision Tree classifer object
        clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)

        # Train Decision Tree Classifer
        clf = clf.fit(X_train, y_train)
        clf.set_params()

        # Predict the response for test dataset
        y_pred = clf.predict(X_test)
        # Model Accuracy, how often is the classifier correct?
        accuracy.append(metrics.accuracy_score(y_test, y_pred))
        if accuracy[-1] >= np.max(accuracy):
            dts = clf

    # Visualizing Decision Trees
    feature_cols = grouped_feature.columns.tolist()
    dot_data = StringIO()
    #export_graphviz(clf, out_file=dot_data,
        #                filled=True, rounded=True, special_characters=True,
        #                feature_names=feature_cols, class_names=['0', '1', '2'])
    export_graphviz(dts, out_file=dot_data, feature_names=feature_cols,
                                   class_names=['0', '1', '2'], filled=True,
                                   rounded=True, special_characters=True)
    #graph = graphviz.Source(dot_data)

    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

    graph.write_png(r'C:\Users\ido.DM\Google Drive\Thesis\Project\Process- Py\Decision Tree\gini' + str(itter) + '.png')
    Image(graph.create_png())

    return dts, accuracy

if __name__ == "__main__":
    all_features = ['Thumbs_dist_Intence', 'Thumbs_proxy_Intence', 'Index_dist_Intence', 'Index_proxy_Intence',
                    'Middle_dist_Intence', 'Middle_proxy_Intence', 'Ring_dist_Intence', 'Ring_proxy_Intence',
                    'Pinky_dist_Intence', 'Pinky_proxy_Intence', 'Palm_arch_Intence', 'Palm_Center_Intence']
    [grouped_feature, subject_id, names] = get_all_features_from_xls(all_features)

    labels = get_labels_by_hands_similiraty(all_features, grouped_feature, subject_id, names)

    char_cols_names = ['SBP', 'DBP', 'PP', 'Age']
    grouped_feature = add_charectaristics_to_features(grouped_feature, char_cols_names)

    labels = get_labels_using_gravity_ratio(all_features)


    grouped_feature = grouped_feature.drop(['Palm_Center_Intence', 'Proxy_dist', 'Dist_dist'], axis=1)
    criterion = "gini"
    max_depth = 3
    dts, accuracy = run_decision_tree(grouped_feature, criterion, max_depth)