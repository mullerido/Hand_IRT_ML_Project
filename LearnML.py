from sklearn.datasets import load_breast_cancer
from keras.datasets import cifar10
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, scale  # use to normalize the data
from sklearn.decomposition import PCA  # Use to perform the PCA transform
import matplotlib.pyplot as plt
import gzip
import os
import time
from utils import fashion_scatter
from sklearn.manifold import TSNE

# For Decision Tree
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
# For Visualizing the Decision Tree
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

#matplotlib inline

import seaborn as sns

def load_mnist(path, kind='train'):  # For tSNE

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def Learn_tSNE():
    '''
        Load training data:
        Rows = N samples
        Columns = features
    '''
    x_train, y_train = load_mnist(r'X:\GA\Users\Ido\Thesis\Learning ML\tSNE', kind='train')

    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})
    RS = 123

    # Subset first 20k data points to visualize
    x_subset = x_train[0:20000]
    y_subset = y_train[0:20000]

    # Try PCA first
    time_start = time.time()

    pca = PCA(n_components=4)
    pca_result = pca.fit_transform(x_subset)
    print('PCA done! Time elapsed: {} seconds'.format(time.time() - time_start))

    pca_df = pd.DataFrame(columns=['pca1', 'pca2', 'pca3', 'pca4'])

    pca_df['pca1'] = pca_result[:, 0]
    pca_df['pca2'] = pca_result[:, 1]
    pca_df['pca3'] = pca_result[:, 2]
    pca_df['pca4'] = pca_result[:, 3]

    print('Variance explained per principal component: {}'.format(pca.explained_variance_ratio_))

    top_two_comp = pca_df[['pca1', 'pca2']]  # taking first and second principal component
    fashion_scatter(top_two_comp.values, y_subset) # Visualizing the PCA output

    # t-SNE
    #time_start = time.time()
    fashion_tsne = TSNE(random_state=RS).fit_transform(x_subset)

    #print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
    #fashion_scatter(fashion_tsne, y_subset)

    time_start = time.time()

    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(x_subset)

    fashion_pca_tsne = TSNE(random_state=RS).fit_transform(pca_result_50)

    fashion_scatter(fashion_pca_tsne, y_subset)


def LearPCA():
    # Principle Component Analysis (PCA)
    # Example I
    # Load the data- Breast Cancer
    breast = load_breast_cancer()
    breast_data = breast.data

    # Extract the lables
    breast_labels = breast.target
    # Create a column with the lables
    labels = np.reshape(breast_labels, (569, 1))

    final_breast_data = np.concatenate([breast_data, labels], axis=1)

    breast_dataset = pd.DataFrame(final_breast_data)

    features = breast.feature_names
    features_labels = np.append(features, 'label')
    breast_dataset.columns = features_labels

    # Change lables to 0='Benign' and 1='Malignant'
    breast_dataset['label'].replace(0, 'Benign', inplace=True)
    breast_dataset['label'].replace(1, 'Malignant', inplace=True)

    # Data Visualization
    x = breast_dataset.loc[:, features].values
    x = StandardScaler().fit_transform(x)  # normalizing the features
    y = scale(breast_dataset.loc[:, features].values)

    # convert the normalized features into a tabular format with the help of DataFrame.
    feat_cols = ['feature' + str(i) for i in range(x.shape[1])]
    normalised_breast = pd.DataFrame(x, columns=feat_cols)
    normalised_breast.tail()

    pca_breast = PCA(n_components=3)
    principalComponents_breast = pca_breast.fit_transform(x)

    # create a DataFrame that will have the principal component values for all samples
    principal_breast_Df = pd.DataFrame(data=principalComponents_breast
                                       , columns=['principal component 1', 'principal component 2'])
    principal_breast_Df.tail()  # Return the last n rows.

    # Show for each principle component how much of the informaion it holds
    print('Explained variation per principal component: {}'.format(pca_breast.explained_variance_ratio_))

    # Plot the visualization of the two PCs
    plt.figure()
    plt.figure(figsize=(10, 10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component - 1', fontsize=20)
    plt.ylabel('Principal Component - 2', fontsize=20)
    plt.title("Principal Component Analysis of Breast Cancer Dataset", fontsize=20)
    targets = ['Benign', 'Malignant']
    colors = ['r', 'g']
    for target, color in zip(targets, colors):
        indicesToKeep = breast_dataset['label'] == target
        plt.scatter(principal_breast_Df.loc[indicesToKeep, 'principal component 1']
                    , principal_breast_Df.loc[indicesToKeep, 'principal component 2'], c=color, s=50)

    plt.legend(targets, prop={'size': 15})

    # Example II
    # Load the CIFAR data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

def Learn_Decision_Tree():
    # Learning phase
    col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
    # load dataset
    pima = pd.read_csv(r'C:\Users\ido.DM\Downloads\diabetes.csv', header=0, names=col_names)

    feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
    X = pima[feature_cols]  # Features
    y = pima.label  # Target variable

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=1)  # 70% training and 30% test

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    # Visualizing Decision Trees
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=feature_cols, class_names=['0', '1'])
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('diabetes_gini.png')
    Image(graph.create_png())

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=feature_cols, class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('diabetes_entropy.png')
    Image(graph.create_png())

if __name__ == "__main__":
    Learn_Decision_Tree()
    #Learn_tSNE()
