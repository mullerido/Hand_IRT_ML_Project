from sklearn.datasets import load_breast_cancer
from keras.datasets import cifar10
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, scale# use to normalize the data
from sklearn.decomposition import PCA# Use to perform the PCA transform
import matplotlib.pyplot as plt
import os
import re
import seaborn as sns

folder = r'C:\Projects\Thesis\Feature-Analysis- matlab\Resize Feature\\'

files = os.listdir(folder)
files_xls = [f for f in files if f[-10:] == 'right.xlsx']
allFeatures = ['Thumbs_dist_Intence', 'Thumbs_proxy_Intence', 'Index_dist_Intence', 'Index_proxy_Intence',
                    'Middle_dist_Intence', 'Middle_proxy_Intence', 'Ring_dist_Intence', 'Ring_proxy_Intence',
                    'Pinky_dist_Intence', 'Pinky_proxy_Intence','Palm_arch_Intence', 'Palm_Center_Intence']
relevantFeatures=allFeatures[1]
groupedFeature = pd.DataFrame()
names = []

normlizeFlag = False
for file in files_xls:
    print(file)
    currentFile = folder+file
    currentFeatures = pd.read_excel(currentFile)
    if normlizeFlag:#Use the ratio
        relevantCols = (currentFeatures.loc[0:,allFeatures]- currentFeatures.loc[0,allFeatures])/currentFeatures.loc[0,allFeatures]
    else:#Use the actual values
        #relevantCols = currentFeatures.filter(regex='Intence')
        relevantCols = currentFeatures.loc[:,relevantFeatures]
    transformData=relevantCols.values.ravel('F')
    transformDataFrame=pd.DataFrame(transformData).T
    indName = file[:-5]
    transformDataFrame = transformDataFrame.rename(index={0:indName})
    groupedFeature = groupedFeature.append(transformDataFrame)
    names.append(indName[:-6])

# Data Visualization
x = groupedFeature.values
#x_norm = StandardScaler().fit_transform(x) # normalizing the features
min_max_scaler = MinMaxScaler()
x_norm = min_max_scaler.fit_transform(x)
y=scale(groupedFeature.values)

groupedFeature['label']='n'

# convert the normalized features into a tabular format with the help of DataFrame.
feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
normalised_Data = pd.DataFrame(x_norm,columns=feat_cols)
normalised_Data.tail()

pca_Data = PCA(n_components=2)
principalComponents = pca_Data.fit_transform(x)

# create a DataFrame that will have the principal component values for all samples
principal_breast_Df = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
principal_breast_Df.tail()#Return the last n rows.

# Show for each principle component how much of the informaion it holds
print('Explained variation per principal component: {}'.format(pca_Data.explained_variance_ratio_))

# Plot the visualization of the two PCs
#plt.subplot()
plt.figure(figsize=(10, 10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1', fontsize=20)
plt.ylabel('Principal Component - 2', fontsize=20)
plt.title("Principal Component Analysis of All Inteneces- MIn_Max Normalization", fontsize=20)
plt.scatter(principal_breast_Df.loc[:, 'principal component 1'],
            principal_breast_Df.loc[:, 'principal component 2'],
            c = 'g', s = 50)
for i, txt in enumerate(names):
    plt.text(principal_breast_Df.loc[i, 'principal component 1'], principal_breast_Df.loc[i, 'principal component 2'], txt)

#plt.show()

# Start plot lines by groups
#x_thresh = input('Enter X threshold: ')
x_thresh = 0 #int(x_thresh) if x_thresh else None
y_thresh =None #input('Enter Y threshold: ')
#y_thresh = int(y_thresh) if y_thresh else None

groups = [[], [], [], []]
if not(x_thresh == None) and not (y_thresh == None):
    choice = np.logical_and(np.greater_equal(principalComponents[:,0], x_thresh), np.greater_equal(principalComponents[:,1], y_thresh))
    groups[0] = np.where(choice)
    choice = np.logical_and(np.greater_equal(principalComponents[:, 0], x_thresh), np.less(principalComponents[:, 1], y_thresh))
    groups[1] = np.where(choice)
    choice = np.logical_and(np.less(principalComponents[:, 0], x_thresh), np.less(principalComponents[:, 1], y_thresh))
    groups[2] = np.where(choice)
    choice = np.logical_and(np.less(principalComponents[:, 0], x_thresh), np.greater_equal(principalComponents[:, 1], y_thresh))
    groups[3] = np.where(choice)
    Legend = ['1Qtr', '2Qtr', '3Qtr', '4Qtr']

elif not(x_thresh==None) and (y_thresh == None):
    choice = np.greater_equal(principalComponents[:, 0], x_thresh)
    groups[0] = np.where(choice)
    choice = np.less(principalComponents[:, 0], x_thresh)
    groups[1] = np.where(choice)
    Legend = ['X_Possitive', 'X_Negative']

elif (x_thresh == None) and not (y_thresh == None):
    choice = np.greater_equal(principalComponents[:, 0], y_thresh)
    groups[0] = np.where(choice)
    choice = np.less(principalComponents[:, 0], y_thresh)
    groups[1] = np.where(choice)
    Legend = ['Y_Possitive', 'Y_Negative']

colors = ['C0', 'C1', 'C2', 'C3', 'C4']

fig, ax = plt.subplots()
plt.xlabel('Group Comparison', fontsize=20)
plt.ylabel('Ratio', fontsize=12)
plt.title("Index of time", fontsize=12)
for ind, groupInds in enumerate(groups):
    if groupInds:
        tempDF = pd.DataFrame()
        tempDF = groupedFeature.iloc[groupInds[0], :-1]
        tempDFVal = tempDF.values
        dataByTimes = np.resize(tempDFVal, (tempDFVal.shape[0]*12, 39))

        if not(normlizeFlag):
            for i in range(dataByTimes.shape[0]):
                dataByTimes[i,:] = (dataByTimes[i, :]-dataByTimes[i, 0])/ dataByTimes[i, 0]
        groupMeans = dataByTimes.mean(axis=0)
        groupStd = dataByTimes.std(axis=0)
        ax.plot(groupMeans,color=colors[ind])
        x = np.linspace(0, len(groupStd) - 1, len(groupStd))
        ax.fill_between(x, groupMeans - groupStd, groupMeans + groupStd, color=colors[ind], alpha=0.3)
plt.legend(Legend[0:ind+1])
plt.show()
        #for s in groupInds[0]:
        #    groupedFeature.iloc[s, -1]=ind
        #    tempDF = groupedFeature.iloc[groupInds[0], :-1]
        #    groupMeans = tempDF.mean(axis=0)
        #    groupStd = tempDF.std(axis=0)
        #    fig, ax = plt.subplots()
        #    x=np.linspace(0,len(groupStd)-1,len(groupStd))
        #    ax.fill_between(x, groupMeans-groupStd, groupMeans+groupStd)
        #    plt.show()
            #sns.lineplot(data=groupedFeature, hue='label')
            #sns.relplot(data)
            #sns.lineplot(data=tempDF.transpose(), hue='')
            #plt.show()
x=1
