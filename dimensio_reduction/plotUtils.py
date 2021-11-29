import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plotimage(image, cmp=None, savePath=""):
    plt.figure()
    plt.imshow(image, cmap=cmp)
    if savePath:
        plt.savefig(savePath)
        plt.close("all")

def plot_comparison(original, filtered, filter_name, savePath=""):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    fig.show()

    if savePath:
        plt.savefig(savePath)
        plt.close("all")

def PlotSecondaryAxis(x1, y1, x2, y2, savePath=""):

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('radius of structuring elements')
    ax1.set_ylabel('exp', color=color)
    ax1.plot(x1, y1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylabel('Area in numbers of pixels')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
    ax2.plot(x2, y2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylabel('dA/dr')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    if savePath:
        plt.savefig(savePath)
        plt.close("all")

def plotOpeningOverTime(foreGroundPixels, rad, savePath=""):

    ForeGroundDiff = np.diff(foreGroundPixels[1:])
    PlotSecondaryAxis(rad, foreGroundPixels, rad[1:-1], -ForeGroundDiff)
    if savePath:
        plt.savefig(savePath)
        plt.close("all")

def plot_areas_by_groups_pca_dim_single_fid(groupedFeature, subject_id, groups, Legend, normalize_flag):
    rCol = []
    for ind in range(len(subject_id)):
        rCol.append('C' + str(ind))

    colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    # plot all on the same figure
    fig, ax = plt.subplots()
    plt.title('Group Comparison', fontsize=20)
    plt.ylabel('Ratio', fontsize=12)
    plt.xlabel("Index of time", fontsize=12)
    for ind, groupInds in enumerate(groups):
        if groupInds:
            tempDF = pd.DataFrame()
            tempDF = groupedFeature.iloc[groupInds[0], :-1]
            tempDFVal = tempDF.values

            if not normalize_flag:
                for i in range(tempDFVal.shape[0]):
                    tempDFVal[i, :] = (tempDFVal[i, :] - tempDFVal[i, 0]) / tempDFVal[i, 0]
            groupMeans = tempDFVal.mean(axis=0)
            groupStd = tempDFVal.std(axis=0)
            ax.plot(groupMeans, color=colors[ind])
            x = np.linspace(0, len(groupStd) - 1, len(groupStd))
            ax.fill_between(x, groupMeans - groupStd, groupMeans + groupStd, color=rCol[ind], alpha=0.3)
    plt.legend(Legend[0:ind + 1])
    plt.show()

def plot_areas_by_groups_pca_dim_multiple_fid(groupedFeature, subject_id, groups, Legend, normalize_flag):
    rCol = []
    for ind in range(len(subject_id)):
        rCol.append('C' + str(ind))

    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    # plot all on different figures
    fig, ax = plt.subplots(np.shape(groups)[0], 1)
    fig.suptitle('Group Comparison', fontsize=20)

    plt.xlabel("Index of time", fontsize=12)
    for ind, groupInds in enumerate(groups):
        if groupInds:
            tempDF = pd.DataFrame()
            tempDF = groupedFeature.iloc[groupInds[0], :-1]
            tempDFVal = tempDF.values
            dataByTimes = np.resize(tempDFVal, (tempDFVal.shape[0] * 12, 39))

            if not normalize_flag:
                for i in range(dataByTimes.shape[0]):
                    dataByTimes[i, :] = (dataByTimes[i, :] - dataByTimes[i, 0]) / dataByTimes[i, 0]
            groupMeans = dataByTimes.mean(axis=0)
            groupStd = dataByTimes.std(axis=0)
            ax[ind].plot(groupMeans, color=colors[ind])
            x = np.linspace(0, len(groupStd) - 1, len(groupStd))
            ax[ind].fill_between(x, groupMeans - groupStd, groupMeans + groupStd, color=rCol[ind], alpha=0.3)
            ax[ind].set(xlabel="Index of time", ylabel="Ratio", title=Legend[ind])
            ax[ind].set_ylim([-0.4, 0.8])
    plt.show()

def plot_subplots_bars(data, title, x_label, y_label):
    x=1