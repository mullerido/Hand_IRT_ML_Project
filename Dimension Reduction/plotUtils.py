import matplotlib.pyplot as plt
import numpy as np

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