import pandas as pd
import numpy as np
from datetime import datetime
import glob
import ntpath


def mainProcess (allFiles, imageType):
    for ind, cFile in enumerate(allFiles):
        try:
            print(str(ind + 1) + '/' + str(len(allFiles)))
            currecntData = pd.read_excel(cFile)
            currecntData.sort_values('Date_Time', axis=0, ascending=True, inplace=True)
            currecntData=currecntData.dropna(thresh=10)
            currecntData = currecntData.set_index(pd.to_datetime(currecntData['Time']))
            findTrialPhase(currecntData, imageType)

            currentFileName = ntpath.basename(cFile)

            if 'left' in currentFileName:
                x=1

            elif 'right' in currentFileName:
                x=1

            else:
                print('hand side not found  :  ' + currentFileName)
                quit()
        except:
            print('error in: ' + cFile)
            quit()

def findTrialPhase (currecntData, imageType):

    currecntData['trial_phase']=None


    if 'jpg' in imageType:
        dTimesFirst=currecntData.index-currecntData.index[0]
        phaseChangeId = dTimesFirst > pd.Timedelta(minutes=10)
        dTimes = currecntData.index.to_series().diff()

    elif 'csv' in imageType:
        currecntData


if __name__ == '__main__':
    imageType = 'jpg'

    characteristicsPath = r'C:\Users\ido\Google Drive\Thesis\Data\Characteristics.xlsx'
    charectaristicsPD = pd.read_excel(characteristicsPath)


    imagesDataPath = r'C:\Users\ido\Google Drive\Thesis\Data\Processed Data\JPG Feature tables\\'
    allFiles = glob.glob(imagesDataPath + '*.xlsx')
    mainProcess(allFiles, imageType)