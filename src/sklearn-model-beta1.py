from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

'''
    Data preprocessing work for training data
'''
#   import data from csv file
fileName = '../day/train_day.csv'
inputData = pd.read_csv(fileName, header=0)

#   one-hot for the label
def oneHot(maxNum, num):
    temp = [0 for i in range(maxNum)]
    temp[num] = 1
    return np.array(temp)

def convertToOneHot(data):
    data['season'] = list(
        map(
            lambda x: oneHot(4, x-1),
            data['season']
        )
    )
    data['mnth'] = list(
        map(
            lambda x: oneHot(12, x-1),
            data['mnth']
        )
    )
    data['weekday'] = list(
        map(
            lambda x: oneHot(7, x),
            data['weekday']
        )
    )
    data['weathersit'] = list(
        map(
            lambda x: oneHot(4, x-1),
            data['weathersit']
        )
    )
    return data

#   shuffle data; get the array; split data into different set
def splitData(data):
    data = shuffle(data, random_state=999)
    trainNum = round(0.8*len(data))
    validNum = len(data) - trainNum
    #   get the four set
    trainSet = data.head(trainNum).ix[:,2:13]
    trainTarget = data.head(trainNum).ix[:, [13,14]].values
    validSet = data.tail(validNum).ix[:, 2:13]
    validTarget = data.tail(validNum).ix[:, [13,14]].values

    #   method to get a suitable array for input
    def mergeList(inputList):
        tmp =[]
        for x in inputList:
            if type(x) is np.ndarray:
                tmp += list(x)
            else:
                tmp.append(x)
        return tmp
    #   generate the proper array
    trainSet = list(
        map(
            lambda x: mergeList(x),
            trainSet.values.tolist()
        )
    )
    validSet = list(
        map(
            lambda x: mergeList(x),
            validSet.values.tolist()
        )
    )
    #   return the four data set
    return (
        np.array(trainSet),
        trainTarget,
        np.array(validSet),
        validTarget
    )

#   convert the label to OneHot, and split the data
trainSet, trainTarget, validSet, validTarget = splitData(convertToOneHot(inputData))#''''''

#   define the model; fit the model; score of the model
#model = RandomForestRegressor(n_estimators=20, criterion="mae", max_features="sqrt")
model = ExtraTreesRegressor(n_estimators=20, criterion="mse", max_features="sqrt")
model.fit(trainSet, trainTarget)
print('OK\n')
print(model.score(validSet, validTarget))
pre_valid = model.predict(validSet)
print("MSE: "+str(mean_squared_error(validTarget, pre_valid)))
#   output the predict result and the origin validation target
pd.DataFrame(model.predict(validSet)).to_excel('pre_valid.xls')
pd.DataFrame(validTarget).to_excel('valid.xls')
