from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import re

'''
    Data preprocessing work for training data
'''
#   import data from csv file
fileName = '../day/train_day.csv'
inputData = pd.read_csv(fileName, header=0)
selectFeature = ['dteday', 'mnth', 'datetype', 
                'weekday', 'weathersit', 'atemp', 
                'hum', 'windspeed']

#   one-hot for the label
def oneHot(maxNum, num):
    temp = [0 for i in range(maxNum)]
    temp[num] = 1
    return np.array(temp)

#   get week-number from the date
def getWeekNum(date):
    pattern = re.compile(r"201(\d)\/(\d+)\/(\d+)")
    matchObj = re.match(pattern, date)
    MONTH_DAY = {
        "1":  0,
        "2":  31,
        "3":  59,
        "4":  90,
        "5":  120,
        "6":  151,
        "7":  181,
        "8":  212,
        "9":  243,
        "10": 273,
        "11": 304,
        "12": 334
    }
    if matchObj:
        whichYear = matchObj.group(1)
        month = matchObj.group(2)
        day = int(matchObj.group(3))
        allDays = MONTH_DAY[month] + day
        if whichYear == '2' and int(month) > 2:
            allDays += 1
        return int(allDays/7)/52
    else:
        print("error input date!!")
        return -1

#   apply to the feature that you want to convert
def convertToOneHot(data):
    data['mnth'] = list(
        map(
            lambda x: oneHot(12, int(x-1)),
            data['mnth']
        )
    )
    data['weekday'] = list(
        map(
            lambda x: oneHot(7, int(x)),
            data['weekday']
        )
    )
    data['weathersit'] = list(
        map(
            lambda x: oneHot(4, int(x-1)),
            data['weathersit']
        )
    )
    return data

#   shuffle data; get the array; split data into different set
def splitData(data):
    #   create the empty set
    column = list(data)
    trainSet = pd.DataFrame(columns = selectFeature)
    trainTarget = pd.DataFrame(columns = [column[13]])
    validSet = pd.DataFrame(columns = selectFeature)
    validTarget = pd.DataFrame(columns = [column[13]])
    #   split the data in different month
    for i in range(1,13):
        temp = data.loc[data.mnth == i]
        temp = shuffle(temp)
        T_num = round(0.8*len(temp))
        V_num = len(data) - T_num
        trainSet = trainSet.append(temp.head(T_num).ix[:, [*selectFeature]])
        trainTarget = trainTarget.append(temp.head(T_num).ix[:, [13]])
        validSet = validSet.append(temp.tail(V_num).ix[:, [*selectFeature]])
        validTarget = validTarget.append(temp.tail(V_num).ix[:, [13]])

    #   apply oneHot
    trainSet = convertToOneHot(trainSet)
    validSet = convertToOneHot(validSet)

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
        trainTarget.values,
        np.array(validSet),
        validTarget.values
    )

def deal(data):
    #   convert the dteday to weekNumber
    data['dteday'] = list(
        map(
            lambda x: getWeekNum(x),
            data['dteday']
        )
    )
    #   get the new feature -- datetype
    data.loc[((data.workingday == 0) & (data.holiday == 0)), 'datetype'] = 0
    data.loc[((data.workingday == 1) & (data.holiday == 0)), 'datetype'] = 1
    data.loc[(data.holiday == 1), 'datetype'] = 2

    dataSet_2011 = data[data.yr == 0]
    dataSet_2012 = data[data.yr == 1]

    (trainSet_2011, trainTarget_2011, validSet_2011, validTarget_2011) = splitData(dataSet_2011)

    (trainSet_2012, trainTarget_2012, validSet_2012, validTarget_2012) = splitData(dataSet_2012)

    return (trainSet_2011, trainTarget_2011, validSet_2011, validTarget_2011,
            trainSet_2012, trainTarget_2012, validSet_2012, validTarget_2012)


#   convert the label to OneHot, and split the data
(trainSet_2011, trainTarget_2011, validSet_2011, validTarget_2011,
 trainSet_2012, trainTarget_2012, validSet_2012, validTarget_2012) = deal(inputData)
#   define the model; fit the model; score of the model
#model = RandomForestRegressor(n_estimators=20, criterion="mae", max_features="sqrt")
model_2011 = GradientBoostingRegressor(n_estimators=100)
model_2012 = GradientBoostingRegressor(n_estimators=100)
model_2011.fit(trainSet_2011, trainTarget_2011)
model_2012.fit(trainSet_2012, trainTarget_2012)
print('OK\n')
print(model.score(validSet_2012, validTarget_2012))
pre_valid = model.predict(validSet_2012)
pre_train = model.predict(trainSet_2012)
print("MSE in train: "+str(mean_squared_error(trainTarget_2012, pre_train)))
print("MSE in valid: "+str(mean_squared_error(validTarget_2012, pre_valid)))
#   output the predict result and the origin validation target
'''pd.DataFrame(model.predict(validSet)).to_excel('pre_valid.xls')
pd.DataFrame(validTarget).to_excel('valid.xls')

print(trainSet_2012[0:10])
print(trainTarget_2012[0:10])'''