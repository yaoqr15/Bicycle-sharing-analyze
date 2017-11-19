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
trainFileName = '../day/train_day.csv'
testFileName = '../day/test_day.csv'
inputData = pd.read_csv(trainFileName, header=0)
testData = pd.read_csv(testFileName, header=0)
#   selected feature for the model input
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
    data['datetype'] = list(
        map(
            lambda x: oneHot(3, int(x)),
            data['datetype']
        )
    )
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

#   method to get a suitable array for input
def mergeList(inputList):
    tmp =[]
    for x in inputList:
        if type(x) is np.ndarray:
            tmp += list(x)
        else:
            tmp.append(x)
    return tmp

#   shuffle data; get the array; split data into different set
def splitData(data):
    #   create the empty set
    column = list(data)
    trainSet = pd.DataFrame(columns = selectFeature)
    trainTarget_reg = pd.DataFrame(columns = [column[14]])
    trainTarget_cnt = pd.DataFrame(columns = [column[15]])
    validSet = pd.DataFrame(columns = selectFeature)
    validTarget_reg = pd.DataFrame(columns = [column[14]])
    validTarget_cnt = pd.DataFrame(columns = [column[15]])
    #   split the data in different month
    for i in range(1,13):
        temp = data.loc[data.mnth == i]
        temp = shuffle(temp)
        T_num = round(0.8*len(temp))
        V_num = len(data) - T_num
        trainSet = trainSet.append(temp.head(T_num).ix[:, [*selectFeature]])
        trainTarget_reg = trainTarget_reg.append(temp.head(T_num).ix[:, [14]])
        trainTarget_cnt = trainTarget_cnt.append(temp.head(T_num).ix[:, [15]])
        validSet = validSet.append(temp.tail(V_num).ix[:, [*selectFeature]])
        validTarget_reg = validTarget_reg.append(temp.tail(V_num).ix[:, [14]])
        validTarget_cnt = validTarget_cnt.append(temp.tail(V_num).ix[:, [15]])

    #   apply oneHot
    trainSet = convertToOneHot(trainSet)
    validSet = convertToOneHot(validSet)

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
        trainTarget_reg.values.ravel(),
        trainTarget_cnt.values.ravel(),
        np.array(validSet),
        validTarget_reg.values.ravel(),
        validTarget_cnt.values.ravel()
    )

def trainDeal(data):
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

    dataSet_2011 = data[data.yr == 0].copy()
    dataSet_2012 = data[data.yr == 1].copy()

    (trainSet_2011, regTrainTarget_2011, cntTrainTarget_2011,
     validSet_2011, regValidTarget_2011, cntValidTarget_2011) = splitData(dataSet_2011)

    (trainSet_2012, regTrainTarget_2012, cntTrainTarget_2012,
     validSet_2012, regValidTarget_2012, cntValidTarget_2012) = splitData(dataSet_2011)

    return (trainSet_2011, regTrainTarget_2011, cntTrainTarget_2011,
            validSet_2011, regValidTarget_2011, cntValidTarget_2011,
            trainSet_2012, regTrainTarget_2012, cntTrainTarget_2012,
            validSet_2012, regValidTarget_2012, cntValidTarget_2012)

def testDeal(data):
    data = data.copy()
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

    #   split data by year
    dataSet_2011 = data[data.yr == 0].copy()
    dataSet_2012 = data[data.yr == 1].copy()
    #   apply oneHot, and select feature
    dataSet_2011 = convertToOneHot(dataSet_2011)
    dataSet_2011 = dataSet_2011.loc[:, [*selectFeature]]
    dataSet_2012 = convertToOneHot(dataSet_2012)
    dataSet_2012 = dataSet_2012.loc[:, [*selectFeature]]
    #   generate the proper array
    dataSet_2011 = list(
        map(
            lambda x: mergeList(x),
            dataSet_2011.values.tolist()
        )
    )
    dataSet_2012 = list(
        map(
            lambda x: mergeList(x),
            dataSet_2012.values.tolist()
        )
    )
    #   return the four data set
    return (np.array(dataSet_2011), np.array(dataSet_2012))

#   split the train data
(trainSet_2011, regTrainTarget_2011, cntTrainTarget_2011,
 validSet_2011, regValidTarget_2011, cntValidTarget_2011,
 trainSet_2012, regTrainTarget_2012, cntTrainTarget_2012,
 validSet_2012, regValidTarget_2012, cntValidTarget_2012) = trainDeal(inputData)

#   define the model; fit the model; score of the model
regModel_2011 = GradientBoostingRegressor(n_estimators=100)
cntModel_2011 = GradientBoostingRegressor(n_estimators=100)
regModel_2012 = GradientBoostingRegressor(n_estimators=100)
cntModel_2012 = GradientBoostingRegressor(n_estimators=100)

regModel_2011.fit(trainSet_2011, regTrainTarget_2011)
cntModel_2011.fit(trainSet_2011, cntTrainTarget_2011)
regModel_2012.fit(trainSet_2012, regTrainTarget_2012)
cntModel_2012.fit(trainSet_2012, cntTrainTarget_2012)

print('OK\n')

print(regModel_2011.score(validSet_2011, regValidTarget_2011))
reg_pre_valid_2011 = regModel_2011.predict(validSet_2011)
reg_pre_train_2011 = regModel_2011.predict(trainSet_2011)
print("MSE in train for 2011 reg: "+str(mean_squared_error(regTrainTarget_2011, reg_pre_train_2011)))
print("MSE in valid for 2011 reg: "+str(mean_squared_error(regValidTarget_2011, reg_pre_valid_2011)))
print('\n')

print(cntModel_2011.score(validSet_2011, cntValidTarget_2011))
cnt_pre_valid_2011 = cntModel_2011.predict(validSet_2011)
cnt_pre_train_2011 = cntModel_2011.predict(trainSet_2011)
print("MSE in train for 2011 cnt: "+str(mean_squared_error(cntTrainTarget_2011, cnt_pre_train_2011)))
print("MSE in valid for 2011 cnt: "+str(mean_squared_error(cntValidTarget_2011, cnt_pre_valid_2011)))
print('\n')

print(regModel_2012.score(validSet_2012, regValidTarget_2012))
reg_pre_valid_2012 = regModel_2012.predict(validSet_2012)
reg_pre_train_2012 = regModel_2012.predict(trainSet_2012)
print("MSE in train for 2012 reg: "+str(mean_squared_error(regTrainTarget_2012, reg_pre_train_2012)))
print("MSE in valid for 2012 reg: "+str(mean_squared_error(regValidTarget_2012, reg_pre_valid_2012)))
print('\n')

print(cntModel_2012.score(validSet_2012, cntValidTarget_2012))
cnt_pre_valid_2012 = cntModel_2012.predict(validSet_2012)
cnt_pre_train_2012 = cntModel_2012.predict(trainSet_2012)
print("MSE in train for 2012 cnt: "+str(mean_squared_error(cntTrainTarget_2012, cnt_pre_train_2012)))
print("MSE in valid for 2012 cnt: "+str(mean_squared_error(cntValidTarget_2012, cnt_pre_valid_2012)))

#   get the test data
testData_2011, testData_2012 = testDeal(testData)
#   output test predict
pre_reg_test_2011 = regModel_2011.predict(testData_2011)
pre_cnt_test_2011 = cntModel_2011.predict(testData_2011)
pre_reg_test_2012 = regModel_2012.predict(testData_2012)
pre_cnt_test_2012 = cntModel_2012.predict(testData_2012)

print(cntTrainTarget_2012)
'''
testData.loc[(testData.yr == 0), 'registered'] = pre_reg_test_2011
testData.loc[(testData.yr == 0), 'cnt'] = pre_cnt_test_2011
testData.loc[(testData.yr == 1), 'registered'] = pre_reg_test_2012
testData.loc[(testData.yr == 1), 'cnt'] = pre_cnt_test_2012
testData.loc[:, 'casual'] = testData['cnt'] - testData['registered']
print(testData.head(20))

#   output the predict result and the origin validation target
pd.DataFrame(model.predict(validSet)).to_excel('pre_valid.xls')
pd.DataFrame(validTarget).to_excel('valid.xls')

print(trainSet_2012[0:10])
print(trainTarget_2012[0:10])'''