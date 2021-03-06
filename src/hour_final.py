# -*- coding: utf-8 -*-
"""
hour最后一版

Created on Sun Nov 19 16:43:04 2017

@author: 姚乔日,宋丹明，陈品
"""
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import re

'''

    Data preprocessing work for training data

'''
#   import data from csv file
fileName = '../hour/train_hour.csv'
inputData = pd.read_csv(fileName, header=0)

testFileName = '../hour/test_hour.csv'
testData = pd.read_csv(testFileName, header=0) 

selectFeature = ['dteday', 'mnth', 'hr', 'datetype', 
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
    '''
        trainTarget_casual = pd.DataFrame(columns = [column[16]])
        上面这一行是之前的错误的代码，我们仅仅修改了column中的索引数字
    '''
    trainTarget_casual = pd.DataFrame(columns = [column[14]])
    trainTarget_registered = pd.DataFrame(columns = [column[15]])
    validSet = pd.DataFrame(columns = selectFeature) 
    '''
        validTarget_casual = pd.DataFrame(columns = [column[16]])
        这里和187行是一样的错误
    '''
    validTarget_casual = pd.DataFrame(columns = [column[14]])
    validTarget_registered = pd.DataFrame(columns = [column[15]])
   
    for i in range(0,24):
        temp = data.loc[data.hr == i]
        temp = shuffle(temp)
        T_num = round(0.8*len(temp))
        V_num = len(temp) - T_num

        trainSet = trainSet.append(temp.head(T_num).ix[:, [*selectFeature]])    
        trainTarget_registered = trainTarget_registered.append(temp.head(T_num).ix[:, [15]])
        '''
            trainTarget_casual = trainTarget_casual.append(temp.head(T_num).ix[:, [16]])
            这里和187行是一样的错误
        '''
        trainTarget_casual = trainTarget_casual.append(temp.head(T_num).ix[:, [14]])
        validSet = validSet.append(temp.tail(V_num).ix[:, [*selectFeature]])
        validTarget_registered = validTarget_registered.append(temp.tail(V_num).ix[:, [15]])
        '''
            validTarget_casual = validTarget_casual.append(temp.tail(V_num).ix[:, [16]])
            这里和187行是一样的错误
        '''
        validTarget_casual = validTarget_casual.append(temp.tail(V_num).ix[:, [14]])
    
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
        trainTarget_registered.values.ravel(),       
        trainTarget_casual.values.ravel(),
        np.array(validSet),
        validTarget_registered.values.ravel(),      
        validTarget_casual.values.ravel()
    )

def deal(data):
    #   convert the dteday to weekNumber
    data['dteday'] = list(
        map(
            lambda x: getWeekNum(x),
            data['dteday']
        )
    )   
    #   data = hourConvert(data)
    #   get the new feature -- datetype
    data.loc[((data.workingday == 0) & (data.holiday == 0)), 'datetype'] = 0
    data.loc[((data.workingday == 1) & (data.holiday == 0)), 'datetype'] = 1
    data.loc[(data.holiday == 1), 'datetype'] = 2
    dataSet_2011 = data[data.yr == 0]
    dataSet_2012 = data[data.yr == 1]
   
    (trainSet_2011, trainTarget_2011_registered, trainTarget_2011_casual,
     validSet_2011, validTarget_2011_registered, validTarget_2011_casual) = splitData(dataSet_2011)

    (trainSet_2012, trainTarget_2012_registered, trainTarget_2012_casual,
     validSet_2012, validTarget_2012_registered, validTarget_2012_casual) = splitData(dataSet_2012)

    return (trainSet_2011, trainTarget_2011_registered, trainTarget_2011_casual,
            validSet_2011, validTarget_2011_registered, validTarget_2011_casual,
            trainSet_2012, trainTarget_2012_registered, trainTarget_2012_casual,
            validSet_2012, validTarget_2012_registered, validTarget_2012_casual)

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

#   convert the label to OneHot, and split the data

(trainSet_2011, trainTarget_2011_registered, trainTarget_2011_casual,
 validSet_2011, validTarget_2011_registered, validTarget_2011_casual,
 trainSet_2012, trainTarget_2012_registered, trainTarget_2012_casual,
 validSet_2012, validTarget_2012_registered, validTarget_2012_casual) = deal(inputData)

#   define the model; fit the model; score of the model
model_2011_registered = RandomForestRegressor(n_estimators = 100)
model_2011_registered.fit(trainSet_2011, trainTarget_2011_registered)
model_2011_casual = RandomForestRegressor(n_estimators = 100)
model_2011_casual.fit(trainSet_2011, trainTarget_2011_casual)

model_2012_registered = RandomForestRegressor(n_estimators = 100)
model_2012_registered.fit(trainSet_2012, trainTarget_2012_registered)
model_2012_casual = RandomForestRegressor(n_estimators = 100)
model_2012_casual.fit(trainSet_2012, trainTarget_2012_casual)

print('hour_data')
print('R^2 in 2011_registered:'+str(model_2011_registered.score(validSet_2011, validTarget_2011_registered)))
print('R^2 in 2011_casual    :'+str(model_2011_casual.score(validSet_2011, validTarget_2011_casual)))
print('R^2 in 2012_registered:'+str(model_2012_registered.score(validSet_2012, validTarget_2012_registered)))
print('R^2 in 2012_casual    :'+str(model_2012_casual.score(validSet_2012, validTarget_2012_casual)))

pre_train_2011_casual = model_2011_casual.predict(trainSet_2011)
pre_train_2011_registered = model_2011_registered.predict(trainSet_2011)
pre_train_2012_casual = model_2012_casual.predict(trainSet_2012)
pre_train_2012_registered = model_2012_registered.predict(trainSet_2012)

pre_valid_2011_casual = model_2011_casual.predict(validSet_2011)
pre_valid_2011_registered = model_2011_registered.predict(validSet_2011)
pre_valid_2012_casual = model_2012_casual.predict(validSet_2012)
pre_valid_2012_registered = model_2012_registered.predict(validSet_2012)

print("MSE in train 2011 casual    : "+str(mean_squared_error(trainTarget_2011_casual, pre_train_2011_casual)))
print("MSE in train 2011 registered: "+str(mean_squared_error(trainTarget_2011_registered, pre_train_2011_registered)))
print("MSE in train 2012 casual    : "+str(mean_squared_error(trainTarget_2012_casual, pre_train_2012_casual)))
print("MSE in train 2012 registered: "+str(mean_squared_error(trainTarget_2012_registered, pre_train_2012_registered)))

print("MSE in valid 2011 casual    : "+str(mean_squared_error(validTarget_2011_casual, pre_valid_2011_casual)))
print("MSE in valid 2011 registered: "+str(mean_squared_error(validTarget_2011_registered, pre_valid_2011_registered)))
print("MSE in valid 2012 casual    : "+str(mean_squared_error(validTarget_2012_casual, pre_valid_2012_casual)))
print("MSE in valid 2012 registered: "+str(mean_squared_error(validTarget_2012_registered, pre_valid_2012_registered)))


testData_2011, testData_2012 = testDeal(testData) 

test_2011_casual = model_2011_casual.predict(testData_2011)
test_2012_casual = model_2012_casual.predict(testData_2012)
test_2011_registered = model_2011_registered.predict(testData_2011)
test_2012_registered = model_2012_registered.predict(testData_2012)

testData.loc[(testData.yr == 0), 'casual'] = test_2011_casual
testData.loc[(testData.yr == 1), 'casual'] = test_2012_casual
testData.loc[(testData.yr == 0), 'registered'] = test_2011_registered
testData.loc[(testData.yr == 1), 'registered'] = test_2012_registered
testData.loc[(testData.yr == 0), 'cnt'] = test_2011_casual + test_2011_registered
testData.loc[(testData.yr == 1), 'cnt'] = test_2012_casual + test_2012_registered
testData.to_excel('new_hour_1.xls')

print('OK\n')