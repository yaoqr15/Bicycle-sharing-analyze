from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import re

'''
    Data preprocessing work for training data
'''
#   import data from csv file
trainFileName = '../day/train_day.csv'
inputData = pd.read_csv(trainFileName, header=0)
#   selected feature for the model input
selectFeature = [ 'mnth', 'yr','yearbin',
                 'weekday', 'datetype', 'weathersit', 
                 'dteday', 'atemp', 'hum', 'windspeed']

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

def getYearBin(data):
    data.loc[(data.yr==0), 'yearbin'] = 1
    data.loc[((data.yr==0) & (data.mnth>3)), 'yearbin'] = 2
    data.loc[((data.yr==0) & (data.mnth>6)), 'yearbin'] = 3
    data.loc[((data.yr==0) & (data.mnth>9)), 'yearbin'] = 4
    data.loc[(data.yr==1), 'yearbin'] = 5
    data.loc[((data.yr==1) & (data.mnth>3)), 'yearbin'] = 6
    data.loc[((data.yr==1) & (data.mnth>6)), 'yearbin'] = 7
    data.loc[((data.yr==1) & (data.mnth>9)), 'yearbin'] = 8
    return data

#   apply to the feature that you want to convert
def convertToOneHot(data):
    data['yearbin'] = list(
        map(
            lambda x: oneHot(8, int(x-1)),
            data['yearbin']
        )
    )
    data['yr'] = list(
        map(
            lambda x: oneHot(2, int(x)),
            data['yr']
        )
    )
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
def splitData(data, seed):
    #   create the empty set
    column = list(data)
    trainSet = pd.DataFrame(columns = selectFeature)
    trainTarget_reg = pd.DataFrame(columns = [column[14]])
    trainTarget_cas = pd.DataFrame(columns = [column[13]])
    validSet = pd.DataFrame(columns = selectFeature)
    validTarget_reg = pd.DataFrame(columns = [column[14]])
    validTarget_cas = pd.DataFrame(columns = [column[13]])
    #   split the data in different month
    for i in range(1,13):
        temp = data.loc[data.mnth == i]
        temp = shuffle(temp, random_state=seed)
        T_num = round(0.8*len(temp))
        V_num = len(temp) - T_num
        trainSet = trainSet.append(temp.head(T_num).ix[:, [*selectFeature]])
        trainTarget_reg = trainTarget_reg.append(temp.head(T_num).ix[:, [14]])
        trainTarget_cas = trainTarget_cas.append(temp.head(T_num).ix[:, [13]])
        validSet = validSet.append(temp.tail(V_num).ix[:, [*selectFeature]])
        validTarget_reg = validTarget_reg.append(temp.tail(V_num).ix[:, [14]])
        validTarget_cas = validTarget_cas.append(temp.tail(V_num).ix[:, [13]])

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
        trainTarget_cas.values.ravel(),
        np.array(validSet),
        validTarget_reg.values.ravel(),
        validTarget_cas.values.ravel()
    )

def trainDeal(data, seed):
    #   convert the dteday to weekNumber
    data['dteday'] = list(
        map(
            lambda x: getWeekNum(x),
            data['dteday']
        )
    )
    #   get year bins
    getYearBin(data)
    #   get the new feature -- datetype
    data.loc[((data.workingday == 0) & (data.holiday == 0)), 'datetype'] = 0
    data.loc[((data.workingday == 1) & (data.holiday == 0)), 'datetype'] = 1
    data.loc[(data.holiday == 1), 'datetype'] = 2

    (trainSet, regTrainTarget, casTrainTarget,
     validSet, regValidTarget, casValidTarget) = splitData(data, seed)

    return (trainSet, regTrainTarget, casTrainTarget,
            validSet, regValidTarget, casValidTarget)


'''
    the code that define the model and fit the model
'''
#   follow code will not run when you import it as a module
if __name__ == '__main__':
    #   split the train data
    (trainSet, regTrainTarget, casTrainTarget,
    validSet, regValidTarget, casValidTarget) = trainDeal(inputData, 591)
    #   define the model and fit 
    regModel = GradientBoostingRegressor(n_estimators=120, learning_rate=0.15)
    casModel = GradientBoostingRegressor()
    regModel.fit(trainSet, regTrainTarget)
    casModel.fit(trainSet, casTrainTarget)

    print('OK\n')

    r1 = regModel.score(validSet, regValidTarget)
    print(r1)
    reg_pre_valid = regModel.predict(validSet)
    reg_pre_train = regModel.predict(trainSet)
    print("MSE in train for reg: "+str(mean_squared_error(regTrainTarget, reg_pre_train)))
    print("MSE in valid for reg: "+str(mean_squared_error(regValidTarget, reg_pre_valid)))
    print('\n')

    r2 = casModel.score(validSet, casValidTarget)
    print(r2)
    cas_pre_valid = casModel.predict(validSet)
    cas_pre_train = casModel.predict(trainSet)
    print("MSE in train for cas: "+str(mean_squared_error(casTrainTarget, cas_pre_train)))
    print("MSE in valid for cas: "+str(mean_squared_error(casValidTarget, cas_pre_valid)))
    print('\n')

    #   save the  casual model
    name2 = 'C:\\Users\\yqr20\\大数据分析比赛\\Bicycle-sharing-analyze\\src\\model\\cas_' + str(591) + '_' +str(r1)[0:5] + str(r2)[0:5] + '.m'
    joblib.dump(casModel, name2)