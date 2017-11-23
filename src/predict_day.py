from day_sklearn_model_final import getWeekNum, getYearBin, convertToOneHot, selectFeature, mergeList, splitData
from sklearn.externals import joblib
from datetime import datetime
import numpy as np
import pandas as pd

def testDeal(data):
    data = data.copy()
    data['dteday'] = list(
        map(
            lambda x: getWeekNum(x),
            data['dteday']
        )
    )
    getYearBin(data)
    #   get the new feature -- datetype
    data.loc[((data.workingday == 0) & (data.holiday == 0)), 'datetype'] = 0
    data.loc[((data.workingday == 1) & (data.holiday == 0)), 'datetype'] = 1
    data.loc[(data.holiday == 1), 'datetype'] = 2

    #   apply oneHot, and select feature
    data = convertToOneHot(data)
    data = data.loc[:, [*selectFeature]]

    #   generate the proper array
    data = list(
        map(
            lambda x: mergeList(x),
            data.values.tolist()
        )
    )
  
    #   return the four data set
    return np.array(data)

#   import data from csv file
testFileName = '../day/test_day.csv'
testData = pd.read_csv(testFileName, header=0)
#   get the test data
inputArray = testDeal(testData)
#   load model from train result
regModel = joblib.load("./model/reg_591_0.9350.926.m")
casModel = joblib.load("./model/cas_591_0.9360.901.m")
#   output test predict
pre_reg_test = regModel.predict(inputArray)
pre_cas_test = casModel.predict(inputArray)

testData.loc[:, 'registered'] = pre_reg_test
testData.loc[:, 'casual'] = pre_cas_test
testData.loc[:, 'cnt'] = testData['casual'] + testData['registered']

#   output the predict result and the origin validation target
outputName = "./output/output_"+str(datetime.now())[:19].replace(' ', '_').replace(':', '_') + ".xls"
testData.to_excel(outputName)
print("Output finish!")