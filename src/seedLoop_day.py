from sklearn.ensemble import GradientBoostingRegressor
from sklearn_model_beta3 import trainDeal
from sklearn.externals import joblib
import pandas as pd
import numpy as np

trainFileName = '../day/train_day.csv'

for x in range(0,1001):
    print(x)
    inputData = pd.read_csv(trainFileName, header=0)
    (trainSet, regTrainTarget, cntTrainTarget,
    validSet, regValidTarget, cntValidTarget) = trainDeal(inputData, x)
    #   define the model; fit the model; score of the model
    regModel = GradientBoostingRegressor(n_estimators=120, learning_rate=0.15)
    cntModel = GradientBoostingRegressor(n_estimators=120, learning_rate=0.15)
    regModel.fit(trainSet, regTrainTarget)
    cntModel.fit(trainSet, cntTrainTarget)
    r1 = regModel.score(validSet, regValidTarget)
    r2 = cntModel.score(validSet, cntValidTarget)
    if r1 > 0.9 and r2 > 0.9:
        name1 = 'C:\\Users\\yqr20\\大数据分析比赛\\Bicycle-sharing-analyze\\src\\model\\reg_' + str(x) + '_' +str(r1)[0:5] + str(r2)[0:5] + '.m'
        name2 = 'C:\\Users\\yqr20\\大数据分析比赛\\Bicycle-sharing-analyze\\src\\model\\cnt_' + str(x) + '_' +str(r1)[0:5] + str(r2)[0:5] + '.m'
        joblib.dump(regModel, name1)
        joblib.dump(cntModel, name2)
        print("This is OK  --------" + str(x))

print('\n\ndone!')