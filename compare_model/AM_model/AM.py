from pandas import Series,concat
import pandas as pd
from matplotlib import pyplot
from pandas.plotting import lag_plot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error,mean_absolute_error
from IPython.core.pylabtools import figsize
import numpy as np
#series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
speed_train = open("/home/sunyanru19s/solar_wind_coding/LSTM/data/speed_train.txt")
speed_val = open("/home/sunyanru19s/solar_wind_coding/LSTM/data/speed_val(2016).txt")

# create lagged dataset
train_speed_list = []
val_speed_list = []
speed_line = speed_train.readline()
while speed_line:
    speed = list(map(float, speed_line.split()))
    train_speed_list.append(speed)
    speed_line = speed_train.readline()
train_speed_list = np.array(train_speed_list, dtype='float64')
print(train_speed_list.shape)#(43824, 1)
#df_train = pd.DataFrame(train_speed_list)

speed_line_val = speed_val.readline()
while speed_line_val:
    speed = list(map(float, speed_line_val.split()))
    val_speed_list.append(speed)
    speed_line_val = speed_val.readline()
val_speed_list = np.array(val_speed_list, dtype='float64')
print(val_speed_list.shape)#(8784, 1)
#df_test = pd.DataFrame(val_speed_list)
'''
#绘制散点图
lag_plot(df_train, lag=24)
pyplot.title('AM_train')
pyplot.savefig('AM_check_train.jpg')
pyplot.show()


lag_plot(df_test, lag=24)
pyplot.title('AM_test')
pyplot.savefig('AM_check_test.jpg')
pyplot.show()

#计算自相关系数
dataframe_train = concat([df_train.shift(48), df_train], axis=1)
dataframe_train.columns = ['t-48', 't']
result = dataframe_train.corr()  # 计算一个相关系数矩阵
print(result)
dataframe_test = concat([df_test.shift(48),df_test],axis=1)
dataframe_test.columns = ['t-48','t']
result_test = dataframe_test.corr()
print(result_test)
'''

'''output
          t-24         t
t-24  1.000000  0.688119
t     0.688119  1.000000
          t-24         t
t-24  1.000000  0.718967
t     0.718967  1.000000
'''
'''
          t-1         t
t-1  1.000000  0.990758
t    0.990758  1.000000
          t-1         t
t-1  1.000000  0.992352
t    0.992352  1.000000
'''
'''
          t-48         t
t-48  1.000000  0.399389
t     0.399389  1.000000
          t-48         t
t-48  1.000000  0.398535
t     0.398535  1.000000
'''
'''
#线状图
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
plot_acf(df_train, lags=31)
#pyplot.figure(figsize=(12, 6))
pyplot.title('acf')
pyplot.savefig('acf.jpg')
plot_pacf(df_train,lags=31)
#pyplot.figure(figsize=(12, 6))
pyplot.title('pacf')
pyplot.savefig('pacf.jpg')
pyplot.show()
'''

#series = Series.from_csv('daily-minimum-temperatures.csv', header=0)

# split dataset
#X = series.values
#train, test = X[1:len(X)-7], X[len(X)-7:]
X_train_speed = []
y_train_speed = []
X_test_speed = []
y_test_speed = []

# split into train and test sets
print("lenlenlenlen:",len(train_speed_list),len(val_speed_list))#lenlenlenlen: 43824 8784
for index in range(len(train_speed_list) - 25 + 1):  # 不包含最后一个数字
    X_train_speed.append(train_speed_list[index])  # 0:24,1:25...不包含最后一个数字，即24,25...
    y_train_speed.append(train_speed_list[index + 25 - 1])  # 0+24+96=120(实际下标为119)

for index in range(len(val_speed_list) - 25 + 1):  # 不包含最后一个数字
    X_test_speed.append(val_speed_list[index])  # 0:24,1:25...不包含最后一个数字，即24,25...
    y_test_speed.append(val_speed_list[index + 25 - 1])  # 0+24+96=120(实际下标为119)
# train autoregression
model = AR(train_speed_list)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
#AR model选择了先前55h的数据作为输入。输出可以看到这55h的值分别对应的系数，通过这些系数我们可以清楚的了解到每一小时对模型预测的贡献。
def computecc(targets, outputs):
    """Computes and stores the average and current value"""
    targets = np.array(targets)
    outputs = np.array(outputs)
    print("***************",targets.shape, outputs.shape)
    xBar = targets.mean()
    yBar = outputs.mean()
    print(xBar,yBar)
    SSR = 0
    varX = 0  # 公式中分子部分
    varY = 0  # 公式中分母部分
    for i in range(0, targets.shape[0]):
        diffXXBar = targets[i] - xBar
        diffYYBar = outputs[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2
    #print(type(varX),type(varY))
    #x = torch.tensor(varX)
    #y = torch.tensor(varY)
    #print(type(x),type(y))
    SST = np.sqrt(varX * varY)
    #SSR = torch.tensor(SSR)
    #print(SSR,SST,type(SSR),type(SST))
    xxx = SSR / SST
    #result = xxx.detach().numpy()
    return xxx

# make predictions
# 注意这里一次预测了整个test
predictions = model_fit.predict(start=len(train_speed_list), end=len(train_speed_list)+len(val_speed_list)-1, dynamic=False)
print("@@@@@@@@@@",len(predictions),len(val_speed_list))
#for i in range(len(predictions)):
    #print('predicted=%f, expected=%f' % (predictions[i], val_speed_list[i]))
error = mean_squared_error(val_speed_list, predictions)
print('Test MSE: %.3f' % error)
n = len(val_speed_list)
#mse = sum(np.square(y_test_speed - predictions)) / n
mae = mean_absolute_error(val_speed_list, predictions)
#print("均方误差（MSE）：均方根误差（RMSE）mse%%%%%%%%%%%%%%%",mse)
print("平均绝对误差（MAE）mae^^^^^^^^^^^^^^^",mae)
test_cc = computecc(val_speed_list, predictions)
print("啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊val_CC:",test_cc)
np.savetxt('predict_val_predict_24.txt', predictions)

# plot results
figsize(20, 5)
pyplot.plot(val_speed_list,color='cyan', label='true', linewidth=1)
pyplot.plot(predictions, color='magenta', label='predict', linewidth=1)
pyplot.legend(loc='upper right', fontsize=10)
pyplot.title('Autoregression Model')
pyplot.savefig('Autoregression Model_24.jpg')
pyplot.show()
'''output(28)
Lag: 55
Coefficients: [     ]

Test MSE: 11423.790
平均绝对误差（MAE）mae^^^^^^^^^^^^^^^ 82.82515577233744
*************** (8784, 1) (8784,)
446.4091530054645 412.6698854116349
啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊val_CC: [0.00075527]
'''
'''(25)
Lag: 55
Coefficients: [     ]

Test MSE: 11423.790
平均绝对误差（MAE）mae^^^^^^^^^^^^^^^ 82.82515577233744
*************** (8784, 1) (8784,)
446.4091530054645 412.6698854116349
啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊val_CC: [0.00075527]
'''
#CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python -u AM.py | tee ./AM_24