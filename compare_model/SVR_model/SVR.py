from pandas import Series,concat
import pandas as pd
from matplotlib import pyplot
from pandas.plotting import lag_plot
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
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

# split dataset
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
print("data okkkkk")

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

linear_svr = SVR(kernel='linear')   #线性核函数初始化的SVR
linear_svr.fit(X_train_speed,y_train_speed)
linear_svr_y_predict=linear_svr.predict(X_test_speed)
np.savetxt('linear_SVR_predict_val_predict.txt', linear_svr_y_predict)
print('R-squared value of linear SVR is',linear_svr.score(X_test_speed,y_test_speed))
print('The mean squared error of linear SVR is',mean_squared_error(y_test_speed,linear_svr_y_predict))
print('The mean absolute error of linear SVR is',mean_absolute_error(y_test_speed,linear_svr_y_predict))
print('The cc of linear SVR is',computecc(y_test_speed,linear_svr_y_predict))
print(' ')
'''R-squared value of linear SVR is 0.44999159392888677
The mean squared error of linear SVR is 5666.923736116933
The mean absolute error of linear SVR is 52.984952235187215
*************** (8760, 1) (8760,)
446.4503424657534 420.22870353358473
The cc of linear SVR is [0.71896712]'''
figsize(20, 5)
pyplot.plot(y_test_speed,color='cyan', label='true', linewidth=1)
pyplot.plot(linear_svr_y_predict, color='magenta', label='predict', linewidth=1)
pyplot.legend(loc='upper right', fontsize=10)
pyplot.title('Linear_SVR')
pyplot.savefig('Linear_SVR.jpg')
pyplot.show()


poly_svr = SVR(kernel='poly')   #多项式核函数初始化的SVR
poly_svr.fit(X_train_speed,y_train_speed)
poly_svr_y_predict=poly_svr.predict(X_test_speed)
np.savetxt('Poly_SVR_predict_val_predict.txt', poly_svr_y_predict)
print('R-squared value of Poly SVR is',poly_svr.score(X_test_speed,y_test_speed))
print('The mean squared error of Poly SVR is',mean_squared_error(y_test_speed,poly_svr_y_predict))
print('The mean absolute error of Poly SVR is',mean_absolute_error(y_test_speed,poly_svr_y_predict))
print('The cc of Poly SVR is',computecc(y_test_speed,poly_svr_y_predict))
print(' ')
'''R-squared value of Poly SVR is 0.43252884144910153
The mean squared error of Poly SVR is 5846.848416236885
The mean absolute error of Poly SVR is 53.254797026634314
*************** (8760, 1) (8760,)
446.4503424657534 421.2174768754547
The cc of Poly SVR is [0.7068545]
'''
figsize(20, 5)
pyplot.plot(y_test_speed,color='cyan', label='true', linewidth=1)
pyplot.plot(poly_svr_y_predict, color='magenta', label='predict', linewidth=1)
pyplot.legend(loc='upper right', fontsize=10)
pyplot.title('Poly_SVR')
pyplot.savefig('Poly_SVR.jpg')
pyplot.show()

rbf_svr = SVR(kernel='rbf')   #径向基核函数初始化的SVR
rbf_svr.fit(X_train_speed,y_train_speed)
rbf_svr_y_predict=rbf_svr.predict(X_test_speed)
np.savetxt('RBF_SVR_predict_val_predict.txt', rbf_svr_y_predict)
print('R-squared value of RBF SVR is',rbf_svr.score(X_test_speed,y_test_speed))
print('The mean squared error of RBF SVR is',mean_squared_error(y_test_speed,rbf_svr_y_predict))
print('The mean absolute error of RBF SVR is',mean_absolute_error(y_test_speed,rbf_svr_y_predict))
print('The cc of RBF SVR is',computecc(y_test_speed,rbf_svr_y_predict))
'''R-squared value of RBF SVR is 0.4482811318495713
The mean squared error of RBF SVR is 5684.547208867542
The mean absolute error of RBF SVR is 53.09768804094388
*************** (8760, 1) (8760,)
446.4503424657534 419.7309506634322
The cc of RBF SVR is [0.71980374]
'''
figsize(20, 5)
pyplot.plot(y_test_speed,color='cyan', label='true', linewidth=1)
pyplot.plot(rbf_svr_y_predict, color='magenta', label='predict', linewidth=1)
pyplot.legend(loc='upper right', fontsize=10)
pyplot.title('RBF_SVR')
pyplot.savefig('RBF_SVR.jpg')
pyplot.show()


#CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python -u SVR.py | tee ./SVR_model