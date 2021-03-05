from pandas import Series
from pandas import DataFrame
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error,mean_absolute_error
from IPython.core.pylabtools import figsize # import figsize
import numpy as np
#series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
print('Loading data... ')
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
speed_line_val = speed_val.readline()
while speed_line_val:
    speed = list(map(float, speed_line_val.split()))
    val_speed_list.append(speed)
    speed_line_val = speed_val.readline()
val_speed_list = np.array(val_speed_list, dtype='float64')

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
#print(X_train_speed)
#print(y_train_speed)
#print("########################")
print(X_test_speed)
print(y_test_speed)
print("modelmodelmodelmodelmodel")
# persistence model
def model_persistence(x):
    return x

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

# walk-forward validation
predictions = list()
for x in X_test_speed:
    yhat = model_persistence(x)
    predictions.append(yhat)
test_score = mean_squared_error(y_test_speed, predictions)
print('Test MSE: %.3f' % test_score)
n = len(y_test_speed)
#mse = sum(np.square(y_test_speed - predictions)) / n
mae = mean_absolute_error(y_test_speed, predictions)
#print("均方误差（MSE）：均方根误差（RMSE）mse%%%%%%%%%%%%%%%",mse)
print("平均绝对误差（MAE）mae^^^^^^^^^^^^^^^",mae)
test_cc = computecc(y_test_speed, predictions)
print("啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊val_CC:",test_cc)
np.savetxt('24_predict_val_predict.txt', predictions)

# plot predictions vs expected
figsize(20, 5)
pyplot.plot(y_test_speed,color='cyan', label='true', linewidth=1)
pyplot.plot(predictions, color='magenta', label='predict', linewidth=1)
pyplot.legend(loc='upper right', fontsize=10)
pyplot.title('24_day_persistence')
pyplot.savefig('24_day_persistence_true_predict.jpg')
pyplot.show()
'''output(28)
Test MSE: 3.423
'''
#CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python -u 27_day_persistence.py | tee ./27_day_persistence
'''
Test MSE: 6631.917
平均绝对误差（MAE）mae^^^^^^^^^^^^^^^ 62.51181911613566
*************** (8757, 1) (8757, 1)
446.4525522439192 446.5546420006852
啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊val_CC: [0.67778709]
'''
'''output(25)
Test MSE: 3.423
'''
#CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python -u 27_day_persistence.py | tee ./24_day_persistence
'''
Test MSE: 5783.632
平均绝对误差（MAE）mae^^^^^^^^^^^^^^^ 57.72374429223744
*************** (8760, 1) (8760, 1)
446.4503424657534 446.51518264840183
啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊val_CC: [0.71896712]
'''