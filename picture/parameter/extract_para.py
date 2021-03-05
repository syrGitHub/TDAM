import numpy as np
import time
import csv
import keras
import keras.backend as K
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential,Model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from IPython.core.pylabtools import figsize
from keras.layers import GRU, TimeDistributed,Lambda,Multiply,GlobalAveragePooling1D,Input,Reshape,Softmax,RepeatVector
from keras.layers import Multiply,Permute
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, Callback, TensorBoard, ReduceLROnPlateau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
import os
from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import mean_absolute_error # 平方绝对误差
import random as rn
from keras.utils import plot_model
# config = tf.ConfigProto( device_count = {'GPU': 2} )
# sess = tf.Session(config=config)
# keras.backend.set_session(sess)
np.random.seed(2017)
# 以下是 Python 在一个明确的初始状态生成固定随机数字所必需的。
rn.seed(12345)
# 强制 TensorFlow 使用单线程。
# 多线程是结果不可复现的一个潜在因素。
# 更多详情，见: https://stackoverflow.com/questions/42022950/
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
# `tf.set_random_seed()` 将会以 TensorFlow 为后端，
# 在一个明确的初始状态下生成固定随机数字。
# 更多详情，见: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

def plottruepredict(para,file_dir):#time 为预测的第一个数字的下标如：24+1-1，24+96-1,前24h预测第1h后的速度
    print(para.shape)
    bins = np.arange(0,len(para[0]),2)
    #figsize(20,5)
    figsize(60,20)
    x = list(range(len(para[0])))
    total_width, n = 0.7, 7
    width = total_width / n
    speed=para[0]
    area=para[1]
    density=para[2]
    temperature=para[3]
    sigma_B=para[4]
    pressure=para[5]
    ICME=para[6]
    '''
    plt.bar(x, speed, label='speed', color='lightcoral')
    plt.bar(x, area, label='area', bottom=speed, color='darkviolet')
    #plt.bar(x, density, label='density', bottom=speed+area, color='darkorange')
    #plt.bar(x, temperature, label='temperature', bottom=speed+density, color='gold')
    '''
    plt.bar(x, speed, width=width, label='speed', fc='lightcoral')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, area, width=width, label='area', fc='darkviolet')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, density, width=width, label='density', fc='darkorange')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, temperature, width=width, label='temperature', fc='gold')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, sigma_B, width=width, label='sigma_B', fc='yellowgreen')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, pressure, width=width, label='pressure', fc='lightseagreen')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, ICME, width=width, label='ICME', fc='skyblue')

    plt.xlabel("Time [hour]", fontsize=23)
    plt.ylabel("Parameter Value", fontsize=23)
    plt.legend(loc='upper right', fontsize=18)
    plt.tick_params(axis='both', labelsize=15)
    plt.show()

    #plt.title('True_Predict')
    plt.savefig("para_400.png", bbox_inches='tight', dpi=400, pad_inches=0.0)
    plt.savefig("para_white_400.png", dpi=400, pad_inches=0.0)
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr
def run_network(model=None, data=None, batch_size = None, epoch=None,sequence_length=None,predict_length=None,
                path=None,i=None,checkpoint_dir=None, area_length=None,area_length_two=None):
    # 载入模型
    adam = optimizers.Adam(lr=0.01, clipvalue=5.0)
    lr_metric = get_lr_metric(adam)
    print("loading model")
    #model = load_model("/home/sunyanru19s/solar_wind_coding/LSTM/Ablation/bset_model/six_1GRU_96_24_50epoch_128_respectively_0.01lr_5_testbs1_zero_all_ICME_GRU/lstm.50-0.4406-0.3799.h5",
                       #custom_objects={"lr":lr_metric})
    model = load_model(
        "/home/sunyanru19s/solar_wind_coding/LSTM/Ablation/feature_dimension=7/lstm.50-0.4355-0.5462.h5",
        custom_objects={"lr": lr_metric})

    print("model okkkkk!!!!!!!!!!")

    if os.path.exists(checkpoint_dir):
        print('INFO:checkpoint exists, Load weights from %s\n' % checkpoint_dir)
        model.load_weights("/home/sunyanru19s/solar_wind_coding/LSTM/Ablation/feature_dimension=7/lstm.50-0.4355-0.5462.h5")
        weight_Dense_2,bias_Dense_2 = model.get_layer('dense_1').get_weights()
        print(weight_Dense_2)  # (32,1)
        print(bias_Dense_2)
        print(weight_Dense_2.shape)     # (32,1)
        print(bias_Dense_2.shape)   # (1, )
        print("checkpoint_loaded OK!!!!!!!!!")
    print(model.summary())
    print(model.layers)
    print(model.layers[0].output)
    print(model.layers[1].output)
    print(model.layers[1])
    print(model.weights)
    for i in range (0,5):
        print(i)
        print(model.layers[i].output)
        print(model.get_weights()[i])   # 2
    print("@@@@@@@@@@@@@@@@@@@@@")
    para = model.get_weights()[0]
    print(para)

    np.savetxt(os.path.join(path, 'para.txt'), para)
    plottruepredict(para,path)
def test(data = None,batch_size=None,model=None, sequence_length=None,predict_length=None,path = None):
    if data is None:
        print('Loading data... ')
        train = open("/home/sunyanru19s/solar_wind_coding/LSTM/data/speed_train.txt")
        line = train.readline()
        train_list = []
        while line:
            num = list(map(float, line.split()))
            train_list.append(num)
            line = train.readline()
        train_list = np.array(train_list, dtype = 'float64')
        train.close()
        val = open("/home/sunyanru19s/solar_wind_coding/LSTM/data/speed_val(2016).txt")
        line = val.readline()
        val_list = []
        while line:
            num = list(map(float, line.split()))
            val_list.append(num)
            line = val.readline()
        val_list = np.array(val_list, dtype = 'float64')
        val.close()
        # print(train_list, val_list, train_list.shape, val_list.shape)   # (43824, 1) (8784, 1)

        train_n, train_low, train_high = Normalize(train_list)
        # 更新了
        val_n = Normalize2(val_list, train_low, train_high)
        print(train_n, val_n)

        X_train, y_train = get_data(gen='train', data_list=train_n, sequence_length=sequence_length, predict_length=predict_length)  # 前24小时预测后一个小时的速度
        X_val, y_val = get_data(gen='val', data_list=val_n, sequence_length=sequence_length, predict_length=predict_length)
        print("###################################")
        print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)  #(43704, 24, 1) (43704,) (8664, 24, 1) (8664,)
        # X为list类型，y为np.array类型
        # print(X_train.type, y_train.type, X_val.type, y_val.type)
    else:
        X_train, y_train, X_val, y_val = data
    print('\nData Loaded. Compiling...\n')

    # 载入模型
    print("loading model")
    model = load_model(model)
    print("model okkkkk!!!!!!!!!!")
    print(model.metrics_names)
    train_Y = FNoramlize(y_train, train_low, train_high)
    test_Y = FNoramlize(y_val, train_low, train_high)
    print("##################")
    print(train_Y, test_Y)  # 正常的训练集和验证集的真实标签y

    train_predict = model.predict(X_train, batch_size=batch_size)  # 预测出来的训练集
    val_predict = model.predict(X_val, batch_size=batch_size)  # 预测出来的验证集

    # 反归一化
    train_predict = FNoramlize(train_predict, train_low, train_high)  # 正常的预测出来的训练集
    test_predict = FNoramlize(val_predict, train_low, train_high)  # 正常的预测出来的验证集
    print(test_predict, test_Y)  # 打印出来预测出来的验证集和验证集标签（正常）

    train_cc = computecc(train_Y, train_predict)
    print("噢噢噢噢噢噢噢噢哦哦哦哦哦哦哦哦哦哦train_CC:",train_cc)
    test_cc = computecc(test_Y, test_predict)
    print("啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊val_CC:",test_cc)

    diff = np.mean((test_predict - test_Y) ** 2, axis=0)
    print("diff:")
    print(np.sqrt(diff))  # 计算标准差

    np.savetxt(os.path.join(path, 'model_predict_train_predict.txt'), train_predict)
    np.savetxt(os.path.join(path,'model_predict_val_predict.txt'), test_predict)
    np.savetxt(os.path.join(path,'model_predict_traincc.txt'), train_cc)
    np.savetxt(os.path.join(path,'model_predict_testcc.txt'), test_cc)
    # 标量测试误差（如果模型只有一个输出且没有评估标准） 或标量列表（如果模型具有多个输出 和/或 评估指标）。
    # 属性 model.metrics_names 将提供标量输出的显示标签。
    kk = model.evaluate(X_val, y_val, batch_size=batch_size, sample_weight=None, verbose=1)
    print(model.metrics_names)
    print(kk)
    # new_file_path = os.path.join(path, 'model_evaluate_kk.txt')
    np.savetxt(os.path.join(path, 'model_evaluate_kk.txt'), kk)
    plottruepredict(y_val,val_predict,sequence_length+predict_length,file_dir=path)

if __name__ == '__main__':
    i = 96  # 48 72 96 120 144 168 136
    predict_length = 24
    run_network(batch_size=128, epoch=50, sequence_length=i, predict_length=predict_length,
                path='', i=1,
                checkpoint_dir="/home/sunyanru19s/solar_wind_coding/LSTM/Ablation/feature_dimension=7/lstm.50-0.4355-0.5462.h5",
                area_length=i + predict_length - 96, area_length_two=10)
        # checkpoint_dir = '/home/sunyanru/solarwind/1GRU/1GRU_192_96_30epoch/lstm.30-0.0184-0.0283.h5')
    # test(batch_size=1,model='1lstm/lstm_24_1/lstm.10-0.0004-0.0004.h5',sequence_length=24,predict_length=1)
    # CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python -u extract_para.py | tee ./extract__para
