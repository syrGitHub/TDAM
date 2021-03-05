#-*- encoding=utf-8 -*-
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from IPython.core.pylabtools import figsize

def probability_distribution(persistence, AM,linear_SVR,Poly_SVR,RBF_SVR,GRU,LSTM,best,observe,bins_interval=1, margin=20):
    # 设置线宽
    figsize(20,30)
    plt.subplot(8,1,1)
    plt.plot(bins_interval, observe, color='royalblue', label='Observation',linewidth=1)
    plt.plot(bins_interval, persistence, color='lightcoral', label='Persistence',linewidth=1, linestyle='--')
    #plt.title("Persistence", fontsize=20)
    plt.xlabel("Time [hour]", fontsize=12)
    plt.ylabel("Bulk Velocity [km/s]", fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.subplot(8, 1, 2)
    plt.plot(bins_interval, observe, color='royalblue', label='Observation',linewidth=1)
    plt.plot(bins_interval, AM, color='lightgreen', label='AM',linewidth=1, linestyle='--')
    #plt.title("AM", fontsize=20)
    plt.xlabel("Time [hour]", fontsize=12)
    plt.ylabel("Bulk Velocity [km/s]", fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.subplot(8, 1, 3)
    plt.plot(bins_interval, observe, color='royalblue', label='Observation',linewidth=1)
    plt.plot(bins_interval,linear_SVR,color='deepskyblue',label='linear_SVR',linewidth=1, linestyle='--')
    #plt.title("SVR_Linear", fontsize=20)
    plt.xlabel("Time [hour]", fontsize=12)
    plt.ylabel("Bulk Velocity [km/s]", fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.subplot(8, 1, 4)
    plt.plot(bins_interval, observe, color='royalblue', label='Observation',linewidth=1)
    plt.plot(bins_interval,Poly_SVR,color='darkorchid',label='Poly_SVR',linewidth=1, linestyle='--')
    #plt.title("SVR_Poly", fontsize=20)
    plt.xlabel("Time [hour]", fontsize=12)
    plt.ylabel("Bulk Velocity [km/s]", fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.subplot(8, 1, 5)
    plt.plot(bins_interval, observe, color='royalblue', label='Observation',linewidth=1)
    plt.plot(bins_interval,RBF_SVR,color='magenta',label='RBF_SVR',linewidth=1, linestyle='--')
    #plt.title("SVR_RBF", fontsize=20)
    plt.xlabel("Time [hour]", fontsize=12)
    plt.ylabel("Bulk Velocity [km/s]", fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.subplot(8, 1, 6)
    plt.plot(bins_interval, observe, color='royalblue', label='Observation',linewidth=1)
    plt.plot(bins_interval, LSTM, color='darkorange', label='LSTM',linewidth=1, linestyle='--')
    #plt.title("LSTM", fontsize=20)
    plt.xlabel("Time [hour]", fontsize=12)
    plt.ylabel("Bulk Velocity [km/s]", fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.subplot(8, 1, 7)
    plt.plot(bins_interval, observe, color='royalblue', label='Observation', linewidth=1)
    plt.plot(bins_interval, GRU, color='aqua', label='GRU', linewidth=1, linestyle='--')
    #plt.title("GRU", fontsize=20)
    plt.xlabel("Time [hour]", fontsize=12)
    plt.ylabel("Bulk Velocity [km/s]", fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.subplot(8, 1, 8)
    plt.plot(bins_interval, observe, color='royalblue', label='Observation',linewidth=1)
    plt.plot(bins_interval, best, color='tomato', label='Our',linewidth=1, linestyle='--')

    # 设置图表标题，并给坐标轴添加标签
    #plt.title("Outlier_handle", fontsize=20)
    plt.xlabel("Time [hour]", fontsize=12)
    plt.ylabel("Bulk Velocity [km/s]", fontsize=12)
    plt.legend(loc='upper right', fontsize=10)

    # 设置坐标轴刻度标记的大小
    plt.tick_params(axis='both', labelsize=10)
    plt.savefig("sub_large_400.png", bbox_inches='tight', dpi=400, pad_inches=0.0)
    plt.savefig("sub_large_white_400.png", dpi=400, pad_inches=0.0)

    plt.show()

if __name__ == '__main__':
    print('Loading data... ')
    persistence = np.loadtxt('/home/sunyanru19s/solar_wind_coding/LSTM/picture/compare_picture/persistence_24_predict_val_predict.txt')
    AM = np.loadtxt("/home/sunyanru19s/solar_wind_coding/LSTM/picture/compare_picture/AM_predict_val_predict_24.txt")
    linear_SVR = np.loadtxt("/home/sunyanru19s/solar_wind_coding/LSTM/picture/compare_picture/linear_SVR_predict_val_predict.txt")
    Poly_SVR = np.loadtxt("/home/sunyanru19s/solar_wind_coding/LSTM/picture/compare_picture/Poly_SVR_predict_val_predict.txt")
    RBF_SVR = np.loadtxt("/home/sunyanru19s/solar_wind_coding/LSTM/picture/compare_picture/RBF_SVR_predict_val_predict.txt")
    GRU = np.loadtxt("/home/sunyanru19s/solar_wind_coding/LSTM/picture/compare_picture/GRU_predict_val_predict.txt")
    LSTM = np.loadtxt("/home/sunyanru19s/solar_wind_coding/LSTM/picture/compare_picture/LSTM_predict_val_predict.txt")
    best = np.loadtxt("/home/sunyanru19s/solar_wind_coding/LSTM/picture/best_result/result_data/predict.txt")
    observe = np.loadtxt("/home/sunyanru19s/solar_wind_coding/LSTM/picture/best_result/result_data/true.txt")
    '''
    persistence = persistence[-6065:-5365]
    AM = AM[-6065:-5365]
    linear_SVR = linear_SVR[-6065:-5365]
    Poly_SVR = Poly_SVR[-6065:-5365]
    RBF_SVR = RBF_SVR[-6065:-5365]
    GRU = GRU[-6065:-5365]
    LSTM = LSTM[-6065:-5365]
    best = best[-6065:-5365]
    observe = observe[-6065:-5365]
    '''
    persistence = persistence[-8665:]
    AM = AM[-8665:]
    linear_SVR = linear_SVR[-8665:]
    Poly_SVR = Poly_SVR[-8665:]
    RBF_SVR = RBF_SVR[-8665:]
    GRU = GRU[-8665:]
    LSTM = LSTM[-8665:]
    best = best[-8665:]
    observe = observe[-8665:]
    bins = np.arange(0,len(observe),1)

    probability_distribution(persistence=persistence, AM=AM, linear_SVR = linear_SVR,Poly_SVR=Poly_SVR,RBF_SVR=RBF_SVR,
                             GRU=GRU,LSTM=LSTM,best=best,observe=observe,bins_interval=bins)

#CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python -u sub_large.py | tee ./sub_large_1