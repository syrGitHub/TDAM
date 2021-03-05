#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

if __name__ == '__main__':
    print('Loading data... ')
    best = np.loadtxt("/home/sunyanru19s/solar_wind_coding/LSTM/picture/best_result/result_data/predict.txt")
    observe = np.loadtxt("/home/sunyanru19s/solar_wind_coding/LSTM/picture/best_result/result_data/true.txt")

    best = best[-8665:]
    observe = observe[-8665:]
    bins = np.arange(0,len(observe),1)

    plt.figure(figsize=(25, 10), dpi=400)
    '''
    p1=plt.subplot(2,1,1)
    p1.plot(bins, best, "tomato", label='Prediction', linewidth=0.5)
    p1.plot(bins, observe, "royalblue", label='Observation', linewidth=0.5)
    p2=plt.subplot(2,1,2)
    p2.plot(bins, best, "tomato", label='Prediction', linewidth=1)
    p2.plot(bins, observe, "royalblue", label='Observation', linewidth=1)
    plt.show()
    '''
    p1 = plt.subplot(2,1,1)
    p2 = plt.subplot(2,1,2)

    p1.plot(bins, best, "tomato", label='Prediction', linewidth=1, linestyle='--')
    p1.plot(bins, observe, "royalblue", label='Observation', linewidth=1)
    p2.plot(bins, best, "tomato", label='Prediction', linewidth=1, linestyle='--')
    p2.plot(bins, observe, "royalblue", label='Observation', linewidth=1)

    #plt.show()
    #print(len(best),len(observe))
    p1.axis([0.0, 8800, 200, 800])

    p1.set_ylabel("Bulk Velocity [km/s]", fontsize=12)
    p1.set_xlabel("Time [hour]", fontsize=14)
    #p1.set_title("A simple example",fontsize=18)
    #p1.grid(True)
    p1.legend(loc='upper right', fontsize=10)


    p2.axis([2600, 3300,300,700])
    p2.set_ylabel("Bulk Velocity [km/s]", fontsize=12)
    p2.set_xlabel("Time [hour]", fontsize=14)
    #p2.grid(True)
    p2.legend(loc='upper right', fontsize=10)

    # plot the box
    tx0 = 2600
    tx1 = 3300
    ty0 = 300
    ty1 = 700
    sx = [tx0, tx1, tx1, tx0, tx0]
    sy = [ty0, ty0, ty1, ty1, ty0]
    p1.plot(sx, sy, "purple")

    # plot patch lines
    xy = (3300, 300)
    xy2 = (3300, 700)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=p2, axesB=p1)
    p2.add_artist(con)

    xy = (2600, 300)
    xy2 = (2600, 700)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=p2, axesB=p1)
    p2.add_artist(con)
    plt.savefig("up_down_400.png", bbox_inches='tight', dpi=400, pad_inches=0.0)
    plt.savefig("up_down_white_400.png", dpi=400, pad_inches=0.0)
    plt.show()
    # CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python -u up_down.py | tee ./up_down