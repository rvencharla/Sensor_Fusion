import numpy as np 
import matplotlib.pyplot as plt
import os

def get_gps_data():
    abs_path = os.getcwd()
    gps = np.load(abs_path + "\path") # Add the path to data file
    return gps

def get_vio_data():
    abs_path = os.getcwd()
    vio = np.load(abs_path + "\path") # Add the path to data file
    return vio

def plot_raw_data(gps, vio):
    plt.figure(figsize=(10,8))
    plt.plot(gps[:,0], gps[:, 1], label="GPS_XY")
    plt.plot(vio[:,0], vio[:, 1], label="VIO")
    plt.grid(True)
    plt.title("Raw Data on Local Coordinate System")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(loc="lower left")
    plt.xticks(np.arange(-10,40,2))
    plt.yticks(np.arange(-30,2,2))
    plt.show()
    
def main():  
    gps = get_gps_data()
    vio = get_vio_data()
    plot_raw_data(gps, vio)

if __name__ == "__main__":
    main()