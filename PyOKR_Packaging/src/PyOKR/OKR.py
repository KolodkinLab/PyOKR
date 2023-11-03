# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'PyOKR_v3_UI.ui'
#
# Created by: PyQt5 UI code generator 5.14.0
#
# WARNING! All changes made in this file will be lost!

"""

Code written by James Kiraly
PyOKR v3.3
December 29, 2022

"""

#Imports
from PyQt5 import QtCore, QtWidgets, QtGui
import os
import sys

from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon

import mouse
import keyboard

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from numpy import array, linspace
from numpy import diff

from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
import scipy.integrate as s

import warnings
from pandas.errors import SettingWithCopyWarning
from matplotlib.backend_bases import MouseButton
import math as m
from pandasgui import show

from PyQt5 import QtCore, QtGui, QtWidgets

from sympy import lambdify
from sympy import sin
from sympy.abc import x


#Warning ignore (optional)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
frame = "Total"
df = pd.DataFrame(dict(A=[1, 2, 3], B=[2, 3, 4]))
df[df['A'] > 2]['B'] = 5  # No warnings for the chained assignment!

#Averaging function (obtains average from a list)


def Averaging(values):
    sum_of_list = 0
    for i in range(len(values)):
        sum_of_list += values[i]
    average = sum_of_list/len(values)
    return average

#Basic filtering of wave (remove outliers)
def Filtering_alg(db, threshold):
    dfilter_db = db
    dx = 1
    
    xvel = list(diff(dfilter_db["epxWave"])/dx)
    xvel.append(xvel[-1])
    xvel = [0 if m.isnan(x) else x for x in xvel]
    
    yvel = list(diff(dfilter_db["epyWave"])/dx)
    yvel.append(xvel[-1])
    yvel = [0 if m.isnan(x) else x for x in yvel]
    
    dfilter_db["Raw_X_vel"] = xvel
    dfilter_db["Raw_Y_vel"] = yvel
    
    dfilter_db["Norm_X_vel"] = dfilter_db["Raw_X_vel"] - np.average(xvel)
    dfilter_db["Norm_Y_vel"] = dfilter_db["Raw_Y_vel"] - np.average(yvel)
    
    fastpointsx = abs(dfilter_db["Norm_X_vel"]) > threshold*np.std(abs(dfilter_db["Norm_X_vel"]))
    fastpointsy = abs(dfilter_db["Norm_Y_vel"]) > threshold*np.std(abs(dfilter_db["Norm_Y_vel"]))
    
    dfilter_db["Fast_X"] = fastpointsx
    dfilter_db["Fast_Y"] = fastpointsy
    
    fastx = dfilter_db[dfilter_db.Fast_X==True].index.values
    fasty = dfilter_db[dfilter_db.Fast_Y==True].index.values
    
    dfilter_db.loc[fastx,"Raw_X_vel"]=0
    dfilter_db.loc[fasty,"Raw_Y_vel"]=0
    
    int_x = s.cumtrapz(dfilter_db["Raw_X_vel"])
    int_y = s.cumtrapz(dfilter_db["Raw_Y_vel"])
    
    dfilter_db["f_epxWave"] = np.append(int_x,int_x[-1])
    dfilter_db["f_epyWave"] = np.append(int_y, int_y[-1])

    return dfilter_db


#Graphing
def Graphing(db):
    plt.plot(wave, data = db)

#Slope Calculator
def Slope(x1,y1,x2,y2):
    slope = 0
    y = (y2-y1)
    x = (x2-x1)
    if x !=0:
        slope = y/x
        return abs(slope)
    else:
        return "NA"
    
#Filter from derivative values
#Used to identify sudden changes in the wave (i.e. a saccade)
def Derivative_filter(db,threshold):
    avg_db = db
    avg_db["X"] = db.index
    lst2 = db.loc[0:len(db.index),wave]
    fl = []
    avg_slope_list = []
    
    for x in lst2:
        fl.append(x)
    for y in fl:
        try:
            x1 = fl.index(y)
            x2 = fl.index(y)+1
            y1 = fl[x1]
            y2 = fl[x2]
            s_after = Slope(x1,y1,x2,y2)
            
            x3 = fl.index(y)
            x4 = fl.index(y)-1
            y3 = fl[x3]
            y4 = fl[x4]
            s_before = Slope(x3,y3,x4,y4)
            
            avg_slope = (s_after+s_before)/2
            
            avg_slope_list.append(avg_slope)
            
        except IndexError:
            avg_slope_list.append(0)
            
    avg_db = pd.DataFrame(avg_slope_list, columns = ["Average Slope"])
    total_slope = avg_db.loc[abs(avg_db["Average Slope"]) > threshold]
    
    return total_slope

#List flattener
def Flatten(lt):
    final_lt=[]
    for x in lt:
        if type(x) is list:
            for y in x:
                final_lt.append(y)
        else:
            final_lt.append(x)
    return final_lt

#Graph of Derivative values 
#Not used  
def Derivative_grapher(db,section):
    graph_db = db
    graph_db["X"] = db.index
    graph_db.plot(kind='scatter', x="X", y="Average Slope")
    
#Scan for ETMs based on KDE maxima of slope values
#Kernel density estimation finds places of maximum slope (i.e. where ETMs are occuring)       
def Scanner(db, full_db,length):
    scan_db = db
    scan_db["X"] = db.index
    scan_ri= scan_db.reset_index()
    lst3 = scan_ri.loc[0:len(scan_ri.index),"X"]
    fl2 = []
    for x in lst3:
        fl2.append(x)

    lst4 = full_db.loc[0:len(full_db.index),"X"]
    fl3 = []
    for x in lst4:
        fl3.append(x)
    
    a = array(lst3).reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(a)
    s = linspace(min(a)-100,max(a)+100)
    e = kde.score_samples(s.reshape(-1,1))
    #plt.plot(s,e)
    
    mini, maxi = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
    #print("Minima:", s[mini])
    #print("Maxima:", s[maxi])
    #mil = s[mini].tolist()
    
    mxl = s[maxi].tolist()
    #final_mil = Flatten(mil)
    final_mxl = Flatten(mxl)
    
    ETM_list_X = []
    etm_list_final=[]
    for z in final_mxl:
        y=min(fl2, key = lambda x:abs(x-z))
        ETM_list_X.append(y)
        
    for n in ETM_list_X:
        if not etm_list_final or abs(n - etm_list_final[-1]) >= length:
            etm_list_final.append(n)
        
    return etm_list_final

#Scan backwards for bottom ETM points
#From ETM point, this scans backwards for an inflection point to label the bottom of saccade
def bot_scan(db, total_db):
    s_db = db
    t_db = total_db
    total_list = []
    max_list = []

    for x in s_db["X"]:
        start = t_db.loc[x,"X"]-40
        fast_list = []
        for y in range(start, start+50):
            try:
                if abs(t_db.loc[y,wave] - t_db.loc[y-1,wave]) < 1.5:
                    fast_list.append(y)
                else:
                    break
            except:
                fast_list.append(0)
        total_list.append(fast_list)
    tot_list = [x for x in total_list if x]
    
    
    tot_y_list = []
    for z in tot_list:
        y_list=[]
        for x in z:
            y=t_db.loc[x,wave]
            y_list.append(y)
        tot_y_list.append(y_list)
    
    max_list = []
    for x in tot_list:
        big = max(x)
        max_list.append(big)    
    return max_list

#Scan forward for top values of ETM
def top_scan(db, total_db):
    s_db = db
    t_db = total_db
    total_list = []
    max_list = []
    for x in s_db["X"]:
        start = t_db.loc[x,"X"]-30
        fast_list = []
        for y in range(start, start+50):
            try:
                if abs(t_db.loc[y,wave] - t_db.loc[y-1,wave]) > 0.3 and t_db.loc[y+1,wave] - t_db.loc[y,wave] < 0:
                    fast_list.append(y)
                    
                else:
                    break
            except:
                fast_list.append(0)
        total_list.append(fast_list)
    tot_list = [x for x in total_list if x]
    
    
    tot_y_list = []
    for z in tot_list:
        y_list=[]
        for x in z:
            y=t_db.loc[x,wave]
            y_list.append(y)
        tot_y_list.append(y_list)
    
    max_list = []
    for x in tot_list:
        big = max(x)
        #print(big)
        max_list.append(big)
    #print(max_list)
    
    return max_list

#Calculates Amplitude of ETM
#Not super necessary, may remove amp filter
def Amplitude(total_db,db1,db2):
    db1_ri = db1.reset_index()
    db2_ri = db2.reset_index()
    list1 = db2_ri.loc[0:len(db2_ri.index),"X"]
    list2 = db2_ri.loc[0:len(db2_ri.index),wave]
    
    dbx = db1_ri
    dbx["X2"] = list1
    dbx[wave2] = list2
        
    #db_xdiff = abs(dbx["X2"] - dbx["X"])
    db_ydiff = abs(dbx[wave2] - dbx[wave])
    ydiff_list = db_ydiff.tolist()
    
    
    db_amp = total_db
    
    db_amp["Amplitude (degrees)"] = ydiff_list

    
    return db_amp

#Filter based on amplitude
def Amp_filter(db, threshold):
    ampf = db[db["Amplitude (degrees)"]>threshold]
    return ampf

#List of top and bottom values in sequential order 
def Point_list():
    pointlist = []
    ranked=[]
    for x in etm_bot["X"]:
        pointlist.append(x)
    for y in etm_top["X"]:
        pointlist.append(y)
    ranked = np.sort(pointlist)   

    return ranked

#Selects regions for slowphases by creating tuples of beginning and end points
def CCW_select(lst):
    tup_lis=[]
    for x in range(len(lst)):
        try:
            tup = lst[x],lst[x+1]
            tup_lis.append(tup)
        except:
            tup = lst[x],float("NaN")
            tup_lis.append(tup)
    select = tup_lis[1::2]
    select_final = select[:-1]
    
    return select_final

#Averages columns based on header string (i.e. average all Ups together, average all Downs together, etc.)
def Column_avg(df):
    lst = []
    directions = ["Up","Down","Forward","Backward"]
    up = df.filter(regex="Upward")
    down = df.filter(regex="Downward")
    nasal = df.filter(regex="Forward")
    temporal = df.filter(regex="Backward")

    lst.append(up)
    lst.append(down)
    lst.append(nasal)
    lst.append(temporal)
    
    for x,y in zip(lst,directions):
        x[y+"_Mean"] = x.mean(axis=1)
    result = pd.concat([up,down,nasal,temporal],axis=1)
        
    return result
    
#Forms a polynomial approximation via Numpy polyfit
#Generates approximate points along said polynomial
#Calculates distance between each point and calculates vel and gain   

def Poly_fit(lst,fd,polythresh,plotthresh,framerate):
    data_list=[]
    
    stim_velocity.reindex_like(fd)
    
    xstim = stim_velocity["epxWave"].to_numpy()
    ystim = stim_velocity["epyWave"].to_numpy()
    
    fd["X_vel_stim"] = xstim
    fd["Y_vel_stim"] = ystim

    for x in lst:
        b, e = x
        setp = fd[b:e]
        
        x1 = setp["X"].to_numpy()
        y1 = setp["f_epxWave"].to_numpy()
        z1 = setp["f_epyWave"].to_numpy()
        
        poly_xy = np.polyfit(x1,y1,polythresh)
        poly_xz = np.polyfit(x1,z1,polythresh)
        
        poly_xy_vel = np.polyder(poly_xy)
        poly_xz_vel = np.polyder(poly_xz)
        
        x2=np.arange(b,e,plotthresh)
        y2=np.polyval(poly_xy,x2)
        z2=np.polyval(poly_xz,x2)
        
        dx2=np.arange(b,e,plotthresh)
        dy2=np.polyval(poly_xy_vel,dx2)
        dz2=np.polyval(poly_xz_vel,dx2)
        
        #polyvals = pd.DataFrame(x2,columns=["X"])
        setp["epxWave_poly"] = y2
        setp["epyWave_poly"] = z2
        if direction == "Forward (Horizontal)" or direction == "Backward (Horizontal)":
            plt.plot(setp["X"],setp["epxWave_poly"])
            
        if direction == "Upward (Vertical)" or direction == "Downward (Vertical)":
            plt.plot(setp["X"],setp["epyWave_poly"])

        setp["X_vel"] = dy2 * framerate
        setp["Y_vel"] = dz2 * framerate
        
        setp["X_gain"] = setp["X_vel"]/setp["X_vel_stim"]

        setp["Y_gain"] = setp["Y_vel"]/setp["Y_vel_stim"]
        
        setp.replace([np.inf, -np.inf], np.nan, inplace=True)
        #polyvals[abs(polyvals["X_gain"]) > 3] = np.nan
        #polyvals[abs(polyvals["Y_gain"]) > 3] = np.nan
        #polyvals.reset_index()
        
        vellist_for = []
        vellist_back =[]
        for x in setp["X_vel"]:
            if x > 0:
                vellist_back.append(x)
            elif x < 0:
                vellist_for.append(x)
        
        vellist_up = []
        vellist_down =[]
        for x in setp["Y_vel"]:
            if x > 0:
                vellist_up.append(x)
            elif x < 0:
                vellist_down.append(x)
        
        gainlist_for = []
        gainlist_back =[]
        for x in setp["X_gain"]:
            if x > 0:
                gainlist_back.append(x)
            elif x < 0:
                gainlist_for.append(x)
        
        gainlist_up = []
        gainlist_down =[]
        for x in setp["Y_gain"]:
            if x > 0:
                gainlist_up.append(x)
            elif x < 0:
                gainlist_down.append(x)

        abs_xvel = setp["X_vel"].abs()
        x_vel_mean = np.mean(abs_xvel)
        abs_yvel = setp["Y_vel"].abs()
        y_vel_mean = np.mean(abs_yvel)
        
        abs_xgain = setp["X_gain"].abs()
        x_gain_mean = np.median(abs_xgain)
        abs_ygain = setp["Y_gain"].abs()
        y_gain_mean = np.median(abs_ygain)
        
        y_vel_up_mean = np.mean(vellist_up)
        y_vel_down_mean = np.mean(vellist_down)
        x_vel_forward_mean = np.mean(vellist_for)
        x_vel_backward_mean = np.mean(vellist_back)
        
        y_gain_up_mean = abs(np.median(gainlist_up))
        y_gain_down_mean = abs(np.median(gainlist_down))
        x_gain_forward_mean = abs(np.median(gainlist_for))
        x_gain_backward_mean = abs(np.median(gainlist_back))
        
        data = [x_vel_mean, y_vel_mean, x_gain_mean, y_gain_mean, x_vel_forward_mean, x_vel_backward_mean, y_vel_up_mean, y_vel_down_mean, x_gain_forward_mean, x_gain_backward_mean, y_gain_up_mean, y_gain_down_mean]
        data_list.append(data)
    
    return data_list

#Average the polynomial calculations within an epoch
def Poly_average(lst):
    epochdata = pd.DataFrame(lst)
    epochdata = epochdata.transpose()
    epochdata["Mean"] = epochdata.mean(axis=1)
    
    e_list = epochdata["Mean"]
    return e_list

#Add previous averages to global dataframe containing all epochs
def Poly_add(lst):
    global f_ep
    count = str(len(f_ep.columns))
    f_ep.reset_index(drop=True,inplace=True)
    f_ep[epoch+"_"+direct+"_"+direction+"_"+count]=lst
    f_ep.index = ['Total X speed', 'Total Y speed', 'Total X gain', 'Total Y gain', 'Forward X velocity', 'Backward X velocity', 'Upward Y velocity', 'Downward Y velocity', 'Forward X gain', 'Backward X gain', 'Upward Y gain','Downward Y gain']
    return f_ep
    
    #epochdata.columns = ['Total distance','Total velocity','Total gain','X distance', 'X velocity','X gain','Y distance', 'Y velocity','Y gain']


#3D graph of slowphase and associated polynomial approximation    
def Poly_graph(fd,b,e,polythresh,plotthresh):

    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection = "3d")
    ax.grid()
    for x in ccw:
        b, e = x
        setp2 = fd[b:e]
        
        x1 = setp2["X"].to_numpy()
        y1 = setp2["f_epxWave"].to_numpy()
        z1 = setp2["f_epxWave"].to_numpy()
        poly_xy = np.polyfit(x1,y1,polythresh)
        poly_xz = np.polyfit(x1,z1,polythresh)
        
        
        x2=np.arange(b,e,plotthresh)
        y2=np.polyval(poly_xy,x2)
        z2=np.polyval(poly_xz,x2)
        
        setp = fd[b:e]
    
        x = setp["X"]
        y = setp["f_epxWave"]
        z = setp["f_epxWave"]  
        
        ax.set_xlabel('Time (cs)')
        ax.set_ylabel('X (deg)')
        ax.set_zlabel('Y (deg)')
        
        ax.scatter(x,y,z, c='r',s=10)
        plt.plot(x,y,z,'.r-')
    
        ax.scatter(x2,y2,z2, c = "b", s=10)
        plt.plot(x2,y2,z2,".b-")

#If you want to generate a movie:        
"""    
    elev_list = list(range(1,90))
    azi_list = list(range(90,135))
    count=0
    elev_list.reverse()
    for x in elev_list: 
        ax.view_init(x,90)
        filename='C:/Users/james/Documents/movie/'+str(count)+'.png'
        plt.savefig(filename, dpi=75)
        count+=1
    for y in azi_list:
        ax.view_init(1,y)
        filename='C:/Users/james/Documents/movie/'+str(count)+'.png'
        plt.savefig(filename, dpi=75)
        count+=1
    elev_list.reverse()
    for z in elev_list:
        ax.view_init(z,135)
        filename='C:/Users/james/Documents/movie/'+str(count)+'.png'
        plt.savefig(filename, dpi=75)
        count+=1
    azi_list.reverse()
    for a in azi_list:
        ax.view_init(89,a)
        filename = 'C:/Users/james/Documents/movie/'+str(count)+'.png'
        plt.savefig(filename, dpi=75)
        count+=1
"""

#Default csv and output folder messages
csv_address="Please input CSV"
wd="Please input WD"

class Ui_MainWindow(object):
    
    def open_file(self):
        global csv_address
        fileName = QFileDialog.getOpenFileName()
        csv_address = str(fileName[0])
        
        return csv_address
        
    def open_folder(self):
        global wd
        foldName = str(QFileDialog.getExistingDirectory())
        wd = foldName
        
        return wd
    
    def open_folder_sorter(self):
        global wd_f
        foldName = str(QFileDialog.getExistingDirectory())
        wd_f = foldName
        
        return wd_f
        

    #A sorter of output data. Takes all final analyses and collects them into a sorted file
    def Output_Sort(self):
        glob_df = pd.DataFrame()
        directory = wd_f
        path_list = []
        for filename in os.scandir(directory):
            if filename.is_file():
                path_list.append(str(filename.path))
        
        path_fil = [x for x in path_list if ".csv" in x]

        for path in path_fil:
            csv = pd.read_csv(path)
            numb = path.rsplit("_",2)[1]
            means = csv.filter(regex="Mean")
            means = means.add_suffix("_"+numb)
            means.index = ['Total X speed', 'Total Y speed', 'Total X gain', 'Total Y gain', 'Forward X velocity', 'Backward X velocity', 'Upward Y velocity', 'Downward Y velocity', 'Forward X gain', 'Backward X gain', 'Upward Y gain','Downward Y gain']
            tmeans = means.transpose()
            glob_df = glob_df.append(tmeans)
        
        glob_df = glob_df.sort_index()
        glob_df.to_csv(wd_f+"/Total Analysis.csv")
        
    #Refreshes Current file and Output folder labels to show current values
    def clicked_refresh(self):
        last_csv = csv_address[-40:]
        self.Currentfile.setText("Current File: ..."+ last_csv)
        last_wd = wd[-36:]
        self.OutputFolder.setText("Output Folder: ..."+last_wd)
        self.updateLabel()
    
    def update_epochcombo(self):
        self.EpochcomboBox_3.clear()
        for x in range(0,int(numepoch)):
            self.EpochcomboBox_3.addItem("Epoch "+str(x+1))
    
    #Tells user if the Animal is set
    #Animal setting initializes the mouse's dataset
    def clicked_mouse_set(self):
        mouse = "Subject set!"
        self.OutputFileLabel.setText("Output File:         "+mouse)
        self.updateLabel()
    
    #Finds Direction and Rotation of stimulus from Combo boxes        
    def direction_find(self):
        global direct
        global direction
        global wave
        global wave2
        
        direct = self.StimRotateComboBox.currentText()
        direction = self.DirectioncomboBox.currentText()
        
        if direction == "Forward (Horizontal)" or direction == "Backward (Horizontal)":
            wave = "f_epxWave"
            wave2 = "f_epxWave2"
            
        elif direction == "Upward (Vertical)" or direction == "Downward (Vertical)":
            wave = "f_epyWave"
            wave2 = "f_epyWave2"
    
    def Poly_default(self):
        if direct == "Oscillatory":
            self.PolySpinBox.setValue(15)
        if direct == "Unidirectional" or direct == "Oblique":
            self.PolySpinBox.setValue(1)
    
    def stim_param_find(self):
        global head
        global tail
        global lenepoch
        global lenpoststimedit
        global numepoch
        global framerate
        global freq
        global amp
        global horspeed
        global verspeed
        
        head = float(self.HeadSpinBox.value())
        tail = float(self.TailSpinBox.value())
        lenepoch = float(self.LenEpochEdit.text())
        lenpoststimedit = float(self.LenPostStimEdit.text())
        numepoch = float(self.NumEpochSpinBox.value())
        horspeed = float(self.HorSpeedEdit.text())
        verspeed = float(self.VerSpeedEdit.text())
        framerate = float(self.FramerateEdit.text())
        freq = float(self.FreqEdit.text())
        amp = float(self.AmpEdit.text())
    
    def stim_generator(self):
        global stim_velocity

        if direct == "Unidirectional" or direct == "Oblique":
            length = numepoch * (lenepoch+lenpoststimedit)

            frame_len = lenepoch * framerate
            frame_length = int(frame_len) + 1
            
            linstim_x = [horspeed] * (frame_length)
            linstim_y = [verspeed] * (frame_length)

            linstim_vel = pd.DataFrame()
            linstim_vel["epxWave"] = linstim_x
            linstim_vel["epyWave"] = linstim_y
            
            stim_velocity = linstim_vel

        if direct == "Oscillatory":
            
            if direction == "Forward (Horizontal)" or direction == "Downward (Vertical)":
                shift = np.pi
            if direction == "Backward (Horizontal)" or direction == "Upward (Vertical)":
                shift = 0
            
            amp_rad = amp * (np.pi/180)
            
            length = numepoch*(lenepoch + lenpoststimedit)
            
            frame_length = lenepoch * framerate
            
            freq_frame = freq/framerate
            
            frame_time = np.arange(0,frame_length+1,1)

            y=(amp_rad*sin((2*np.pi*freq_frame*x))+shift).diff(x)
            s = amp_rad*sin((2*np.pi*freq_frame*x)+shift)
            
            z = lambdify(x,y)
            s = lambdify(x,s)
            
            stim_list = []
            stim_dist_list = []
            for a in frame_time:
                stim_list.append(z(a))
                stim_dist_list.append(s(a))
            
            #stim_list.extend(tail_list)
            #head_list.extend(stim_list)
            
            stim_list = [float(framerate*x/(np.pi/180)) for x in stim_list]
            stim_dist_list = [float(x/(np.pi/180)) for x in stim_dist_list]
            
            df_sine = pd.DataFrame()
            df_sine["Frame"] = frame_time
            
            if direction == "Forward (Horizontal)" or direction == "Backward (Horizontal)":
                df_sine["epxWave"] = stim_list
                df_sine["epyWave"] = [0] * len(df_sine["epxWave"])
                
                df_sine["epxWave_dist"] = stim_dist_list
                df_sine["epyWave_dist"] = [0] * len(df_sine["epxWave_dist"])
            
            if direction == "Upward (Vertical)" or direction == "Downward (Vertical)":
                df_sine["epyWave"] = stim_list
                df_sine["epxWave"] = [0] * len(df_sine["epyWave"])

                df_sine["epyWave_dist"] = stim_dist_list
                df_sine["epxWave_dist"] = [0] * len(df_sine["epyWave_dist"])
                
            stim_velocity = df_sine

            
    def upload_stim_vector(self):
        global stim_velocity
        stim_file = QFileDialog.getOpenFileName()
        stim_file_name = str(stim_file[0])
        stim_velocity = pd.read_csv(stim_file_name)
        
    #Finds epoch from epoch combobox    
    def epoch_find(self):
        global epoch
        global beg
        global end
        
        len_between_epoch = (lenepoch+lenpoststimedit)*framerate
        lenepoch_frame = lenepoch*framerate
        head_frame = head*framerate
        tail_frame = tail*framerate
        
        epoch_db = pd.DataFrame()
        
        epochnum = []
        epochbeg = []
        epochend = []
        if direct == "Unidirectional" or direct == "Oblique":
            initial = 0 + head_frame
        if direct == "Oscillatory":
            initial = 0 + head_frame
        
        epoch = self.EpochcomboBox_3.currentText()
        
        for x in range(1,int(numepoch)+1):
            epochnum.append("Epoch "+str(x))
            beginning = initial
            ending = beginning + lenepoch_frame
            initial += len_between_epoch
            epochbeg.append(beginning)
            epochend.append(ending)
            
        epoch_db["Epoch"] = epochnum
        epoch_db["Beg"] = epochbeg
        epoch_db["End"] = epochend

        beg = int(epoch_db.loc[epoch_db["Epoch"] == epoch,"Beg"])
        end = int(epoch_db.loc[epoch_db["Epoch"] == epoch, "End"])

    
    #Finds output file from user input
    def OutputFile(self):
        global out
        out = self.OutputFileName.text()
        
    #Label updater resize
    def updateLabel(self):
        self.Currentfile.adjustSize()
        self.OutputFolder.adjustSize()
        self.OutputFileLabel.adjustSize()        
    
    def filtered(self):
        global filt
        filt = self.FilterSpinBox.value()
        
    def unfiltered(self):
        global filt
        filt = 100
    
    #Initial ETM supervision
    def ETM_Super(self):
        global db
        global fd
        global etm_df_bottom
        global scanned
        global scanned2
        global etm_bot_df
        global etm_top_df
        
        db_raw=pd.read_csv(csv_address)
        db_1 = db_raw.loc[beg:end]
        db = db_1.reset_index()
        db['epxWave'] = db['epxWave'].fillna(0)
        db['epyWave'] = db['epyWave'].fillna(0)

        fd = Filtering_alg(db,filt) 
        
        dv_data = Derivative_filter(fd,2)
        
        etm_x_bottom = Scanner(dv_data,fd,100)
        
        etm_x_bottom_final = [x for x in etm_x_bottom]
        
        etm_df_bottom = fd[fd["X"].isin(etm_x_bottom_final)]
        scanned = bot_scan(etm_df_bottom, fd)
        #scanned3 = top_scan(etm_df_bottom,fd)
        scanned2 = [x+5 for x in scanned]
        etm_bot_df = fd[fd["X"].isin(scanned)]
        etm_top_df = fd[fd["X"].isin(scanned2)]
        
        #global xdata_b
        #xdata_b=list(etm_df_bottom["X"])
        
        fig, ax = plt.subplots()
        a = plt.plot(fd["X"],fd[wave],color='red',picker = 10,zorder=1)[0]
        b = plt.scatter(etm_df_bottom["X"],etm_df_bottom[wave],color='b',s=300,picker=100,zorder=2,alpha=0.8)
        def add_or_remove_point(event):
            xydata_a = np.stack(a.get_data(),axis=1)
            xdata_a = a.get_xdata()
            ydata_a = a.get_ydata()
            xydata_b = b.get_offsets()
            
            xdata_b = b.get_offsets()[:,0]
            ydata_b = b.get_offsets()[:,1]    
            global xdata_click
            global xdata_nearest_index_a
            global xdata_nearest_index_b
            global delete_xdata_b
            global new_xdata_point_b
            global new_xydata_point_b
            
            #click x-value
            xdata_click = event.xdata
            #index of nearest x-value in a
            xdata_nearest_index_a = (np.abs(xdata_a-xdata_click)).argmin()
            xdata_nearest_index_b = (np.abs(xdata_b-xdata_click)).argmin()
            delete_xdata_b = xdata_b[xdata_nearest_index_b]
            #new scatter point x-value
            new_xdata_point_b = xdata_a[xdata_nearest_index_a]
            #new scatter point [x-value, y-value]
            new_xydata_point_b = xydata_a[new_xdata_point_b,:]
            
            if event.button is MouseButton.RIGHT:
                if new_xdata_point_b not in xdata_b:
                    
                    #insert new scatter point into b
                    new_xydata_b = np.insert(xydata_b,0,new_xydata_point_b,axis=0)
                    #sort b based on x-axis values
                    new_xydata_b = new_xydata_b[np.argsort(new_xydata_b[:,0])]
                    #update b
                    b.set_offsets(new_xydata_b)
                    
                    plt.draw()
                    
                    
            elif event.button is MouseButton.LEFT:
                #remove xdata point b EDIT for loop in each direction
                new_xydata_b =np.delete(xydata_b,np.where(xdata_b==delete_xdata_b),axis=0)
                #update b
                b.set_offsets(new_xydata_b)
                plt.draw()
                
            if event.button is MouseButton.MIDDLE:
                plt.disconnect(fig)
                global xdb
                xdb = xdata_b
                print("disconnecting")
                
        fig.canvas.mpl_connect('button_press_event',add_or_remove_point)
    
    #Top and Bot calculator with user supervision refresh    
    def Top_Bot(self):
        etm_middle = xdb
        
        etm_df_real = fd[fd["X"].isin(etm_middle)]
        real = bot_scan(etm_df_real, fd)
        real2 = [x+5 for x in real]
        real3 = [x+3 for x in real]
        etm_bot_df_real = fd[fd["X"].isin(real)]
        etm_top_df_real = fd[fd["X"].isin(real2)]
        etm_df_r = fd[fd["X"].isin(real3)]
        
        amp = Amplitude(etm_df_r,etm_top_df_real,etm_bot_df_real)
        
        #Change number to threshold amplitude of ETM
        #f_amp = Amp_filter(amp,2)
        f_amp = amp
        real3 = bot_scan(f_amp, fd)
        real4 = [x+10 for x in real3]
        
        global etm_bot_final
        global etm_top_final
        global xdata_b_top
        global xdata_bot
        global xdata_t_bot
        
        if direct == "Unidirectional" or direct == "Oblique":
            if direction == "Backward (Horizontal)" or direction == "Upward (Vertical)":
                etm_bot_final = fd[fd["X"].isin(real3)]
                etm_top_final = fd[fd["X"].isin(real4)]
            
            if direction == "Forward (Horizontal)" or direction == "Downward (Vertical)":
                etm_top_final = fd[fd["X"].isin(real3)]
                etm_bot_final = fd[fd["X"].isin(real4)]
        
        if direct == "Oscillatory":
                etm_top_final = fd[fd["X"].isin(real3)]
                etm_bot_final = fd[fd["X"].isin(real4)]
        
        #plt.scatter("X",wave, data = etm_bot_final,c="g",s=300)
        #plt.scatter("X",wave, data = etm_top_final,c="r",s=300)
        #plt.scatter("X",wave, data = f_amp, c="b",s=300)
        # etm_top_final = fd[fd["X"].isin(real3)]
        # etm_bot_final = fd[fd["X"].isin(real4)]
    
    #Bot point supervision
    def Point_Super(self):
        fig, ax = plt.subplots()
        a = plt.plot(fd["X"],fd[wave],color='blue',picker = 10,zorder=1)[0]
        b = plt.scatter(etm_top_final["X"],etm_top_final[wave],color='r',s=300,picker=100,zorder=2,alpha=0.8)
        
        def add_or_remove_point_b(event):
            
            xydata_a = np.stack(a.get_data(),axis=1)
            xdata_a = a.get_xdata()
            ydata_a = a.get_ydata()
            
            xydata_b = b.get_offsets()
            #global xdata_b_top
            xdata_b_top = b.get_offsets()[:,0]
            ydata_b = b.get_offsets()[:,1]    
            
            #click x-value
            xdata_click = event.xdata
            #index of nearest x-value in a
            xdata_nearest_index_a = (np.abs(xdata_a-xdata_click)).argmin()
            xdata_nearest_index_b = (np.abs(xdata_b_top-xdata_click)).argmin()
            delete_xdata_b = xdata_b_top[xdata_nearest_index_b]
            #new scatter point x-value
            new_xdata_point_b = xdata_a[xdata_nearest_index_a]
            #new scatter point [x-value, y-value]
            new_xydata_point_b = xydata_a[new_xdata_point_b,:]
        
            if event.button is MouseButton.RIGHT and keyboard.is_pressed('shift'):
                if new_xdata_point_b not in xdata_b_top:
                    
                    #insert new scatter point into b
                    new_xydata_b = np.insert(xydata_b,0,new_xydata_point_b,axis=0)
                    #sort b based on x-axis values
                    new_xydata_b = new_xydata_b[np.argsort(new_xydata_b[:,0])]
                    #update b
                    b.set_offsets(new_xydata_b)
                    
                    plt.draw()
                    
                    
            elif event.button is MouseButton.LEFT  and keyboard.is_pressed('shift'):
                #remove xdata point b EDIT for loop in each direction
                new_xydata_b =np.delete(xydata_b,np.where(xdata_b_top==delete_xdata_b),axis=0)
                #update b
                b.set_offsets(new_xydata_b)
                plt.draw()
                
            if event.button is MouseButton.MIDDLE:
                global botfinal
                botfinal = xdata_b_top
                print("disconnecting")
                plt.disconnect(fig)
                
        t = plt.scatter(etm_bot_final["X"],etm_bot_final[wave],color='g',s=300,picker=100,zorder=2,alpha=0.8)
        
        def add_or_remove_point_t(event):
            
            xydata_a = np.stack(a.get_data(),axis=1)
            xdata_a = a.get_xdata()
            ydata_a = a.get_ydata()
            
            xydata_t = t.get_offsets()
            #global xdata_b_bot
            xdata_t_bot = t.get_offsets()[:,0]
            ydata_t = t.get_offsets()[:,1]    
            
            #click x-value
            xdata_click = event.xdata
            #index of nearest x-value in a
            xdata_nearest_index_a = (np.abs(xdata_a-xdata_click)).argmin()
            xdata_nearest_index_t = (np.abs(xdata_t_bot-xdata_click)).argmin()
            delete_xdata_t = xdata_t_bot[xdata_nearest_index_t]
            #new scatter point x-value
            new_xdata_point_t = xdata_a[xdata_nearest_index_a]
            #new scatter point [x-value, y-value]
            new_xydata_point_t = xydata_a[new_xdata_point_t,:]
        
            if event.button is MouseButton.RIGHT and keyboard.is_pressed('shift') == False:
                if new_xdata_point_t not in xdata_t_bot:
                    
                    #insert new scatter point into b
                    new_xydata_t = np.insert(xydata_t,0,new_xydata_point_t,axis=0)
                    #sort b based on x-axis values
                    new_xydata_t = new_xydata_t[np.argsort(new_xydata_t[:,0])]
                    #update t
                    t.set_offsets(new_xydata_t)
                    
                    plt.draw()
                    
                    
            elif event.button is MouseButton.LEFT and keyboard.is_pressed('shift') == False:
                #remove xdata point b EDIT for loop in each direction
                new_xydata_t =np.delete(xydata_t,np.where(xdata_t_bot==delete_xdata_t),axis=0)
                #update t
                t.set_offsets(new_xydata_t)
                plt.draw()
                
            if event.button is MouseButton.MIDDLE:
                global topfinal
                topfinal = xdata_t_bot
                print("disconnecting")
                plt.disconnect(fig)
                
        
        fig.canvas.mpl_connect('button_press_event',add_or_remove_point_b)
        fig.canvas.mpl_connect('button_press_event',add_or_remove_point_t)
        
    #Find polynomial order from spin box
    def PolySet(self):
        global poly

        poly = self.PolySpinBox.value()
    
    #Initialize a new animal
    def MouseSet(self):
        global f_ep
        f_ep = pd.DataFrame()
    
    #Find distance between polynomial approximation points from spin box
    def DistSet(self):
        global dista
        dista = 1
    
    #Plot 2D graph of data based on horizontal or vertical set earlier
    def TwoDGraph(self):
        global etm_top
        global etm_bot
        
        etm_top = fd[fd["X"].isin(topfinal)]
        etm_bot = fd[fd["X"].isin(botfinal)]
        bot_lst_f = etm_bot["X"]
        
        real5 = [x+3 for x in bot_lst_f]
        etm_final_r = fd[fd["X"].isin(real5)]
        
        fb=Graphing(fd)
        
        plt.scatter("X",wave, data = etm_bot,c="r",s=300, zorder=3,alpha=0.8)
        plt.scatter("X",wave, data = etm_top,c="g",s=300, zorder=3,alpha=0.8)
        plt.scatter("X",wave, data = etm_final_r, c="b",s=300, zorder=1)
        
        if direct == "Oscillatory":
            if direction == "Forward (Horizontal)" or direction == "Backward (Horizontal)":
                plt.plot(stim_velocity["Frame"],stim_velocity["epxWave_dist"])
            if direction == "Upward (Vertical)" or direction == "Downward (Vertical)":
                plt.plot(stim_velocity["Frame"],stim_velocity["epyWave_dist"])            

        #etms = len(final_amp["X"])
        #print("# of ETMs: " + str(len(final_amp["X"])))
        #avg_amp = np.mean(final_amp["Amplitude (degrees)"])            
    
    #Highlights on 2D graph what slowphase will be measured
    def Selector(self):
        global select_tup
        
        direc = direction
        rank = Point_list()
        select_tup = CCW_select(rank)
        if direc == "Backward (Horizontal)" or direc == "Forward (Horizontal)":
            for x in select_tup:
                select_fd = fd.loc[x[0]:x[1]]
                plt.plot("X","f_epxWave",data=select_fd,color="orange",zorder=2)
        if direc == "Upward (Vertical)" or direc == "Downward (Vertical)":
            for x in select_tup:
                select_fd = fd.loc[x[0]:x[1]]
                plt.plot("X","f_epyWave",data=select_fd,color="orange",zorder=2)            
        
    #Final analysis to calculate polynomial approximations
    def Final_Analysis(self):
        global ccw
        global ave
        global fit
        rank = Point_list()
        ccw = CCW_select(rank)
        fit = Poly_fit(ccw,fd,poly,dista,framerate)
        ave = Poly_average(fit)
        
    #Add analysis to global set
    def Add(self):
        
        final = Poly_add(ave)
        global final_w_averages
        final_w_averages = Column_avg(final)
        
    #Read the table using PysimpleGUI
    def TableRead(self):        
        show(final_w_averages)
    
    #3D plotting graph
    def ThreeD_Graph(self):
        graph = Poly_graph(fd,beg,end,poly,dista)
        
    #Export data to CSV to set output folder
    def Export(self):
        pathname = wd + "/" + out + ".csv"
        #df.to_csv(pathname)
        
        final_w_averages.to_csv(pathname)
    
    def updater(self):
        if direct == "Oscillatory":
            self.LenEpochEdit.setText("100")
            self.LenPostStimEdit.setText("0")
            self.NumEpochSpinBox.setValue(1)
            self.HorSpeedEdit.setText("0")
            self.VerSpeedEdit.setText("0")
            
        if direct == "Unidirectional":
            if direction == "Forward (Horizontal)" or direction == "Backward (Horizontal)":
                self.LenEpochEdit.setText("30")
                self.LenPostStimEdit.setText("30")
                self.NumEpochSpinBox.setValue(5)
                self.HorSpeedEdit.setText("5")
                self.VerSpeedEdit.setText("0")
            
            if direction == "Upward (Vertical)" or direction == "Downward (Vertical)":
                self.LenEpochEdit.setText("30")
                self.LenPostStimEdit.setText("30")
                self.NumEpochSpinBox.setValue(5)
                self.HorSpeedEdit.setText("0")
                self.VerSpeedEdit.setText("5")
        
        if direct == "Oblique":
            self.LenEpochEdit.setText("30")
            self.LenPostStimEdit.setText("30")
            self.NumEpochSpinBox.setValue(5)
            self.HorSpeedEdit.setText("5")
            self.VerSpeedEdit.setText("5")
        
    def setupUi(self, MainWindow):
        
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 574)
        
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        
        #self.update_timer = QtCore.QTimer()
        #self.update_timer.start(100)
        #self.update_timer.timeout.connect(self.direction_find)
        #self.update_timer.timeout.connect(self.updater)
        
        self.SortOutputLayout = QtWidgets.QVBoxLayout()
        self.SortOutputLayout.setObjectName("SortOutputLayout")
        self.SortOutputLabel = QtWidgets.QLabel(self.centralwidget)
        self.SortOutputLabel.setObjectName("SortOutputLabel")
        self.SortOutputLayout.addWidget(self.SortOutputLabel)
        self.SortOutputButton = QtWidgets.QPushButton(self.centralwidget)
        self.SortOutputButton.setObjectName("SortOutputButton")
        self.SortOutputLayout.addWidget(self.SortOutputButton)
        self.SortOutputButton.clicked.connect(self.open_folder_sorter)
        self.SortOutputButton.clicked.connect(self.Output_Sort)
        
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.SortOutputLayout.addItem(spacerItem)
        self.gridLayout_2.addLayout(self.SortOutputLayout, 12, 5, 1, 1)
        
        self.FinalAnalysisLayout = QtWidgets.QVBoxLayout()
        self.FinalAnalysisLayout.setObjectName("FinalAnalysisLayout")
        self.FinalAnalysisLabel = QtWidgets.QLabel(self.centralwidget)
        self.FinalAnalysisLabel.setObjectName("FinalAnalysisLabel")
        self.FinalAnalysisLayout.addWidget(self.FinalAnalysisLabel)
        self.FinalAnalysisButton = QtWidgets.QPushButton(self.centralwidget)
        self.FinalAnalysisButton.setObjectName("FinalAnalysisButton")
        self.FinalAnalysisLayout.addWidget(self.FinalAnalysisButton)
        self.FinalAnalysisButton.clicked.connect(self.PolySet)
        self.FinalAnalysisButton.clicked.connect(self.DistSet)
        self.FinalAnalysisButton.clicked.connect(self.direction_find)
        self.FinalAnalysisButton.clicked.connect(self.Top_Bot)
        self.FinalAnalysisButton.clicked.connect(self.stim_param_find)
        self.FinalAnalysisButton.clicked.connect(self.TwoDGraph)
        self.FinalAnalysisButton.clicked.connect(self.Selector)
        self.FinalAnalysisButton.clicked.connect(self.Final_Analysis)
        
        self.ViewGraphLabel = QtWidgets.QLabel(self.centralwidget)
        self.ViewGraphLabel.setObjectName("ViewGraphLabel")
        
        self.FinalAnalysisLayout.addWidget(self.ViewGraphLabel)
        
        self.View2DButton = QtWidgets.QPushButton(self.centralwidget)
        self.View2DButton.setObjectName("View2DButton")
        self.View2DButton.clicked.connect(self.Top_Bot)
        self.View2DButton.clicked.connect(self.TwoDGraph)
        
        self.FinalAnalysisLayout.addWidget(self.View2DButton)
        
        self.View3DButton = QtWidgets.QPushButton(self.centralwidget)
        self.View3DButton.setObjectName("View3DButton")
        self.View3DButton.clicked.connect(self.PolySet)
        self.View3DButton.clicked.connect(self.DistSet)
        self.View3DButton.clicked.connect(self.ThreeD_Graph)
        
        self.FinalAnalysisLayout.addWidget(self.View3DButton)
        
        self.EpochAddLabel = QtWidgets.QLabel(self.centralwidget)
        self.EpochAddLabel.setObjectName("EpochAddLabel")
        
        self.FinalAnalysisLayout.addWidget(self.EpochAddLabel)
        
        self.EpochAddButton = QtWidgets.QPushButton(self.centralwidget)
        self.EpochAddButton.setObjectName("EpochAddButton")
        self.EpochAddButton.clicked.connect(self.Add)
        
        self.FinalAnalysisLayout.addWidget(self.EpochAddButton)
        
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.FinalAnalysisLayout.addItem(spacerItem1)
        
        self.gridLayout_2.addLayout(self.FinalAnalysisLayout, 8, 5, 1, 1)
        self.FinalExportLayout = QtWidgets.QVBoxLayout()
        self.FinalExportLayout.setObjectName("FinalExportLayout")
        
        self.ViewDatasetLabel = QtWidgets.QLabel(self.centralwidget)
        self.ViewDatasetLabel.setObjectName("ViewDatasetLabel")
        self.FinalExportLayout.addWidget(self.ViewDatasetLabel)
        
        self.ViewDatasetButton = QtWidgets.QPushButton(self.centralwidget)
        self.ViewDatasetButton.setObjectName("ViewDatasetButton")
        self.ViewDatasetButton.clicked.connect(self.TableRead)
        
        self.FinalExportLayout.addWidget(self.ViewDatasetButton)
        
        self.FinalExportLabel = QtWidgets.QLabel(self.centralwidget)
        self.FinalExportLabel.setObjectName("FinalExportLabel")
        
        self.FinalExportLayout.addWidget(self.FinalExportLabel)
        
        self.FinalExportButton = QtWidgets.QPushButton(self.centralwidget)
        self.FinalExportButton.setObjectName("FinalExportButton")
        self.FinalExportButton.clicked.connect(self.OutputFile)
        self.FinalExportButton.clicked.connect(self.Export)
        
        self.FinalExportLayout.addWidget(self.FinalExportButton)
        
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.FinalExportLayout.addItem(spacerItem2)
        
        self.gridLayout_2.addLayout(self.FinalExportLayout, 10, 5, 1, 1)
        
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        
        self.FramerateEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.FramerateEdit.setObjectName("FramerateEdit")
        self.gridLayout.addWidget(self.FramerateEdit, 4, 1, 1, 1)
        
        self.FreqEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.FreqEdit.setObjectName("FreqEdit")
        self.gridLayout.addWidget(self.FreqEdit, 6, 1, 1, 1)
        
        self.AmpEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.AmpEdit.setObjectName("AmpEdit")
        self.gridLayout.addWidget(self.AmpEdit, 6, 3, 1, 1)
        
        self.LenEpochLabel = QtWidgets.QLabel(self.centralwidget)
        self.LenEpochLabel.setObjectName("LenEpochLabel")
        self.gridLayout.addWidget(self.LenEpochLabel, 2, 0, 1, 1)
        
        self.Framerate = QtWidgets.QLabel(self.centralwidget)
        self.Framerate.setObjectName("Framerate")
        self.gridLayout.addWidget(self.Framerate, 4, 0, 1, 1)
        
        self.HorSpeedEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.HorSpeedEdit.setObjectName("HorSpeedEdit")
        self.gridLayout.addWidget(self.HorSpeedEdit, 3, 3, 1, 1)
        
        self.HorSpeedLabel = QtWidgets.QLabel(self.centralwidget)
        self.HorSpeedLabel.setObjectName("HorSpeedLabel")
        self.gridLayout.addWidget(self.HorSpeedLabel, 3, 2, 1, 1)
        
        self.AmpLabel = QtWidgets.QLabel(self.centralwidget)
        self.AmpLabel.setObjectName("AmpLabel")
        self.gridLayout.addWidget(self.AmpLabel, 6, 2, 1, 1)
        
        self.VerSpeedEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.VerSpeedEdit.setObjectName("VerSpeedEdit")
        self.gridLayout.addWidget(self.VerSpeedEdit, 4, 3, 1, 1)
        
        self.OscillatoryLabel = QtWidgets.QLabel(self.centralwidget)
        self.OscillatoryLabel.setObjectName("OscillatoryLabel")
        self.gridLayout.addWidget(self.OscillatoryLabel, 5, 0, 1, 1)
        
        self.VerSpeedLabel = QtWidgets.QLabel(self.centralwidget)
        self.VerSpeedLabel.setObjectName("VerSpeedLabel")
        self.gridLayout.addWidget(self.VerSpeedLabel, 4, 2, 1, 1)
        
        self.FreqLabel = QtWidgets.QLabel(self.centralwidget)
        self.FreqLabel.setObjectName("FreqLabel")
        self.gridLayout.addWidget(self.FreqLabel, 6, 0, 1, 1)
        
        self.TailSpinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.TailSpinBox.setProperty("value", 3)
        self.TailSpinBox.setObjectName("TailSpinBox")
        self.gridLayout.addWidget(self.TailSpinBox, 1, 3, 1, 1)
        
        self.HeadLabel = QtWidgets.QLabel(self.centralwidget)
        self.HeadLabel.setObjectName("HeadLabel")
        self.gridLayout.addWidget(self.HeadLabel, 1, 0, 1, 1)
        
        self.LenPostStimEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.LenPostStimEdit.setObjectName("LenPostStimEdit")
        self.gridLayout.addWidget(self.LenPostStimEdit, 2, 3, 1, 1)
        
        self.TailLabel = QtWidgets.QLabel(self.centralwidget)
        self.TailLabel.setObjectName("TailLabel")
        self.gridLayout.addWidget(self.TailLabel, 1, 2, 1, 1)
        
        self.NumEpochLabel = QtWidgets.QLabel(self.centralwidget)
        self.NumEpochLabel.setObjectName("NumEpochLabel")
        self.gridLayout.addWidget(self.NumEpochLabel, 3, 0, 1, 1)
        
        self.NumEpochSpinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.NumEpochSpinBox.setProperty("value", 5)
        self.NumEpochSpinBox.setObjectName("NumEpochSpinBox")
        self.gridLayout.addWidget(self.NumEpochSpinBox, 3, 1, 1, 1)
        
        self.LenEpochEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.LenEpochEdit.setObjectName("LenEpochEdit")
        self.gridLayout.addWidget(self.LenEpochEdit, 2, 1, 1, 1)
        
        self.LenPostStim = QtWidgets.QLabel(self.centralwidget)
        self.LenPostStim.setObjectName("LenPostStim")
        self.gridLayout.addWidget(self.LenPostStim, 2, 2, 1, 1)
        
        self.StimParamLabel = QtWidgets.QLabel(self.centralwidget)
        self.StimParamLabel.setObjectName("StimParamLabel")
        self.gridLayout.addWidget(self.StimParamLabel, 0, 0, 1, 1)
        
        self.HeadSpinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.HeadSpinBox.setProperty("value", 3)
        self.HeadSpinBox.setObjectName("HeadSpinBox")
        self.gridLayout.addWidget(self.HeadSpinBox, 1, 1, 1, 1)
        
        self.gridLayout_2.addLayout(self.gridLayout, 8, 0, 1, 1)
        
        self.line_4 = QtWidgets.QFrame(self.centralwidget)
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.gridLayout_2.addWidget(self.line_4, 9, 5, 1, 1)
        
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout_2.addWidget(self.line, 7, 0, 1, 2)
        
        self.PolyLayout = QtWidgets.QHBoxLayout()
        self.PolyLayout.setObjectName("PolyLayout")
        self.PolyLabel = QtWidgets.QLabel(self.centralwidget)
        self.PolyLabel.setObjectName("PolyLabel")
        self.PolyLayout.addWidget(self.PolyLabel)
        self.PolySpinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.PolySpinBox.setProperty("value", 1)
        self.PolySpinBox.setObjectName("PolySpinBox")
        self.PolyLayout.addWidget(self.PolySpinBox)
        
        self.gridLayout_2.addLayout(self.PolyLayout, 6, 5, 1, 1)
        
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.gridLayout_2.addWidget(self.line_2, 9, 0, 1, 1)
        
        self.Folders = QtWidgets.QVBoxLayout()
        self.Folders.setObjectName("Folders")
        
        self.Currentfile = QtWidgets.QLabel(self.centralwidget)
        self.Currentfile.setObjectName("Currentfile")
        self.Folders.addWidget(self.Currentfile)
        
        self.OutputFolder = QtWidgets.QLabel(self.centralwidget)
        self.OutputFolder.setObjectName("OutputFolder")
        self.Folders.addWidget(self.OutputFolder)
        
        self.OutputFileLabel = QtWidgets.QLabel(self.centralwidget)
        self.OutputFileLabel.setObjectName("OutputFileLabel")
        self.Folders.addWidget(self.OutputFileLabel)
        
        self.OutputFileName = QtWidgets.QLineEdit(self.centralwidget)
        self.OutputFileName.setObjectName("OutputFileName")
        self.Folders.addWidget(self.OutputFileName)
        
        self.gridLayout_2.addLayout(self.Folders, 0, 0, 1, 1)
        
        self.Direction = QtWidgets.QVBoxLayout()
        self.Direction.setObjectName("Direction")
        
        self.stimDirection = QtWidgets.QLabel(self.centralwidget)
        self.stimDirection.setObjectName("stimDirection")
        
        self.Direction.addWidget(self.stimDirection)
        self.DirectioncomboBox = QtWidgets.QComboBox(self.centralwidget)
        self.DirectioncomboBox.setObjectName("DirectioncomboBox")
        self.DirectioncomboBox.addItem("")
        self.DirectioncomboBox.addItem("")
        self.DirectioncomboBox.addItem("")
        self.DirectioncomboBox.addItem("")
        self.Direction.addWidget(self.DirectioncomboBox)
        
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        
        self.Direction.addItem(spacerItem3)
        self.gridLayout_2.addLayout(self.Direction, 5, 0, 1, 1)
        
        self.Rotation = QtWidgets.QVBoxLayout()
        self.Rotation.setObjectName("Rotation")
        self.StimRotate = QtWidgets.QLabel(self.centralwidget)
        self.StimRotate.setObjectName("StimRotate")
        self.Rotation.addWidget(self.StimRotate)
        self.StimRotateComboBox = QtWidgets.QComboBox(self.centralwidget)
        self.StimRotateComboBox.setObjectName("StimRotateComboBox")
        self.StimRotateComboBox.addItem("")
        self.StimRotateComboBox.addItem("")
        self.StimRotateComboBox.addItem("")
        self.Rotation.addWidget(self.StimRotateComboBox)
        
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.Rotation.addItem(spacerItem4)
        
        self.gridLayout_2.addLayout(self.Rotation, 6, 0, 1, 1)
        
        self.VLine1 = QtWidgets.QFrame(self.centralwidget)
        self.VLine1.setFrameShape(QtWidgets.QFrame.VLine)
        self.VLine1.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.VLine1.setObjectName("VLine1")
        self.gridLayout_2.addWidget(self.VLine1, 0, 2, 14, 2)
        
        self.EpochLayout = QtWidgets.QVBoxLayout()
        self.EpochLayout.setObjectName("EpochLayout")
        self.EpochSelect = QtWidgets.QLabel(self.centralwidget)
        self.EpochSelect.setObjectName("EpochSelect")
        self.EpochLayout.addWidget(self.EpochSelect)
        
        self.EpochcomboBox_3 = QtWidgets.QComboBox(self.centralwidget)
        self.EpochcomboBox_3.setObjectName("EpochcomboBox_3")
        self.EpochLayout.addWidget(self.EpochcomboBox_3)
        
        spacerItem5 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.EpochLayout.addItem(spacerItem5)
        
        self.gridLayout_2.addLayout(self.EpochLayout, 12, 0, 1, 1)
        
        self.StimVectorLayout = QtWidgets.QVBoxLayout()
        self.StimVectorLayout.setObjectName("StimVectorLayout")
        self.StimVectorData = QtWidgets.QLabel(self.centralwidget)
        self.StimVectorData.setObjectName("StimVectorData")
        self.StimVectorLayout.addWidget(self.StimVectorData)

        self.GenerateStimVector = QtWidgets.QPushButton(self.centralwidget)
        self.GenerateStimVector.setObjectName("GenerateStimVector")
        self.GenerateStimVector.clicked.connect(self.direction_find)
        self.GenerateStimVector.clicked.connect(self.stim_param_find)
        self.GenerateStimVector.clicked.connect(self.stim_generator)
        self.GenerateStimVector.clicked.connect(self.update_epochcombo)
        self.GenerateStimVector.clicked.connect(self.Poly_default)
        
        self.StimVectorLayout.addWidget(self.GenerateStimVector)
        
        self.ImportStimVector = QtWidgets.QPushButton(self.centralwidget)
        self.ImportStimVector.setObjectName("ImportStimVector")
        self.ImportStimVector.clicked.connect(self.stim_param_find)
        self.ImportStimVector.clicked.connect(self.update_epochcombo)
        self.ImportStimVector.clicked.connect(self.upload_stim_vector)
        self.ImportStimVector.clicked.connect(self.Poly_default)   
        
        self.StimVectorLayout.addWidget(self.ImportStimVector)
        
        spacerItem6 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.StimVectorLayout.addItem(spacerItem6)
        
        self.gridLayout_2.addLayout(self.StimVectorLayout, 10, 0, 1, 1)
        
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.gridLayout_2.addWidget(self.line_3, 7, 5, 1, 1)
        
        self.PrelimAdjustmentLayout = QtWidgets.QVBoxLayout()
        self.PrelimAdjustmentLayout.setObjectName("PrelimAdjustmentLayout")
        self.PrelimAdjustmentLabel = QtWidgets.QLabel(self.centralwidget)
        self.PrelimAdjustmentLabel.setObjectName("PrelimAdjustmentLabel")
        self.PrelimAdjustmentLayout.addWidget(self.PrelimAdjustmentLabel)
        
        self.LinearStim = QtWidgets.QPushButton(self.centralwidget)
        self.LinearStim.setObjectName("LinearStim")
        self.LinearStim.clicked.connect(self.direction_find)
        self.LinearStim.clicked.connect(self.stim_param_find)
        self.LinearStim.clicked.connect(self.epoch_find)
        self.LinearStim.clicked.connect(self.unfiltered)
        self.LinearStim.clicked.connect(self.ETM_Super)
        self.PrelimAdjustmentLayout.addWidget(self.LinearStim)
        
        self.OscillatoryStim = QtWidgets.QPushButton(self.centralwidget)
        self.OscillatoryStim.setObjectName("OscillatoryStim")
        self.OscillatoryStim.clicked.connect(self.direction_find)
        self.OscillatoryStim.clicked.connect(self.stim_param_find)
        self.OscillatoryStim.clicked.connect(self.epoch_find)
        self.OscillatoryStim.clicked.connect(self.filtered)
        self.OscillatoryStim.clicked.connect(self.ETM_Super)
        self.PrelimAdjustmentLayout.addWidget(self.OscillatoryStim)
        
        self.FilterLabel = QtWidgets.QLabel(self.centralwidget)
        self.FilterLabel.setObjectName("FilterLabel")
        self.PrelimAdjustmentLayout.addWidget(self.FilterLabel)
        
        self.FilterSpinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.FilterSpinBox.setProperty("value", 6)
        self.FilterSpinBox.setObjectName("FilterSpinBox")
        self.PrelimAdjustmentLayout.addWidget(self.FilterSpinBox)
        
        spacerItem7 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.PrelimAdjustmentLayout.addItem(spacerItem7)
        
        self.gridLayout_2.addLayout(self.PrelimAdjustmentLayout, 0, 5, 1, 1)
        
        self.TopBotAdj = QtWidgets.QVBoxLayout()
        self.TopBotAdj.setObjectName("TopBotAdj")
        self.PointLabel = QtWidgets.QLabel(self.centralwidget)
        self.PointLabel.setObjectName("PointLabel")
        self.TopBotAdj.addWidget(self.PointLabel)
        
        self.PointAdj = QtWidgets.QPushButton(self.centralwidget)
        self.PointAdj.setObjectName("PointAdj")
        self.TopBotAdj.addWidget(self.PointAdj)
        self.PointAdj.clicked.connect(self.Top_Bot)
        self.PointAdj.clicked.connect(self.Point_Super)
        
        spacerItem8 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.TopBotAdj.addItem(spacerItem8)
        
        self.gridLayout_2.addLayout(self.TopBotAdj, 5, 5, 1, 1)
        
        self.View3DGraph = QtWidgets.QLabel(self.centralwidget)
        self.View3DGraph.setGeometry(QtCore.QRect(630, 160, 158, 11))
        self.View3DGraph.setText("")
        self.View3DGraph.setObjectName("View3DGraph")
        MainWindow.setCentralWidget(self.centralwidget)
        
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 616, 20))
        
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menu_View = QtWidgets.QMenu(self.menubar) 
        self.menu_View.setObjectName("menu_View")
        MainWindow.setMenuBar(self.menubar)
        
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionSave.triggered.connect(self.MouseSet)
        self.actionSave.triggered.connect(self.clicked_mouse_set)
        
        
        self.actionExport = QtWidgets.QAction(MainWindow)
        self.actionExport.setObjectName("actionExport")
        self.actionExport.triggered.connect(self.open_folder)
        self.actionExport.triggered.connect(self.clicked_refresh)
        
        self.actionDefault = QtWidgets.QAction(MainWindow)
        self.actionDefault.setObjectName("actionDefault")
        self.actionDefault.triggered.connect(self.direction_find)
        self.actionDefault.triggered.connect(self.updater)

        self.action_View_3D_plot = QtWidgets.QAction(MainWindow)
        self.action_View_3D_plot.setObjectName("action_View_3D_plot")
        self.action_View_3D_plot.triggered.connect(self.ThreeD_Graph)
        
        self.action_View_2D_plot = QtWidgets.QAction(MainWindow)
        self.action_View_2D_plot.setObjectName("action_View_2D_plot")
        self.action_View_2D_plot.triggered.connect(self.Top_Bot)
        self.action_View_2D_plot.triggered.connect(self.TwoDGraph)
        
        self.actionCurrent_analysis_data = QtWidgets.QAction(MainWindow)
        self.actionCurrent_analysis_data.setObjectName("actionCurrent_analysis_data")
        self.actionCurrent_analysis_data.triggered.connect(self.TableRead)
        
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionOpen.triggered.connect(self.open_file)
        self.actionOpen.triggered.connect(self.clicked_refresh)
        
        self.actionSort_Data = QtWidgets.QAction(MainWindow)
        self.actionSort_Data.setObjectName("actionSort_Data")
        self.actionSort_Data.triggered.connect(self.open_folder_sorter)
        self.actionSort_Data.triggered.connect(self.Output_Sort)
        
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExport)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionDefault)
        
        self.menu_View.addAction(self.action_View_3D_plot)
        self.menu_View.addAction(self.action_View_2D_plot)
        self.menu_View.addSeparator()
        self.menu_View.addAction(self.actionCurrent_analysis_data)
        self.menu_View.addSeparator()
        self.menu_View.addAction(self.actionSort_Data)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menu_View.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.SortOutputLabel.setText(_translate("MainWindow", "Sort Output data:"))
        self.SortOutputButton.setText(_translate("MainWindow", "Select folder"))
        self.FinalAnalysisLabel.setText(_translate("MainWindow", "Final analysis:"))
        self.FinalAnalysisButton.setText(_translate("MainWindow", "Final Analysis"))
        self.ViewGraphLabel.setText(_translate("MainWindow", "View graphs:"))
        self.View2DButton.setText(_translate("MainWindow", "View 2D graph"))
        self.View3DButton.setText(_translate("MainWindow", "View 3D graph"))
        self.EpochAddLabel.setText(_translate("MainWindow", "Add epoch to output file:"))
        self.EpochAddButton.setText(_translate("MainWindow", "Add epoch"))
        self.ViewDatasetLabel.setText(_translate("MainWindow", "View current dataset:"))
        self.ViewDatasetButton.setText(_translate("MainWindow", "View dataset"))
        self.FinalExportLabel.setText(_translate("MainWindow", "Final export of mouse data:"))
        self.FinalExportButton.setText(_translate("MainWindow", "Export data"))
        self.FramerateEdit.setText(_translate("MainWindow", "100"))
        self.FreqEdit.setText(_translate("MainWindow", "0.1"))
        self.AmpEdit.setText(_translate("MainWindow", "5"))
        self.LenEpochLabel.setText(_translate("MainWindow", "Length of epoch (s):"))
        self.Framerate.setText(_translate("MainWindow", "Capture framerate:"))
        self.HorSpeedEdit.setText(_translate("MainWindow", "5"))
        self.HorSpeedLabel.setText(_translate("MainWindow", "Horizontal speed (deg/s):"))
        self.AmpLabel.setText(_translate("MainWindow", "Amplitude (deg):"))
        self.VerSpeedEdit.setText(_translate("MainWindow", "0"))
        self.OscillatoryLabel.setText(_translate("MainWindow", "For oscillatory stimuli:"))
        self.VerSpeedLabel.setText(_translate("MainWindow", "Vertical speed (deg/s):"))
        self.FreqLabel.setText(_translate("MainWindow", "Frequency (hz):"))
        self.HeadLabel.setText(_translate("MainWindow", "Head (s):"))
        self.LenPostStimEdit.setText(_translate("MainWindow", "30"))
        self.TailLabel.setText(_translate("MainWindow", "Tail (s):"))
        self.NumEpochLabel.setText(_translate("MainWindow", "Number of epochs:"))
        self.LenEpochEdit.setText(_translate("MainWindow", "30"))
        self.LenPostStim.setText(_translate("MainWindow", "Length of post-stimulus (s):"))
        self.StimParamLabel.setText(_translate("MainWindow", "Stimulus parameters:"))
        self.PolyLabel.setText(_translate("MainWindow", "Set Polynomial Order:"))
        self.Currentfile.setText(_translate("MainWindow", "Current File:"))
        self.OutputFolder.setText(_translate("MainWindow", "Output Folder:"))
        self.OutputFileLabel.setText(_translate("MainWindow", "Output File:         Subject NOT set!"))
        self.stimDirection.setText(_translate("MainWindow", "Select stimulus direction:"))
        self.DirectioncomboBox.setItemText(0, _translate("MainWindow", "Forward (Horizontal)"))
        self.DirectioncomboBox.setItemText(1, _translate("MainWindow", "Backward (Horizontal)"))
        self.DirectioncomboBox.setItemText(2, _translate("MainWindow", "Upward (Vertical)"))
        self.DirectioncomboBox.setItemText(3, _translate("MainWindow", "Downward (Vertical)"))
        self.StimRotate.setText(_translate("MainWindow", "Select stimulus type:"))
        self.StimRotateComboBox.setItemText(0, _translate("MainWindow", "Unidirectional"))
        self.StimRotateComboBox.setItemText(1, _translate("MainWindow", "Oscillatory"))
        self.StimRotateComboBox.setItemText(2, _translate("MainWindow", "Oblique"))
        self.EpochSelect.setText(_translate("MainWindow", "Select epoch:"))
        self.StimVectorData.setText(_translate("MainWindow", "Stimulus vector data:"))
        self.GenerateStimVector.setText(_translate("MainWindow", "Generate stimulus vector from parameters"))
        self.ImportStimVector.setText(_translate("MainWindow", "Import own stimulus vector data"))
        self.PrelimAdjustmentLabel.setText(_translate("MainWindow", "Preliminary adjustment:"))
        self.LinearStim.setText(_translate("MainWindow", "Unfiltered Data"))
        self.OscillatoryStim.setText(_translate("MainWindow", "Filtered Data"))
        self.FilterLabel.setText(_translate("MainWindow","Filter Z-Score Threshold"))
        self.PointLabel.setText(_translate("MainWindow", "Top/Bottom Interval Point Adjustment"))
        self.PointAdj.setText(_translate("MainWindow", "Point Adjustment"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menu_View.setTitle(_translate("MainWindow", "Analysis"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionSave.setText(_translate("MainWindow", "&Set Subject"))
        self.actionSave.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionExport.setText(_translate("MainWindow", "&Export Folder"))
        self.actionExport.setShortcut(_translate("MainWindow", "Ctrl+E"))
        self.actionDefault.setText(_translate("MainWindow","&Default Values"))
        self.actionDefault.setShortcut(_translate("MainWindow", "Ctrl+D"))
        self.action_View_3D_plot.setText(_translate("MainWindow", "&3D plot"))
        self.action_View_2D_plot.setText(_translate("MainWindow", "&2D plot"))
        self.actionCurrent_analysis_data.setText(_translate("MainWindow", "Current analysis data"))
        self.actionOpen.setText(_translate("MainWindow", "&Open"))
        self.actionOpen.setShortcut(_translate("MainWindow","Ctrl+O"))
        self.actionSort_Data.setText(_translate("MainWindow", "Sort Data"))
            
def run():
    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    
    MainWindow.show()
    sys.exit(app.exec_())
    
if __name__ == '__main__':
    r = run()

