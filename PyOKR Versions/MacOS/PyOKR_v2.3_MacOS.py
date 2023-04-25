# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 12:06:18 2023

@author: james
"""

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Counter.ui'
#
# Created by: PyQt5 UI code generator 5.14.0
#
# WARNING! All changes made in this file will be lost!

"""

Code written by James Kiraly
ETM Analysis Method with GUI v2.2
December 29, 2022

"""

#%%


#Imports
from PyQt5 import QtCore, QtWidgets, QtGui
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import array, linspace
#May need to replace sklearn.neighbors.kde
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
import warnings
#import PySimpleGUI as sg
from pandas.core.common import SettingWithCopyWarning
from matplotlib.backend_bases import MouseButton
import math as m
import os

from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon
from pandasgui import show

#Warning ignore
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
frame = "Total"
df = pd.DataFrame(dict(A=[1, 2, 3], B=[2, 3, 4]))
df[df['A'] > 2]['B'] = 5  # No warnings for the chained assignment!

#Averaging function
def Averaging(values):
    sum_of_list = 0
    for i in range(len(values)):
        sum_of_list += values[i]
    average = sum_of_list/len(values)
    return average

#Basic filtering of wave
def Filtering_alg(db, lead, threshold):
    diff_db = db    
    diff_list = []
    length = len(db.index)
    lst1 = db.loc[0:length, wave]  
    
    new_list = []
    for x in lst1:
        new_list.append(x)
        
    for z in new_list:
        total = new_list[new_list.index(z):new_list.index(z)+lead]
        avg = Averaging(total)
        abs_diff = abs(z-avg)
        diff_list.append(abs_diff)
    #print(len(diff_list))
    diff_db["Absolute difference"]  = diff_list
    
    filtered_db = diff_db[diff_db["Absolute difference"]<threshold]
    
    return filtered_db

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

#Calculates distance between two points from x and y values (2D)
def Distance_calculator(db):
    
    distance_list = []
    
    for x in db.index:
        try:
            distance = np.sqrt((db.loc[x+1,"epxWave"]-db.loc[x,"epxWave"])**2 + (db.loc[x+1,"epyWave"]-db.loc[x,"epyWave"])**2)
            distance_list.append(distance)
        except:
            distance_list.append(float("NaN"))
    dis_no_nan = [x for x in distance_list if not(m.isnan(x)) == True]
    
    tot_dis = sum(dis_no_nan)
    
    return(tot_dis)

#calculates velocity from distance traveled/time
def Velocity_calculator(dist,b,e):
    vel = 100*(dist/(abs(e-b)))
    
    return vel

#Divides velocity/stimulus speed
def Gain_calculator(vel,stim):
    gain = vel/stim
    return gain

#epxWave distance traveled
def X_dist(db):
    distance_list = []
    for x in db.index:
        try:
            distance = abs(db.loc[x+1,"epxWave"]-db.loc[x,"epxWave"])
            #distance = db.loc[x+1,"epxWave"]-db.loc[x,"epxWave"]
            distance_list.append(distance)
        except:
            distance_list.append(float("NaN"))
    dist_list_nan = [x for x in distance_list if not(m.isnan(x))==True]
    tot_dist = sum(dist_list_nan)
    
    return tot_dist

#epyWave distance traveled    
def Y_dist(db):
    distance_list = []
    for x in db.index:
        try:
            distance = abs(db.loc[x+1,"epyWave"]-db.loc[x,"epyWave"])
            #distance = db.loc[x+1,"epyWave"]-db.loc[x,"epyWave"]
            distance_list.append(distance)
        except:
            distance_list.append(float("NaN"))
    dist_list_nan = [x for x in distance_list if not(m.isnan(x))==True]
    tot_dist = sum(dist_list_nan)
    
    return tot_dist

#Averages columns based on header string (i.e. average all Ups together, average all Downs together, etc.)
def Column_avg(df):
    lst = []
    directions = ["Up","Down","Nasal","Temporal"]
    up = df.filter(regex="_CCW_Vertical")
    down = df.filter(regex="_CW_Vertical")
    nasal = df.filter(regex="_CW_Horizontal")
    temporal = df.filter(regex="_CCW_Horizontal")
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
def Poly_fit(lst,fd,polythresh,plotthresh,stim):
    data_list=[]
    for x in lst:
        b, e = x
        setp = fd[b:e]
        
        x1 = setp["X"].to_numpy()
        y1 = setp["epxWave"].to_numpy()
        z1 = setp["epyWave"].to_numpy()
        poly_xy = np.polyfit(x1,y1,polythresh)
        poly_xz = np.polyfit(x1,z1,polythresh)
        
        x2=np.arange(b,e,plotthresh)
        y2=np.polyval(poly_xy,x2)
        z2=np.polyval(poly_xz,x2)
        
        polyvals = pd.DataFrame(x2,columns=["X"])
        polyvals["epxWave"] = y2
        polyvals["epyWave"] = z2
        polyvals.reset_index()
        
        distance = Distance_calculator(polyvals)
        velocity = Velocity_calculator(distance,b,e)
        gain = Gain_calculator(velocity,stim)
        
        xdist = X_dist(polyvals)
        xvel = Velocity_calculator(xdist,b,e)
        xgain = Gain_calculator(xvel,stim)
        
        ydist = Y_dist(polyvals)
        yvel = Velocity_calculator(ydist,b,e)
        ygain = Gain_calculator(yvel,stim)
        
        data = [distance,velocity,gain,xdist,xvel,xgain,ydist,yvel,ygain]
        
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
    
    f_ep.index = ['Total distance','Total velocity','Total gain','X distance', 'X velocity','X gain','Y distance', 'Y velocity','Y gain']
    
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
        y1 = setp2["epxWave"].to_numpy()
        z1 = setp2["epyWave"].to_numpy()
        poly_xy = np.polyfit(x1,y1,polythresh)
        poly_xz = np.polyfit(x1,z1,polythresh)
        
        x2=np.arange(b,e,plotthresh)
        y2=np.polyval(poly_xy,x2)
        z2=np.polyval(poly_xz,x2)
        
        setp = fd[b:e]
    
        x = setp["X"]
        y = setp["epxWave"]
        z = setp["epyWave"]  
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('X (deg)')
        ax.set_zlabel('Y (deg)')
        
        ax.scatter(x,y,z, c='r',s=10)
        plt.plot(x,y,z,'.r-')
    
        ax.scatter(x2,y2,z2, c = "b", s=10)
        plt.plot(x2,y2,z2,".b-")
        
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

#Class: form UI main window PyQT5 MainWindow
#This class forms the UI
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
    
    """ 
    #PysimpleGUI CSV browser
    def CSV_Button(self,MainWindow):
           global csv_address
           
           layout = [  
                       [sg.Text("Choose a wave CSV file:")],
                       [sg.InputText(key="-FILE_PATH-"), 
                       sg.FileBrowse(file_types=[("CSV Files", "*.csv")])],
                       [sg.Button('Submit'), sg.Exit()]
                   ]
           
           window = sg.Window("Display CSV", layout)
           
           while True:
               event, values = window.read()
               if event in (sg.WIN_CLOSED, 'Exit'):
                   csv_address = "Please select CSV file"
                   break
               elif event == "Submit":
                   csv_address = values["-FILE_PATH-"]
                   break
           window.close()
           
           return csv_address 
       
    #PysimpleGUI Working directory browser
    def WD_Button(self,MainWindow):
        global wd
        
        layout = [
            [sg.Text("Choose working directory:")],
            [sg.InputText(key="-FOLDER-"),
             sg.FolderBrowse()],
            [sg.Button("Submit"),sg.Exit()]
            ]
        window = sg.Window("Display",layout,resizable = True)
        while True:
            event, values = window.read()
            if event in (sg.WIN_CLOSED, 'Exit'):
                break
            elif event == "Submit":
                wd_address = values["-FOLDER-"]
                break
            
        window.close()
        wd = wd_address    
        """
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
            means.index = ['Total distance','Total velocity','Total gain','X distance', 'X velocity','X gain','Y distance', 'Y velocity','Y gain']
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
    
    #Tells user if the Animal is set
    def clicked_mouse_set(self):
        mouse = "Animal set!"
        self.stimDirection_2.setText("Output File:         "+mouse)
        self.updateLabel()
    
    #Finds Direction and Rotation of stimulus from Combo boxes        
    def direction_find(self):
        global direct
        global direction
        global wave
        global wave2
        
        direct = self.StimRotateComboBox.currentText()
        direction = self.DirectioncomboBox.currentText()
        
        if direction == "Horizontal":
            wave = "epxWave"
            wave2 = "epxWave2"
            
        elif direction == "Vertical":
            wave = "epyWave"
            wave2 = "epyWave2"
    #Finds epoch from epoch combobox    
    def epoch_find(self):
        global epoch
        global beg
        global end
        
        epoch = self.EpochcomboBox_3.currentText()
        
        if epoch == "Epoch 1 (3-30s)":
            beg,end = 0,3300
        if epoch == "Epoch 2 (63-93s)":
            beg,end = 6300,9300
        if epoch == "Epoch 3 (123-153s)":
            beg,end = 12300, 15300
        if epoch == "Epoch 4 (183-213s)":
            beg,end = 18300, 21300
        if epoch == "Epoch 5 (243-273s)":
            beg,end = 24300,27300
    
    #Finds output file from user input
    def OutputFile(self):
        global out
        out = self.OutputFileName.text()
    
    #Finds Stimulus speed from user input
    def Stim_Speed(self):
        global speed
        speed = int(self.StimSpeedInput.text())
     
    #Label updater resize
    def updateLabel(self):
        self.Currentfile.adjustSize()
        self.OutputFolder.adjustSize()
        self.stimDirection_2.adjustSize()
    
    #First calculation of ETMs from button
    def InitialETM(self):
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

        fd = Filtering_alg(db,50,20) 
        
        frame = "Total"
        #Graphing(db)
        #fgraph = Graphing(fd)
        
        dv_data = Derivative_filter(fd,2)
        dv_data["Average Slope"] = dv_data["Average Slope"]
        
        etm_x_bottom = Scanner(dv_data,fd,100)
        
        etm_x_bottom_final = [x for x in etm_x_bottom]
        
        
        etm_df_bottom = fd[fd["X"].isin(etm_x_bottom_final)]
        scanned = bot_scan(etm_df_bottom, fd)
        #scanned3 = top_scan(etm_df_bottom,fd)
        scanned2 = [x+5 for x in scanned]
        etm_bot_df = fd[fd["X"].isin(scanned)]
        etm_top_df = fd[fd["X"].isin(scanned2)]
        
        global xdata_b
        xdata_b=list(etm_df_bottom["X"])
        
        plt.plot(fd["X"],fd[wave],color='red',picker = 10)[0]
        plt.scatter(etm_df_bottom["X"],etm_df_bottom[wave],color='b',s=300,picker=100)
    
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

        fd = Filtering_alg(db,50,20) 
        
        frame = "Total"
        #Graphing(db)
        #fgraph = Graphing(fd)
        
        dv_data = Derivative_filter(fd,2)
        dv_data["Average Slope"] = dv_data["Average Slope"]
        
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
        a = plt.plot(fd["X"],fd[wave],color='red',picker = 10)[0]
        b = plt.scatter(etm_df_bottom["X"],etm_df_bottom[wave],color='b',s=300,picker=100)
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
    
    #Initial Top and Bottom point calculation
    def Top_Bot_init(self):
        global xdb
        xdb = xdata_b
        global xdata_b_top
        xdata_b_top=list(etm_top_final["X"])
        
        global xdata_b_bot
        xdata_b_bot=list(etm_bot_final["X"])
        
        global topfinal
        global botfinal
        topfinal = xdata_b_bot
        botfinal = xdata_b_top
        
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
        
        if direct == "CW":
            etm_top_final = fd[fd["X"].isin(real3)]
            etm_bot_final = fd[fd["X"].isin(real4)]
        
        if direct == "CCW":
            etm_bot_final = fd[fd["X"].isin(real3)]
            etm_top_final = fd[fd["X"].isin(real4)]
        
        #plt.scatter("X",wave, data = etm_bot_final,c="g",s=300)
        #plt.scatter("X",wave, data = etm_top_final,c="r",s=300)
        #plt.scatter("X",wave, data = f_amp, c="b",s=300)
        
    
    #Bot point supervision
    def Bot_Super(self):
        fig, ax = plt.subplots()
        a = plt.plot(fd["X"],fd[wave],color='blue',picker = 10)[0]
        b = plt.scatter(etm_top_final["X"],etm_top_final[wave],color='r',s=300,picker=100)
        
        def add_or_remove_point(event):
            
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
        
            if event.button is MouseButton.RIGHT:
                if new_xdata_point_b not in xdata_b_top:
                    
                    #insert new scatter point into b
                    new_xydata_b = np.insert(xydata_b,0,new_xydata_point_b,axis=0)
                    #sort b based on x-axis values
                    new_xydata_b = new_xydata_b[np.argsort(new_xydata_b[:,0])]
                    #update b
                    b.set_offsets(new_xydata_b)
                    
                    plt.draw()
                    
                    
            elif event.button is MouseButton.LEFT:
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
                
        
        fig.canvas.mpl_connect('button_press_event',add_or_remove_point)
        
    #Top point supervision
    def Top_Super(self):
        fig, ax = plt.subplots()
        a = plt.plot(fd["X"],fd[wave],color='blue',picker = 10)[0]
        b = plt.scatter(etm_bot_final["X"],etm_bot_final[wave],color='g',s=200,picker=100)
        
        def add_or_remove_point(event):
            
            xydata_a = np.stack(a.get_data(),axis=1)
            xdata_a = a.get_xdata()
            ydata_a = a.get_ydata()
            
            xydata_b = b.get_offsets()
            #global xdata_b_bot
            xdata_b_bot = b.get_offsets()[:,0]
            ydata_b = b.get_offsets()[:,1]    
            
            #click x-value
            xdata_click = event.xdata
            #index of nearest x-value in a
            xdata_nearest_index_a = (np.abs(xdata_a-xdata_click)).argmin()
            xdata_nearest_index_b = (np.abs(xdata_b_bot-xdata_click)).argmin()
            delete_xdata_b = xdata_b_bot[xdata_nearest_index_b]
            #new scatter point x-value
            new_xdata_point_b = xdata_a[xdata_nearest_index_a]
            #new scatter point [x-value, y-value]
            new_xydata_point_b = xydata_a[new_xdata_point_b,:]
        
            if event.button is MouseButton.RIGHT:
                if new_xdata_point_b not in xdata_b_bot:
                    
                    #insert new scatter point into b
                    new_xydata_b = np.insert(xydata_b,0,new_xydata_point_b,axis=0)
                    #sort b based on x-axis values
                    new_xydata_b = new_xydata_b[np.argsort(new_xydata_b[:,0])]
                    #update b
                    b.set_offsets(new_xydata_b)
                    
                    plt.draw()
                    
                    
            elif event.button is MouseButton.LEFT:
                #remove xdata point b EDIT for loop in each direction
                new_xydata_b =np.delete(xydata_b,np.where(xdata_b_bot==delete_xdata_b),axis=0)
                #update b
                b.set_offsets(new_xydata_b)
                plt.draw()
                
            if event.button is MouseButton.MIDDLE:
                global topfinal
                topfinal = xdata_b_bot
                print("disconnecting")
                plt.disconnect(fig)
                
        
        fig.canvas.mpl_connect('button_press_event',add_or_remove_point)
       
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
        dista = self.DistSpinBox.value()
    
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
        
        plt.scatter("X",wave, data = etm_bot,c="r",s=300)
        plt.scatter("X",wave, data = etm_top,c="g",s=300)
        plt.scatter("X",wave, data = etm_final_r, c="b",s=300)
        #etms = len(final_amp["X"])
        #print("# of ETMs: " + str(len(final_amp["X"])))
        #avg_amp = np.mean(final_amp["Amplitude (degrees)"])
        
    #Final analysis to calculate polynomial approximations
    def Final_Analysis(self):
        global ccw
        global ave
        rank = Point_list()
        ccw = CCW_select(rank)
        fit = Poly_fit(ccw,fd,poly,dista,speed)
        ave = Poly_average(fit)
        #print(ave)
        
    #Add analysis to global set
    def Add(self):
        
        final = Poly_add(ave)
        global final_w_averages
        final_w_averages = Column_avg(final)
        
    #Read the table using PysimpleGUI
    def TableRead(self):        
        print("Not available in MacOS. Select variable from variable explorer to view")
    
    #3D plotting graph
    def ThreeD_Graph(self):
        graph = Poly_graph(fd,beg,end,poly,dista)
        
    #Export data to CSV to set output folder
    def Export(self):
        pathname = wd + "/" + out + ".csv"
        df.to_csv(pathname)
        
        final_w_averages.to_csv(pathname)
       
    #Setup the UI and the associated functions with buttons, etc.
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1500, 500)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        
        self.ETMAdjustmentLayout = QtWidgets.QVBoxLayout()
        self.ETMAdjustmentLayout.setObjectName("ETMAdjustmentLayout")
        self.ETMAdjustmentLabel = QtWidgets.QLabel(self.centralwidget)
        self.ETMAdjustmentLabel.setObjectName("ETMAdjustmentLabel")
        self.ETMAdjustmentLayout.addWidget(self.ETMAdjustmentLabel)
        self.ETMAdjustment = QtWidgets.QPushButton(self.centralwidget)
        self.ETMAdjustment.setObjectName("ETMAdjustment")
        self.ETMAdjustment.clicked.connect(self.direction_find)
        self.ETMAdjustment.clicked.connect(self.epoch_find)
        self.ETMAdjustment.clicked.connect(self.Stim_Speed)
        self.ETMAdjustment.clicked.connect(self.ETM_Super)
        
        self.ETMAdjustmentLayout.addWidget(self.ETMAdjustment)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.ETMAdjustmentLayout.addItem(spacerItem)
        
        self.gridLayout_2.addLayout(self.ETMAdjustmentLayout, 4, 4, 2, 1)
        
        self.VLine1 = QtWidgets.QFrame(self.centralwidget)
        self.VLine1.setFrameShape(QtWidgets.QFrame.VLine)
        self.VLine1.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.VLine1.setObjectName("VLine1")
        
        self.gridLayout_2.addWidget(self.VLine1, 0, 1, 13, 2)
        
        self.Folders = QtWidgets.QVBoxLayout()
        self.Folders.setObjectName("Folders")
        
        self.Currentfile = QtWidgets.QLabel(self.centralwidget)
        self.Currentfile.setObjectName("Currentfile")
        
        self.Folders.addWidget(self.Currentfile)
        self.OutputFolder = QtWidgets.QLabel(self.centralwidget)
        self.OutputFolder.setObjectName("OutputFolder")
        
        self.Folders.addWidget(self.OutputFolder)
        self.gridLayout_2.addLayout(self.Folders, 0, 0, 1, 1)
        
        self.HLine1 = QtWidgets.QFrame(self.centralwidget)
        self.HLine1.setFrameShape(QtWidgets.QFrame.HLine)
        self.HLine1.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.HLine1.setObjectName("HLine1")
        
        self.gridLayout_2.addWidget(self.HLine1, 8, 3, 1, 3)
        
        self.View3DLayout = QtWidgets.QVBoxLayout()
        self.View3DLayout.setObjectName("View3DLayout")
        self.View3DLabel = QtWidgets.QLabel(self.centralwidget)
        self.View3DLabel.setObjectName("View3DLabel")
        self.View3DLayout.addWidget(self.View3DLabel)
        self.View3DButton = QtWidgets.QPushButton(self.centralwidget)
        self.View3DButton.setObjectName("View3DButton")
        self.View3DButton.clicked.connect(self.PolySet)
        self.View3DButton.clicked.connect(self.DistSet)
        self.View3DButton.clicked.connect(self.ThreeD_Graph)
        
        self.View3DLayout.addWidget(self.View3DButton)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.View3DLayout.addItem(spacerItem1)
        
        self.gridLayout_2.addLayout(self.View3DLayout, 7, 7, 2, 2)
        
        self.EpochAddLayout = QtWidgets.QVBoxLayout()
        self.EpochAddLayout.setObjectName("EpochAddLayout")
        self.EpochAddLabel = QtWidgets.QLabel(self.centralwidget)
        self.EpochAddLabel.setObjectName("EpochAddLabel")
        self.EpochAddLayout.addWidget(self.EpochAddLabel)
        self.EpochAddButton = QtWidgets.QPushButton(self.centralwidget)
        self.EpochAddButton.setObjectName("EpochAddButton")
        self.EpochAddButton.clicked.connect(self.Add)
        
        self.EpochAddLayout.addWidget(self.EpochAddButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.TableRead)
        
        self.EpochAddLayout.addWidget(self.pushButton_2)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.EpochAddLayout.addItem(spacerItem2)
        
        self.gridLayout_2.addLayout(self.EpochAddLayout, 9, 7, 2, 2)
        self.FinalAnalysisLayout = QtWidgets.QVBoxLayout()
        self.FinalAnalysisLayout.setObjectName("FinalAnalysisLayout")
        self.FinalAnalysisLabel = QtWidgets.QLabel(self.centralwidget)
        self.FinalAnalysisLabel.setObjectName("FinalAnalysisLabel")
        self.FinalAnalysisLayout.addWidget(self.FinalAnalysisLabel)
        self.FinalAnalysisButton = QtWidgets.QPushButton(self.centralwidget)
        self.FinalAnalysisButton.setObjectName("FinalAnalysisButton")
        self.FinalAnalysisButton.clicked.connect(self.PolySet)
        self.FinalAnalysisButton.clicked.connect(self.DistSet)
        self.FinalAnalysisButton.clicked.connect(self.Top_Bot)
        self.FinalAnalysisButton.clicked.connect(self.Stim_Speed)
        self.FinalAnalysisButton.clicked.connect(self.TwoDGraph)
        self.FinalAnalysisButton.clicked.connect(self.Final_Analysis)
        
        self.FinalAnalysisLayout.addWidget(self.FinalAnalysisButton)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.FinalAnalysisLayout.addItem(spacerItem3)
        
        self.gridLayout_2.addLayout(self.FinalAnalysisLayout, 5, 7, 1, 2)
        self.FinalExportLayout = QtWidgets.QVBoxLayout()
        self.FinalExportLayout.setObjectName("FinalExportLayout")
        self.FinalExportLabel = QtWidgets.QLabel(self.centralwidget)
        self.FinalExportLabel.setObjectName("FinalExportLabel")
        self.FinalExportLayout.addWidget(self.FinalExportLabel)
        self.FinalExportButton = QtWidgets.QPushButton(self.centralwidget)
        self.FinalExportButton.setObjectName("FinalExportButton")
        self.FinalExportButton.clicked.connect(self.OutputFile)
        self.FinalExportButton.clicked.connect(self.Export)
        #self.FinalExportButton.clicked.connect(self.MouseSet)
        
        self.FinalExportLayout.addWidget(self.FinalExportButton)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.FinalExportLayout.addItem(spacerItem4)
        
        self.gridLayout_2.addLayout(self.FinalExportLayout, 11, 7, 2, 2)
        self.PolyDistLayout = QtWidgets.QVBoxLayout()
        self.PolyDistLayout.setObjectName("PolyDistLayout")
        self.PolyLabel = QtWidgets.QLabel(self.centralwidget)
        self.PolyLabel.setObjectName("PolyLabel")
        self.PolyDistLayout.addWidget(self.PolyLabel)
        self.DistLabel = QtWidgets.QLabel(self.centralwidget)
        self.DistLabel.setObjectName("DistLabel")
        self.PolyDistLayout.addWidget(self.DistLabel)
        
        self.gridLayout_2.addLayout(self.PolyDistLayout, 0, 7, 5, 1)
        self.Rotation = QtWidgets.QVBoxLayout()
        self.Rotation.setObjectName("Rotation")
        self.StimRotate = QtWidgets.QLabel(self.centralwidget)
        self.StimRotate.setObjectName("StimRotate")
        self.Rotation.addWidget(self.StimRotate)
        self.StimRotateComboBox = QtWidgets.QComboBox(self.centralwidget)
        self.StimRotateComboBox.setObjectName("StimRotateComboBox")
        self.StimRotateComboBox.addItem("")
        self.StimRotateComboBox.addItem("")
        self.Rotation.addWidget(self.StimRotateComboBox)
        spacerItem5 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.Rotation.addItem(spacerItem5)
        
        self.gridLayout_2.addLayout(self.Rotation, 6, 0, 2, 1)
        self.Direction = QtWidgets.QVBoxLayout()
        self.Direction.setObjectName("Direction")
        self.stimDirection = QtWidgets.QLabel(self.centralwidget)
        self.stimDirection.setObjectName("stimDirection")
        self.Direction.addWidget(self.stimDirection)
        self.DirectioncomboBox = QtWidgets.QComboBox(self.centralwidget)
        self.DirectioncomboBox.setObjectName("DirectioncomboBox")
        self.DirectioncomboBox.addItem("")
        self.DirectioncomboBox.addItem("")
        self.Direction.addWidget(self.DirectioncomboBox)
        spacerItem6 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.Direction.addItem(spacerItem6)
        
        self.gridLayout_2.addLayout(self.Direction, 5, 0, 1, 1)
        self.EpochLayout = QtWidgets.QVBoxLayout()
        self.EpochLayout.setObjectName("EpochLayout")
        self.EpochSelect = QtWidgets.QLabel(self.centralwidget)
        self.EpochSelect.setObjectName("EpochSelect")
        self.EpochLayout.addWidget(self.EpochSelect)
        self.EpochcomboBox_3 = QtWidgets.QComboBox(self.centralwidget)
        self.EpochcomboBox_3.setObjectName("EpochcomboBox_3")
        self.EpochcomboBox_3.addItem("")
        self.EpochcomboBox_3.addItem("")
        self.EpochcomboBox_3.addItem("")
        self.EpochcomboBox_3.addItem("")
        self.EpochcomboBox_3.addItem("")
        self.EpochLayout.addWidget(self.EpochcomboBox_3)
        spacerItem7 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.EpochLayout.addItem(spacerItem7)
        
        self.gridLayout_2.addLayout(self.EpochLayout, 8, 0, 2, 1)
        self.DistSpinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.DistSpinBox.setProperty("value", 10)
        self.DistSpinBox.setObjectName("DistSpinBox")
        self.gridLayout_2.addWidget(self.DistSpinBox, 1, 8, 4, 1)
        self.PolySpinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.PolySpinBox.setProperty("value", 1)
        self.PolySpinBox.setObjectName("PolySpinBox")
        
        self.gridLayout_2.addWidget(self.PolySpinBox, 0, 8, 1, 1)
        self.VLine2 = QtWidgets.QFrame(self.centralwidget)
        self.VLine2.setFrameShape(QtWidgets.QFrame.VLine)
        self.VLine2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.VLine2.setObjectName("VLine2")
        
        self.gridLayout_2.addWidget(self.VLine2, 0, 6, 13, 1)
        self.stimDirection_2 = QtWidgets.QLabel(self.centralwidget)
        self.stimDirection_2.setObjectName("stimDirection_2")
        
        self.gridLayout_2.addWidget(self.stimDirection_2, 1, 0, 1, 1)
        self.OutputFileName = QtWidgets.QLineEdit(self.centralwidget)
        self.OutputFileName.setObjectName("OutputFileName")
        
        self.gridLayout_2.addWidget(self.OutputFileName, 2, 0, 1, 1)
        self.InitPlotLayout = QtWidgets.QVBoxLayout()
        self.InitPlotLayout.setObjectName("InitPlotLayout")
        self.InitPlotLabel = QtWidgets.QLabel(self.centralwidget)
        self.InitPlotLabel.setObjectName("InitPlotLabel")
        self.InitPlotLayout.addWidget(self.InitPlotLabel)
        self.InitPlotButton = QtWidgets.QPushButton(self.centralwidget)
        self.InitPlotButton.setObjectName("InitPlotButton")
        self.InitPlotButton.clicked.connect(self.open_file)
        self.InitPlotButton.clicked.connect(self.clicked_refresh)
        
        self.InitPlotLayout.addWidget(self.InitPlotButton)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.open_folder)
        self.pushButton.clicked.connect(self.clicked_refresh)
        
        self.InitPlotLayout.addWidget(self.pushButton)
        spacerItem8 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.InitPlotLayout.addItem(spacerItem8)
        
        self.gridLayout_2.addLayout(self.InitPlotLayout, 12, 0, 1, 1)
        self.StimSpeed = QtWidgets.QVBoxLayout()
        self.StimSpeed.setObjectName("StimSpeed")
        self.StimSpeedLabel = QtWidgets.QLabel(self.centralwidget)
        self.StimSpeedLabel.setObjectName("StimSpeedLabel")
        self.StimSpeed.addWidget(self.StimSpeedLabel)
        self.StimSpeedInput = QtWidgets.QLineEdit(self.centralwidget)
        self.StimSpeedInput.setObjectName("StimSpeedInput")
        
        self.StimSpeed.addWidget(self.StimSpeedInput)
        spacerItem9 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.StimSpeed.addItem(spacerItem9)
        
        self.gridLayout_2.addLayout(self.StimSpeed, 10, 0, 2, 1)
        self.TopBotAdj = QtWidgets.QVBoxLayout()
        self.TopBotAdj.setObjectName("TopBotAdj")
        self.TopBotETMLabel = QtWidgets.QLabel(self.centralwidget)
        self.TopBotETMLabel.setObjectName("TopBotETMLabel")
        self.TopBotAdj.addWidget(self.TopBotETMLabel)
        self.TopPointAdj = QtWidgets.QPushButton(self.centralwidget)
        self.TopPointAdj.clicked.connect(self.Top_Bot)
        self.TopPointAdj.setObjectName("TopPointAdj")
        self.TopPointAdj.clicked.connect(self.Top_Super)
        
        self.TopBotAdj.addWidget(self.TopPointAdj)
        self.BotPointAdj = QtWidgets.QPushButton(self.centralwidget)
        self.BotPointAdj.setObjectName("BotPointAdj")
        self.BotPointAdj.clicked.connect(self.Top_Bot)
        self.BotPointAdj.clicked.connect(self.Bot_Super)
        self.TopBotAdj.addWidget(self.BotPointAdj)
        spacerItem10 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.TopBotAdj.addItem(spacerItem10)
        
        self.gridLayout_2.addLayout(self.TopBotAdj, 10, 4, 2, 1)
        self.View3DGraph = QtWidgets.QLabel(self.centralwidget)
        self.View3DGraph.setGeometry(QtCore.QRect(630, 160, 158, 11))
        self.View3DGraph.setText("")
        self.View3DGraph.setObjectName("View3DGraph")
        self.stimDirection_2.raise_()
        self.VLine1.raise_()
        self.VLine2.raise_()
        self.HLine1.raise_()
        self.PolySpinBox.raise_()
        self.DistSpinBox.raise_()
        self.OutputFileName.raise_()
        
        
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 537, 20))
        self.menubar.setObjectName("menubar")
        
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        
        self.menu_Open = QtWidgets.QMenu(self.menuFile)
        self.menu_Open.setObjectName("menu_Open")
        
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
          
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionExit.triggered.connect(self.open_folder)
        self.actionExit.triggered.connect(self.clicked_refresh)
  
        self.action_Open = QtWidgets.QAction(MainWindow)
        self.action_Open.setObjectName("action_Open")
        self.action_Open.triggered.connect(self.open_file)
        self.action_Open.triggered.connect(self.clicked_refresh)
        
        self.action_Sorter = QtWidgets.QAction(MainWindow)
        self.action_Sorter.setObjectName("action_Sorter")
        self.action_Sorter.triggered.connect(self.open_folder_sorter)
        self.action_Sorter.triggered.connect(self.Output_Sort)

        self.action_View_3D_plot = QtWidgets.QAction(MainWindow)
        self.action_View_3D_plot.setObjectName("action_View_3D_plot")
        self.action_View_3D_plot.triggered.connect(self.ThreeD_Graph)
        
        self.action_View_2D_plot = QtWidgets.QAction(MainWindow)
        self.action_View_2D_plot.setObjectName("action_View_2D_plot")
        self.action_View_2D_plot.triggered.connect(self.Top_Bot)
        self.action_View_2D_plot.triggered.connect(self.TwoDGraph)
        
        #self.actionExit_2 = QtWidgets.QAction(MainWindow)
        #self.actionExit_2.setObjectName("actionExit_2")
        
        self.actionCurrent_analysis_data = QtWidgets.QAction(MainWindow)
        self.actionCurrent_analysis_data.setObjectName("actionCurrent_analysis_data")
        self.actionCurrent_analysis_data.triggered.connect(self.TableRead)
        
        self.menuFile.addAction(self.action_Open)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        self.menuFile.addSeparator()
        #self.menuFile.addAction(self.actionExit_2)
        self.menu_View.addAction(self.action_View_3D_plot)
        self.menu_View.addAction(self.action_View_2D_plot)
        self.menu_View.addSeparator()
        self.menu_View.addAction(self.actionCurrent_analysis_data)
        self.menu_View.addSeparator()
        self.menu_View.addAction(self.action_Sorter)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menu_View.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
    #Translate the UI widgets with names and texts on the actual thing
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.ETMAdjustmentLabel.setText(_translate("MainWindow", "Preliminary ETM plot:"))
        self.ETMAdjustment.setText(_translate("MainWindow", "ETM point adjustment"))
        self.Currentfile.setText(_translate("MainWindow", "Current File:"))
        self.OutputFolder.setText(_translate("MainWindow", "Output Folder:"))
        self.View3DLabel.setText(_translate("MainWindow", "View 3D graph:"))
        self.View3DButton.setText(_translate("MainWindow", "3D graph"))
        self.EpochAddLabel.setText(_translate("MainWindow", "Add epoch to output file:"))
        self.EpochAddButton.setText(_translate("MainWindow", "Add epoch"))
        self.pushButton_2.setText(_translate("MainWindow", "N/A for Mac: View current dataset"))
        self.FinalAnalysisLabel.setText(_translate("MainWindow", "Final Analysis:"))
        self.FinalAnalysisButton.setText(_translate("MainWindow", "Analyze current ETMs"))
        self.FinalExportLabel.setText(_translate("MainWindow", "Final export of mouse data"))
        self.FinalExportButton.setText(_translate("MainWindow", "Export data"))
        self.PolyLabel.setText(_translate("MainWindow", "Set Polynomial Order:"))
        self.DistLabel.setText(_translate("MainWindow", "Set Distance between points:"))
        self.StimRotate.setText(_translate("MainWindow", "Select stimulus rotation:"))
        self.StimRotateComboBox.setItemText(0, _translate("MainWindow", "CW"))
        self.StimRotateComboBox.setItemText(1, _translate("MainWindow", "CCW"))
        self.stimDirection.setText(_translate("MainWindow", "Select stimulus direction:"))
        self.DirectioncomboBox.setItemText(0, _translate("MainWindow", "Horizontal"))
        self.DirectioncomboBox.setItemText(1, _translate("MainWindow", "Vertical"))
        self.EpochSelect.setText(_translate("MainWindow", "Select epoch:"))
        self.EpochcomboBox_3.setItemText(0, _translate("MainWindow", "Epoch 1 (3-30s)"))
        self.EpochcomboBox_3.setItemText(1, _translate("MainWindow", "Epoch 2 (63-93s)"))
        self.EpochcomboBox_3.setItemText(2, _translate("MainWindow", "Epoch 3 (123-153s)"))
        self.EpochcomboBox_3.setItemText(3, _translate("MainWindow", "Epoch 4 (183-213s)"))
        self.EpochcomboBox_3.setItemText(4, _translate("MainWindow", "Epoch 5 (243-273s)"))
        self.stimDirection_2.setText(_translate("MainWindow", "Output file:        Animal NOT set!"))
        self.InitPlotLabel.setText(_translate("MainWindow", "Graph of plotted wave"))
        self.InitPlotButton.setText(_translate("MainWindow", "Set CSV"))
        self.pushButton.setText(_translate("MainWindow", "Set Output folder"))
        self.StimSpeedLabel.setText(_translate("MainWindow", "Enter stimulus speed (deg/s):"))
        self.StimSpeedInput.setText(_translate("MainWindow", "5"))
        self.TopBotETMLabel.setText(_translate("MainWindow", "Top and Bottom plot:"))
        self.TopPointAdj.setText(_translate("MainWindow", "Top point adjustment"))
        self.BotPointAdj.setText(_translate("MainWindow", "Bottom point adjustment"))
        self.menuFile.setTitle(_translate("MainWindow", "&File"))
        self.menu_Open.setTitle(_translate("MainWindow", "&Open"))
        
        self.menu_View.setTitle(_translate("MainWindow", "&Analysis"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionSave.setText(_translate("MainWindow", "Set Mouse"))
        self.actionSave.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionExit.setText(_translate("MainWindow", "Select Output File"))
        self.actionExit.setShortcut(_translate("MainWindow", "Ctrl+E"))
        self.action_Open.setText(_translate("MainWindow", "Open"))
        self.action_Open.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.action_Sorter.setText(_translate("MainWindow", "Sort Data"))
        self.action_View_3D_plot.setText(_translate("MainWindow", "&3D plot"))
        self.action_View_3D_plot.setShortcut(_translate("MainWindow", "Ctrl+3"))
        self.action_View_2D_plot.setText(_translate("MainWindow", "&2D plot"))
        self.action_View_2D_plot.setShortcut(_translate("MainWindow", "Ctrl+2"))
        #self.actionExit_2.setText(_translate("MainWindow", "Exit"))
        self.actionCurrent_analysis_data.setText(_translate("MainWindow", "Current analysis data"))

#Call the UI_MainWindow class 
#Safely close application following X button
def window():
    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    
    MainWindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    w = window()
    