# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 12:45:48 2022

@author: james
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 13:42:47 2022

@author: james
"""

#%%
#Chunk 1: imports and working directory
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import array, linspace
import sklearn.cluster as sk
from sklearn.neighbors.kde import KernelDensity
from scipy.signal import argrelextrema
import warnings
import PySimpleGUI as sg
import csv, os
from pandas.core.common import SettingWithCopyWarning
from matplotlib.backend_bases import MouseButton
import math as m
from scipy.integrate import quad


warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

df = pd.DataFrame(dict(A=[1, 2, 3], B=[2, 3, 4]))
df[df['A'] > 2]['B'] = 5  # No warnings for the chained assignment!

#SET WORKING DIRECTORY
os.chdir("C:/Users/james/Documents/ETM counter/WD/Outputs")

#%%
#Chunk 1: Animal DF formation: this stores all the epoch data for a single animal
#RUN ONCE PER ANIMAL
ep_data = pd.DataFrame()
ecounter= 0


#%%
#Chunk 2: CSV selection
#Load CSV file containing epx and epyWave data from a single animal
wd = os.getcwd()

layout = [  
            [sg.Text("Choose a wave CSV file:")],
            [sg.InputText(key="-FILE_PATH-"), 
            sg.FileBrowse(initial_folder=wd, file_types=[("CSV Files", "*.csv")])],
            [sg.Button('Submit'), sg.Exit()]
        ]

window = sg.Window("Display CSV", layout)

def convert_csv_array(csv_address):
    file = open(csv_address)
    csv_reader = csv.reader(file)
    header = next(csv_reader)
    rows = []
    for row in csv_reader:
        rows.append(row)
    file.close()
    return rows

while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    elif event == "Submit":
        csv_address = values["-FILE_PATH-"]
        break
        
window.close()

#%%
#Chunk 3: Direction and Rotation options
#VERY IMPORTANT TO CORRECTLY SELECT OPTIONS CORRESPONDING TO YOUR DATA

import PySimpleGUI as sg

lay_dict = {1:"Direction",2:"Rotation"}

# ----------- Create the 3 layouts this Window will display -----------
layout1 = [[sg.Text('Select stimulus direction')],
           *[[sg.CB("Horizontal")],[sg.CB("Vertical")]]]

layout2 = [[sg.Text('Select stimulus rotation')],
           [sg.CB("CCW")],
           [sg.CB("CW")]]

# ----------- Create actual layout using Columns and a row of Buttons
layout = [[sg.Column(layout1, key='-COL1-'), sg.Column(layout2, visible=False, key='-COL2-')],
          [sg.Button('Next'), sg.Button('1'), sg.Button('2'), sg.Button('Exit')]]

window = sg.Window('Swapping the contents of a window', layout)

layout = 1  # The currently visible layout
while True:
    event, values = window.read()
    #print(event, values)
    if event in (None, 'Exit'):
        break
    if event == 'Next':
        window[f'-COL{layout}-'].update(visible=False)
        layout = layout + 1 if layout < 2 else 1
        window[f'-COL{layout}-'].update(visible=True)
    elif event in '12':
        window[f'-COL{layout}-'].update(visible=False)
        layout = int(event)
        window[f'-COL{layout}-'].update(visible=True)
window.close()
criteria_list = ["Horizontal","Vertical","CCW","CW"]
crit_dict = values
for index, value in enumerate(criteria_list):
    crit_dict[value] = crit_dict.pop(index)
    
criteria = {key:val for key, val in crit_dict.items() if val == True}
crit_list = list(criteria.keys())

direct = crit_list[0]
direction = crit_list[1]

if direct == "Horizontal":
    wave = "epxWave"
    wave2 = "epxWave2"
elif direct == "Vertical":
    wave = "epyWave"
    wave2 = "epyWave2"
else:
    print("Please put a Horizontal or Vertical")

#%%
#Chunk 3: Data import epoch selection
#RE-RUN THIS CHUNK TO SELECT EPOCHS OF THE SAME TRIAL

#Import data
db_raw=pd.read_csv(csv_address)
#db_raw=pd.read_csv("C:/Users/james/Documents/ETM counter/WD/025 temporal.csv")

lay_dict = {1:"Epoch"}

# ----------- Create the 3 layouts this Window will display -----------

layout1 = [[sg.Text('Select epoch')],
           *[[sg.CB(f'Epoch {i}', 0)] for i in range(1,6)]]

# ----------- Create actual layout using Columns and a row of Buttons
layout = [[sg.Column(layout1, key='-COL1-')], [ sg.Button('Exit')]]

window = sg.Window('Swapping the contents of a window', layout)

layout = 1  # The currently visible layout
while True:
    event, values = window.read()
    #print(event, values)
    if event in (None, 'Exit'):
        break
window.close()
epoch_list = ["Epoch 1","Epoch 2","Epoch 3","Epoch 4", "Epoch 5"]
ep_dict = values
for index, value in enumerate(epoch_list):
    ep_dict[value] = ep_dict.pop(index)

ep_crit = {key:val for key, val in ep_dict.items() if val == True}
ep_list = list(ep_crit.keys())

epoch = ep_list[0]

if epoch == "Epoch 1":
    b,e = 0,3300
if epoch == "Epoch 2":
    b,e = 6300,9300
if epoch == "Epoch 3":
    b,e = 12300, 15300
if epoch == "Epoch 4":
    b,e = 18300, 21300
if epoch == "Epoch 5":
    b,e = 24300,27300

db_1 = db_raw.loc[b:e]
db = db_1.reset_index()

#%%

#Filtering:

def Averaging(values):
    sum_of_list = 0
    for i in range(len(values)):
        sum_of_list += values[i]
    average = sum_of_list/len(values)
    return average

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
def Graphing(db,section):
    if section == "1":
        trimmed_db = db[0:3304]
    if section == "2":
        trimmed_db = db[6304:9304]
    if section == "3":
        trimmed_db = db[12040:15040]
    elif section == "4":
        trimmed_db = db[18340:21340]
    elif section == "5":
        trimmed_db = db[24304:27304]
    elif section == "total":
        trimmed_db = db[0:30604]
    plt.plot(wave, data = db)


def Slope(x1,y1,x2,y2):
    slope = 0
    y = (y2-y1)
    x = (x2-x1)
    if x !=0:
        slope = y/x
        return abs(slope)
    else:
        return "NA"
    
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

    
def Flatten(lt):
    final_lt=[]
    for x in lt:
        if type(x) is list:
            for y in x:
                final_lt.append(y)
        else:
            final_lt.append(x)
    return final_lt
    
def Derivative_grapher(db,section):
    if section == "1":
        trimmed_db = db[0:3304]
    if section == "2":
        trimmed_db = db[6304:9304]
    if section == "3":
        trimmed_db = db[12040:15040]
    elif section == "4":
        trimmed_db = db[18340:21340]
    elif section == "5":
        trimmed_db = db[24304:27304]
    elif section == "total":
        trimmed_db = db[0:30604]
    graph_db = db
    graph_db["X"] = db.index
    graph_db.plot(kind='scatter', x="X", y="Average Slope")
        
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
    mil = s[mini].tolist()
    mxl = s[maxi].tolist()
    
    final_mil = Flatten(mil)
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

def bot_scan(db, total_db):
    s_db = db
    t_db = total_db
    total_list = []
    max_list = []

    for x in s_db["X"]:
        start = t_db.loc[x,"X"]-40
        fast_list = []
        iterator = start
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

def top_scan(db, total_db):
    s_db = db
    t_db = total_db
    total_list = []
    max_list = []
    for x in s_db["X"]:
        start = t_db.loc[x,"X"]-30
        fast_list = []
        iterator = start
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

def Amp_filter(db, threshold):
    ampf = db[db["Amplitude (degrees)"]>threshold]
    return ampf
        
    
#Filtering: 
#Lead is set to 50, and threshold to 10
#Lead corresponds to how many points ahead to use to generate a moving average
#Threshold is the limit difference between the leading mean and the point value


fd = Filtering_alg(db,50,20) 

frame = "Total"
#Graphing(db,frame)
#fgraph = Graphing(fd,frame)

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

print(xdata_b)
#plt.scatter("X",wave, data = etm_df_bottom,c="b",s=100)

#%%
#Chunk 3: ETM supervision
#Left click to remove points, right click to add to nearest, middle click to finish and save the points (always do this)
#Can place anywhere along the fast phase saccade
#For some reason, if there are gaps in the data, the clicking won't be accurate so just keep clicking until it looks good

fig, ax = plt.subplots()
a = plt.plot(fd["X"],fd[wave],color='red',picker = 10)[0]
b = plt.scatter(etm_df_bottom["X"],etm_df_bottom[wave],color='b',s=300,picker=100)

def add_or_remove_point(event):
    global a
    xydata_a = np.stack(a.get_data(),axis=1)
    xdata_a = a.get_xdata()
    ydata_a = a.get_ydata()
    global b
    xydata_b = b.get_offsets()
    global xdata_b 
    xdata_b = b.get_offsets()[:,0]
    ydata_b = b.get_offsets()[:,1]    
    
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
        print("disconnecting")
        plt.disconnect(fig)
        
fig.canvas.mpl_connect('button_press_event',add_or_remove_point)

#%%
#Chunk 4: Prelim ETM calculator for top and bottom points and plot
#Check that all ETMs are there. If not, then re-do chunk 4

etm_middle = xdata_b
print(etm_middle)
fgraph = Graphing(fd,frame)

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
real4 = [x+8 for x in real3]

if direction == "CW":
    etm_top_final = fd[fd["X"].isin(real3)]
    etm_bot_final = fd[fd["X"].isin(real4)]

if direction == "CCW":
    etm_bot_final = fd[fd["X"].isin(real3)]
    etm_top_final = fd[fd["X"].isin(real4)]

plt.scatter("X",wave, data = etm_bot_final,c="g",s=300)
plt.scatter("X",wave, data = etm_top_final,c="r",s=300)
plt.scatter("X",wave, data = f_amp, c="b",s=300)

#%%
#Chunk 5: Adding and removing bot points
#If CCW, then for bottom points. If CW, then for top points

fig, ax = plt.subplots()
a = plt.plot(fd["X"],fd[wave],color='blue',picker = 10)[0]
b = plt.scatter(etm_top_final["X"],etm_top_final[wave],color='r',s=300,picker=100)

def add_or_remove_point(event):
    global a
    xydata_a = np.stack(a.get_data(),axis=1)
    xdata_a = a.get_xdata()
    ydata_a = a.get_ydata()
    global b
    xydata_b = b.get_offsets()
    global xdata_b_top
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
        print("disconnecting")
        plt.disconnect(fig)
        

fig.canvas.mpl_connect('button_press_event',add_or_remove_point)

#%%
#Chunk 6: Adding and removing top points
#If CCW, then this is for top points. If CW, then for bottom points
fig, ax = plt.subplots()
a = plt.plot(fd["X"],fd[wave],color='blue',picker = 10)[0]
b = plt.scatter(etm_bot_final["X"],etm_bot_final[wave],color='g',s=150,picker=100)

def add_or_remove_point(event):
    global a
    xydata_a = np.stack(a.get_data(),axis=1)
    xdata_a = a.get_xdata()
    ydata_a = a.get_ydata()
    global b
    xydata_b = b.get_offsets()
    global xdata_b_bot
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
        print("disconnecting")
        plt.disconnect(fig)
        

fig.canvas.mpl_connect('button_press_event',add_or_remove_point)

#%%
#Chunk 7: Analysis
#IF YOU GET A AN INDEX LENGTH ERROR, START AGAIN AT CHUNK 4 (you did not keep the number of top and bottoms consistent)

etm_top = fd[fd["X"].isin(xdata_b_top)]
etm_bot = fd[fd["X"].isin(xdata_b_bot)]
bot_lst_f = etm_bot["X"]

real5 = [x+3 for x in bot_lst_f]
etm_final_r = fd[fd["X"].isin(real5)]

fgraph = Graphing(fd,frame)

amp = Amplitude(etm_final_r,etm_top,etm_bot)
#Change number to threshold amplitude of ETM
final_amp = Amp_filter(amp,2)

plt.scatter("X",wave, data = etm_bot,c="g",s=300)
plt.scatter("X",wave, data = etm_top,c="r",s=300)
plt.scatter("X",wave, data = etm_final_r, c="b",s=300)
etms = len(final_amp["X"])
#print("# of ETMs: " + str(len(final_amp["X"])))
avg_amp = np.mean(final_amp["Amplitude (degrees)"])
#print("Average amplitude: " + str(avg_amp) + " degrees")

def ETM_Freq(db1):
    db1ri = db1.reset_index()
    xdist = []
    for x in db1ri.index:
        try:
            d = (db1ri.loc[x+1,"X"]-db1ri.loc[x,"X"])
            xdist.append(d)
        except:
            xdist.append(0)
    avg_freq = (sum(xdist)/(len(xdist)-1))/100
    return avg_freq
    
#dist = SlowPhaseGain(etm_top_final, etm_bot_final, f_amp)
#print("Average gain: " + str(dist))
freq = ETM_Freq(f_amp)
#print("Average time of slow phase: " + str(freq) + " seconds")


#%%
#Chunk 8: Final analysis
#It will print out data points in this order:
"""
Number of ETMs per 30s
Avg amplitude of saccade 
Avg gain of the slow phase in the X direction
Avg gain of the slow phase in the Y direction
Avg size of the total slow phase vector (how far the eye travels)
Avg size of y part of the slow phase vector (how far it travels vertically)
Avg velocity of y part of slow phase vector (how fast it travels vertically)
Avg size of x part of slow phase vector (how far travels horizontally)
Avg velocity of x part of slow phase vector (how fast it travels horizontally)

"""
#x
def tot_vector_magnitude_CCW(db_top, db_bot, db_total):
    top_ri = db_top.reset_index()
    bot_ri = db_bot.reset_index()
    list_1 = top_ri.loc[0:len(bot_ri.index),"X"]
    list_2 = top_ri.loc[0:len(bot_ri.index),"epxWave"]
    list_3 = top_ri.loc[0:len(bot_ri.index),"epyWave"]

    dbx = bot_ri
    dbx["X2"] = list_1
    dbx["epxWave2"] = list_2
    dbx["epyWave2"] = list_3
    
    
    distance_list=[]
    vel_list=[]
    for x in dbx.index:
        try:
            distance = np.sqrt((dbx.loc[x,"epyWave2"]-dbx.loc[x+1,"epyWave"])**2 + (dbx.loc[x,"epxWave2"]-dbx.loc[x+1,"epxWave"])**2)
            distance_list.append(distance)
        except:
            distance_list.append(float("NaN"))
            
    count=0
    try:
        for x in distance_list:
            vel = x/abs(dbx.loc[count+1,"X"]-dbx.loc[count,"X2"])
            
            count+=1
            real_vel = vel*100
            vel_list.append(real_vel)
    except:
        vel_list.append(float('NaN'))
    
    return distance_list
    #Gain calculation:
    """
    gain = [x/5 for x in vel_list]
    db_gain = dbx
    db_gain["Gain"] = gain
    
    #average gain:
    avg_g = np.mean(db_gain["Gain"])
    
    return avg_g
    """
def tot_vector_vel_CCW(db_top, db_bot, db_total):
    top_ri = db_top.reset_index()
    bot_ri = db_bot.reset_index()
    list_1 = bot_ri.loc[0:len(bot_ri.index),"X"]
    list_2 = bot_ri.loc[0:len(bot_ri.index),"epxWave"]
    list_3 = bot_ri.loc[0:len(bot_ri.index),"epyWave"]

    dbx = top_ri
    dbx["X2"] = list_1
    dbx["epxWave2"] = list_2
    dbx["epyWave2"] = list_3
    
    distance_list=[]
    vel_list=[]
    for x in dbx.index:
        try:
            distance = np.sqrt((dbx.loc[x,"epyWave"]-dbx.loc[x+1,"epyWave"])**2 + (dbx.loc[x,"epxWave2"]-dbx.loc[x+1,"epxWave"])**2)
            distance_list.append(distance)
        except:
            distance_list.append(float("NaN"))
    count=0
    try:
        for x in distance_list:
            vel = x/abs(dbx.loc[count+1,"X"]-dbx.loc[count,"X2"])
            
            count+=1
            real_vel = vel*100
            vel_list.append(real_vel)
    except:
        vel_list.append(float('NaN'))
    
    return vel_list

def y_vector_distance_CCW(db_top, db_bot, db_total):
    top_ri = db_top.reset_index()
    bot_ri = db_bot.reset_index()
    list_1 = bot_ri.loc[0:len(bot_ri.index),"X"]
    #list_2 = bot_ri.loc[0:len(bot_ri.index),"epxWave"]
    list_3 = bot_ri.loc[0:len(bot_ri.index),"epyWave"]

    dbx = top_ri
    dbx["X2"] = list_1
    #dbx["epxWave2"] = list_2
    dbx["epyWave2"] = list_3
    
    distance_list=[]
    
    for x in dbx.index:
        try:
            distance = abs(dbx.loc[x,"epyWave"]-dbx.loc[x+1,"epyWave2"])
            distance_list.append(distance)
            
        except:
            distance_list.append(float("NaN"))
    return distance_list
#X=top, X2=bot
def y_vector_velocity_CCW(lst,db_top, db_bot, db_total):
    top_ri = db_top.reset_index()
    bot_ri = db_bot.reset_index()
    list_1 = bot_ri.loc[0:len(bot_ri.index),"X"]
    #list_2 = bot_ri.loc[0:len(bot_ri.index),"epxWave"]
    list_3 = bot_ri.loc[0:len(bot_ri.index),"epyWave"]
    vel_list=[]
    dbx = top_ri
    dbx["X2"] = list_1
    #dbx["epxWave2"] = list_2
    dbx["epyWave2"] = list_3
    count=0
    try:
        for x in lst:
            vel = x/abs(dbx.loc[count+1,"X"]-dbx.loc[count,"X2"])
            
            count+=1
            real_vel = vel*100
            vel_list.append(real_vel)
    except:
        vel_list.append(float('NaN'))
        
    return vel_list

def y_vector_angle_CCW(lst1,lst2):
    ang_list = []
    for y, h in zip(lst1,lst2):
        yh = y/h
        angle = m.degrees(np.arcsin(yh))
        ang_list.append(angle)
    return ang_list

def x_vector_mag_CCW(db_bot, db_top,  db_total):
    top_ri = db_top.reset_index()
    bot_ri = db_bot.reset_index()
    list_1 = bot_ri.loc[0:len(bot_ri.index),"X"]
    list_2 = bot_ri.loc[0:len(bot_ri.index),"epxWave"]
   #list_3 = bot_ri.loc[0:len(bot_ri.index),"epyWave"]

    dbx = top_ri
    dbx["X2"] = list_1
    dbx["epxWave2"] = list_2
    #dbx["epyWave2"] = list_3
    
    distance_list=[]
    vel_list=[]
    for x in dbx.index:
        try:
            distance = abs(dbx.loc[x,"epxWave"]-dbx.loc[x+1,"epxWave2"])
            distance_list.append(distance)
        except:
            distance_list.append(float("NaN"))
        
    return distance_list

def x_vector_vel_CCW(lst, db_bot, db_top,  db_total):
    top_ri = db_top.reset_index()
    bot_ri = db_bot.reset_index()
    list_1 = bot_ri.loc[0:len(bot_ri.index),"X"]
    list_2 = bot_ri.loc[0:len(bot_ri.index),"epxWave"]
   #list_3 = bot_ri.loc[0:len(bot_ri.index),"epyWave"]

    dbx = top_ri
    dbx["X2"] = list_1
    dbx["epxWave2"] = list_2
    #dbx["epyWave2"] = list_3
    
    #Temporal:
    distance_list=[]
    vel_list=[]
    for x in dbx.index:
        try:
            distance = abs(dbx.loc[x,"epxWave"]-dbx.loc[x+1,"epxWave2"])
            distance_list.append(distance)
        except:
            distance_list.append(float("NaN"))
    count=0
    try:
        for x in distance_list:
            vel = x/abs(dbx.loc[count+1,"X"]-dbx.loc[count,"X2"])
            
            count+=1
            real_vel = vel*100
            vel_list.append(real_vel)
    except:
        vel_list.append(float('NaN'))
    return vel_list

def x_vector_angle_CCW(lst1,lst2):
    ang_list = []
    for x, h in zip(lst1,lst2):
        xh = x/h
        angle = m.degrees(np.arcsin(xh))
        ang_list.append(angle)
    return ang_list

def tot_vector_magnitude_CW(db_top, db_bot, db_total):
    top_ri = db_top.reset_index()
    bot_ri = db_bot.reset_index()
    list_1 = bot_ri.loc[0:len(bot_ri.index),"X"]
    list_2 = bot_ri.loc[0:len(bot_ri.index),"epxWave"]
    list_3 = bot_ri.loc[0:len(bot_ri.index),"epyWave"]

    dbx = top_ri
    dbx["X2"] = list_1
    dbx["epxWave2"] = list_2
    dbx["epyWave2"] = list_3
    
    distance_list=[]
    vel_list=[]
    for x in dbx.index:
        try:
            distance = np.sqrt((dbx.loc[x,"epyWave2"]-dbx.loc[x+1,"epyWave"])**2 + (dbx.loc[x,"epxWave2"]-dbx.loc[x+1,"epxWave"])**2)
            distance_list.append(distance)
        except:
            distance_list.append(float("NaN"))
            
    count=0
    try:
        for x in distance_list:
            vel = x/abs(dbx.loc[count+1,"X"]-dbx.loc[count,"X2"])
            
            count+=1
            real_vel = vel*100
            vel_list.append(real_vel)
    except:
        vel_list.append(float('NaN'))
    
    return distance_list
    #Gain calculation:
    """
    gain = [x/5 for x in vel_list]
    db_gain = dbx
    db_gain["Gain"] = gain
    
    #average gain:
    avg_g = np.mean(db_gain["Gain"])
    
    return avg_g
    """
def tot_vector_vel_CW(db_top, db_bot, db_total):
    top_ri = db_top.reset_index()
    bot_ri = db_bot.reset_index()
    list_1 = bot_ri.loc[0:len(bot_ri.index),"X"]
    list_2 = bot_ri.loc[0:len(bot_ri.index),"epxWave"]
    list_3 = bot_ri.loc[0:len(bot_ri.index),"epyWave"]
    dbx = top_ri
    dbx["X2"] = list_1
    dbx["epxWave2"] = list_2
    dbx["epyWave2"] = list_3
    
    distance_list=[]
    vel_list=[]
    for x in dbx.index:
        try:
            distance = np.sqrt((dbx.loc[x,"epyWave"]-dbx.loc[x+1,"epyWave"])**2 + (dbx.loc[x,"epxWave2"]-dbx.loc[x+1,"epxWave"])**2)
            distance_list.append(distance)
        except:
            distance_list.append(float("NaN"))
    count=0
    try:
        for x in distance_list:
            vel = x/abs(dbx.loc[count+1,"X"]-dbx.loc[count,"X2"])
            
            count+=1
            real_vel = vel*100
            vel_list.append(real_vel)
    except:
        vel_list.append(float('NaN'))
    
    return vel_list

def y_vector_distance_CW(db_top, db_bot, db_total):
    top_ri = db_top.reset_index()
    bot_ri = db_bot.reset_index()
    list_1 = bot_ri.loc[0:len(bot_ri.index),"X"]
    #list_2 = bot_ri.loc[0:len(bot_ri.index),"epxWave"]
    list_3 = bot_ri.loc[0:len(bot_ri.index),"epyWave"]

    dbx = top_ri
    dbx["X2"] = list_1
    #dbx["epxWave2"] = list_2
    dbx["epyWave2"] = list_3
    
    distance_list=[]
    
    for x in dbx.index:
        try:
            distance = abs(dbx.loc[x,"epyWave2"]-dbx.loc[x+1,"epyWave"])
            distance_list.append(distance)
            
        except:
            distance_list.append(float("NaN"))
    
    return distance_list

def y_vector_velocity_CW(lst,db_top, db_bot, db_total):
    top_ri = db_top.reset_index()
    bot_ri = db_bot.reset_index()
    list_1 = bot_ri.loc[0:len(bot_ri.index),"X"]
    #list_2 = bot_ri.loc[0:len(bot_ri.index),"epxWave"]
    list_3 = bot_ri.loc[0:len(bot_ri.index),"epyWave"]
    vel_list=[]
    dbx = top_ri
    dbx["X2"] = list_1
    #dbx["epxWave2"] = list_2
    dbx["epyWave2"] = list_3
    count=0
    try:
        for x in lst:
            vel = x/abs(dbx.loc[count+1,"X2"]-dbx.loc[count,"X"])
            
            count+=1
            real_vel = vel*100
            vel_list.append(real_vel)
    except:
        vel_list.append(float('NaN'))
        
    return vel_list

def y_vector_angle_CW(lst1,lst2):
    ang_list = []
    for y, h in zip(lst1,lst2):
        yh = y/h
        angle = m.degrees(np.arcsin(yh))
        ang_list.append(angle)
    return ang_list

def x_vector_mag_CW(db_bot, db_top,  db_total):
    top_ri = db_bot.reset_index()
    bot_ri = db_top.reset_index()
    list_1 = bot_ri.loc[0:len(bot_ri.index),"X"]
    list_2 = bot_ri.loc[0:len(bot_ri.index),"epxWave"]
    #list_3 = bot_ri.loc[0:len(bot_ri.index),"epyWave"]

    dbx = top_ri
    dbx["X2"] = list_1
    dbx["epxWave2"] = list_2
    #dbx["epyWave2"] = list_3
    
    distance_list=[]
    vel_list=[]
    for x in dbx.index:
        try:
            distance = abs(dbx.loc[x,"epxWave2"]-dbx.loc[x+1,"epxWave"])
            distance_list.append(distance)
        except:
            distance_list.append(float("NaN"))
        
    return distance_list

def x_vector_vel_CW(lst, db_bot, db_top,  db_total):
    top_ri = db_bot.reset_index()
    bot_ri = db_top.reset_index()
    list_1 = bot_ri.loc[0:len(bot_ri.index),"X"]
    list_2 = bot_ri.loc[0:len(bot_ri.index),"epxWave"]
   #list_3 = bot_ri.loc[0:len(bot_ri.index),"epyWave"]

    dbx = top_ri
    dbx["X2"] = list_1
    dbx["epxWave2"] = list_2
    #dbx["epyWave2"] = list_3
    
    #Temporal:
    distance_list=[]
    vel_list=[]
    for x in dbx.index:
        try:
            distance = abs(dbx.loc[x,"epxWave2"]-dbx.loc[x+1,"epxWave"])
            distance_list.append(distance)
        except:
            distance_list.append(float("NaN"))
    count=0
    try:
        for x in distance_list:
            vel = x/abs(dbx.loc[count+1,"X"]-dbx.loc[count,"X2"])
            
            count+=1
            real_vel = vel*100
            vel_list.append(real_vel)
    except:
        vel_list.append(float('NaN'))
    return vel_list

def x_vector_angle(lst1,lst2):
    ang_list = []
    for x, h in zip(lst1,lst2):
        xh = x/h
        angle = m.degrees(np.arcsin(xh))
        ang_list.append(angle)
    return ang_list

if direction == "CCW":
    atot_vec_d = tot_vector_magnitude_CCW(etm_top,etm_bot,etm_final_r)
    ay_vec_d = y_vector_distance_CCW(etm_top,etm_bot,etm_final_r)
    ay_vec_v = y_vector_velocity_CCW(ay_vec_d,etm_top,etm_bot,etm_final_r)
    #ay_vec_ang = y_vector_angle(ay_vec_d, atot_vec_d)
    ax_vec_d = x_vector_mag_CCW(etm_top,etm_bot,etm_final_r)
    ax_vec_v = x_vector_vel_CCW(ay_vec_d,etm_top,etm_bot,etm_final_r)
    #ax_vec_ang = x_vector_angle(ax_vec_d, atot_vec_d)
elif direction == "CW":
    atot_vec_d = tot_vector_magnitude_CW(etm_top,etm_bot,etm_final_r)
    ay_vec_d = y_vector_distance_CW(etm_top,etm_bot,etm_final_r)
    ay_vec_v = y_vector_velocity_CW(ay_vec_d,etm_top,etm_bot,etm_final_r)
    #ay_vec_ang = y_vector_angle(ay_vec_d, atot_vec_d)
    ax_vec_d = x_vector_mag_CW(etm_top,etm_bot,etm_final_r)
    ax_vec_v = x_vector_vel_CW(ay_vec_d,etm_top,etm_bot,etm_final_r)
    #ax_vec_ang = x_vector_angle(ax_vec_d, atot_vec_d)

tot_vec_d = [x for x in atot_vec_d if not(m.isnan(x)) == True]
y_vec_d = [x for x in ay_vec_d if not(m.isnan(x)) == True]
y_vec_v = [x for x in ay_vec_v if not(m.isnan(x)) == True]
#y_vec_ang = [x for x in ay_vec_ang if not(m.isnan(x)) == True]

x_vec_d = [x for x in ax_vec_d if not(m.isnan(x)) == True]
x_vec_v = [x for x in ax_vec_v if not(m.isnan(x)) == True]
#x_vec_ang = [x for x in ax_vec_ang if not(m.isnan(x)) == True]

avg_totvec = Averaging(tot_vec_d)
avg_yd = Averaging(y_vec_d)
avg_yv = Averaging(y_vec_v)
avg_yg = avg_yv/5
#avg_ya = Averaging(y_vec_ang)

avg_xd = Averaging(x_vec_d)
avg_xv = Averaging(x_vec_v)
#avg_xa = Averaging(x_vec_ang)
avg_xg = avg_xv/5

data_list = [etms,avg_amp,avg_xg,avg_yg,avg_totvec,avg_yd,avg_yv,avg_xd,avg_xv]
#ADDS DATA TO TOTAL DATAFRAME
e_c = str(ecounter)
ep_data[epoch+"_"+direct+"_"+direction+"_"+e_c] = data_list
ecounter+=1

"""
print(etms)
print(avg_amp)
print(avg_xg) #avg x gain
print(avg_yg) #avg y gain
print(avg_totvec)
print(avg_yd)
print(avg_yv)
#print(avg_ya)

print(avg_xd)
print(avg_xv)
#print(avg_xa)
"""


#%%
#DO THIS LAST WHEN DONE WITH ANIMAL
ep_data["Mean"] = ep_data.mean(axis=1)
ep_data.index = ["ETMs per 30s","Avg Saccade Amplitude","Avg X gain","Avg Y gain","Avg total vector magnitude","Avg Y magnitude","Avg Y velocity","Avg X magnitude","Avg X velocity"]

cwd = os.getcwd()
#CHANGE BASED ON YOUR MOUSE
path = cwd + "\Mmp12_199_t1_005.csv"
df.to_csv(path)

ep_data.to_csv(path)



#%%
#Chunk 9: 3D Graph
# X-axis = Time, Y-axis = Horizontal data, Z-axis = Vertical data
# Blue lines are approximations of slope of slow phases
# Looks cool

def graph(fd):
    x = fd["X"]
    y = fd["epxWave"]
    z = fd["epyWave"]    
    
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.grid()
    
    ax.scatter(x, y, z, c = 'r', s = 50)
    ax.set_title('3D Scatter Plot')
    
    # Set axes label
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)
    
    plt.plot(x, y, z, '.r-') 
    plt.show()

def vec_graph_CCW(db_top, db_bot, db_total):
    top_ri = db_top.reset_index()
    bot_ri = db_bot.reset_index()
    list_1 = bot_ri.loc[0:len(bot_ri.index),"X"]
    list_2 = bot_ri.loc[0:len(bot_ri.index),"epxWave"]
    list_3 = bot_ri.loc[0:len(bot_ri.index),"epyWave"]

    dbx = top_ri
    dbx["X2"] = list_1
    dbx["epxWave2"] = list_2
    dbx["epyWave2"] = list_3
    
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.grid()
    try:
        for a in dbx.index:
            x = [int(dbx.loc[a,"X"]),int(dbx.loc[a+1,"X2"])]
            y = [int(dbx.loc[a,"epxWave"]),int(dbx.loc[a+1,"epxWave2"])]
            z = [int(dbx.loc[a,"epyWave"]),int(dbx.loc[a+1,"epyWave2"])]
            
            ax.scatter(x, y, z, c = 'b', s = 50)
            #ax.set_title('3D Scatter Plot')
            ax.set_xlabel('x', labelpad=20)
            ax.set_ylabel('y', labelpad=20)
            ax.set_zlabel('z', labelpad=20)
        
            plt.plot(x, y, z, '.b-',linewidth=7) 
            plt.show()
           
    except:
        plt.plot(x, y, z, '.b-') 
        plt.show()
    
    x = fd["X"]
    y = fd["epxWave"]
    z = fd["epyWave"]
            
    plt.plot(x, y, z, '.r-') 
    plt.show()

def vec_graph_CW(db_top, db_bot, db_total):
    top_ri = db_top.reset_index()
    bot_ri = db_bot.reset_index()
    list_1 = top_ri.loc[0:len(bot_ri.index),"X"]
    list_2 = top_ri.loc[0:len(bot_ri.index),"epxWave"]
    list_3 = top_ri.loc[0:len(bot_ri.index),"epyWave"]

    dbx = bot_ri
    dbx["X2"] = list_1
    dbx["epxWave2"] = list_2
    dbx["epyWave2"] = list_3
    
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.grid()
    try:
        for a in dbx.index:
            x = [int(dbx.loc[a,"X"]),int(dbx.loc[a+1,"X2"])]
            y = [int(dbx.loc[a,"epxWave"]),int(dbx.loc[a+1,"epxWave2"])]
            z = [int(dbx.loc[a,"epyWave"]),int(dbx.loc[a+1,"epyWave2"])]
            
            ax.scatter(x, y, z, c = 'b', s = 50)
            #ax.set_title('3D Scatter Plot')
            ax.set_xlabel('x', labelpad=20)
            ax.set_ylabel('y', labelpad=20)
            ax.set_zlabel('z', labelpad=20)
        
            plt.plot(x, y, z, '.b-',linewidth=7) 
            plt.show()
           
    except:
        plt.plot(x, y, z, '.b-') 
        plt.show()
    
    x = fd["X"]
    y = fd["epxWave"]
    z = fd["epyWave"]
            
    plt.plot(x, y, z, '.r-') 
    plt.show()
#direction="CCW"

if direction == "CW":
    vec_graph_CW(etm_top,etm_bot,etm_final_r)
    
if direction == "CCW":
    vec_graph_CCW(etm_top,etm_bot,etm_final_r)







