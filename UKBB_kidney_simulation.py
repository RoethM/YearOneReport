# Importing required variables
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
import pandas as pd
from sklearn.metrics import r2_score

plt.close('all');

####################################################
#Functions
####################################################

def rotx(alpha, m): # function to perform rotation in x
    mx = m[0]
    my = math.cos(alpha)*m[1] + math.sin(alpha)*m[2]
    mz = -math.sin(alpha)*m[1] + math.cos(alpha)*m[2]
    m[0] = mx
    m[1] = my
    m[2] = mz
    
    return m

def bFFErelrot(alpha, m, t, T2, T1, M0): #function to perform bFFE relaxation 
    mx = (math.exp(-t/T2))*(math.cos(alpha)*m[0] + math.sin(alpha)*m[1])
    my = (math.exp(-t/T2))*(-math.sin(alpha)*m[0] + math.cos(alpha)*m[1])
    mz = m[2]*math.exp(-t/T1) + M0*(1-math.exp(-t/T1))
    m[0] = mx 
    m[1] = my
    m[2] = mz
    return m

def relrot(alpha, m, t_small, split, T2, T1, M0): #function to perform relaxation
    s_xy = np.zeros(split)#, 1))
    s_z = np.zeros(split)#, 1))
    mx = (math.exp(-t_small/T2))*(math.cos(alpha)*m[0] + math.sin(alpha)*m[1])
    my = (math.exp(-t_small/T2))*(-math.sin(alpha)*m[0] + math.cos(alpha)*m[1])
    mz = m[2]*math.exp(-t_small/T1) + M0*(1-math.exp(-t_small/T1))
    m[0] = mx
    m[1] = my
    m[2] = mz
    s_xy[0] = np.sqrt(m[0]*m[0]+m[1]*m[1])
    s_z[0] = m[2]
    
    for k in range(1, split):
        mx = math.exp(-t_small/T2)*(math.cos(alpha)*m[0] + math.sin(alpha)*m[1])
        my = math.exp(-t_small/T2)*(-math.sin(alpha)*m[0] + math.cos(alpha)*m[1])
        mz = m[2]*math.exp(-t_small/T1) + M0*(1-math.exp(-t_small/T1))
        m[0] = mx
        m[1] = my
        m[2] = mz
        s_xy[k] = np.sqrt(m[0]*m[0]+m[1]*m[1])
        s_z[k] = m[2]
        
    return m, s_xy, s_z

def fun_bffe(m, TR_bFFE, Total_shots, T2, T1_input, M0, alpha, readout_shots, reference_shots, startup_shots): # function to simulate fisp signal
    s_xy = np.zeros(Total_shots)#, 1))
    s_z = np.zeros(Total_shots)#, 1))
    
    # start up cycle - 1/2 alpha
    m = rotx(-alpha/2, m)
    s_xy[0] = np.sqrt(m[0]*m[0] + m[1]*m[1]) #? is this part needed?
    s_z[0] = m[2] #? is this part needed?
    m = bFFErelrot(alpha, m, TR_bFFE/2, T2, T1_input, M0)
    
    # rest of startup cycle 
    m = rotx(alpha, m)
    s_xy[1] = np.sqrt(m[0]*m[0] + m[1]*m[1]) #? is this part needed?
    s_z[1] = m[2] #? is this part needed?
    m = bFFErelrot(alpha, m, TR_bFFE, T2, T1_input, M0)
    
    for i in range(2, startup_shots, 2): #still startup cycle
        m = rotx(-alpha, m)
        s_xy[i] = np.sqrt(m[0]*m[0] + m[1]*m[1])
        s_z[i] = m[2]
        m = bFFErelrot(alpha, m, TR_bFFE, T2, T1_input, M0)
        m = rotx(alpha, m)
        s_xy[i+1] = np.sqrt(m[0]*m[0] + m[1]*m[1])
        s_z[i+1] = m[2]
        m = bFFErelrot(alpha, m, TR_bFFE, T2, T1_input, M0)
    
    
    # main pulses
    for i in range(startup_shots, readout_shots+startup_shots, 2):
        m = rotx(-alpha, m)
        s_xy[i] = np.sqrt(m[0]*m[0] + m[1]*m[1])
        s_z[i] = m[2]
        m = bFFErelrot(alpha, m, TR_bFFE, T2, T1_input, M0)
        m = rotx(alpha, m)
        s_xy[i+1] = np.sqrt(m[0]*m[0] + m[1]*m[1])
        s_z[i+1] = m[2]
        m = bFFErelrot(alpha, m, TR_bFFE, T2, T1_input, M0)
        
        
    # reference lines 
    for i in range(readout_shots+startup_shots, reference_shots+readout_shots+startup_shots, 2):
        m = rotx(-alpha, m)
        s_xy[i] = np.sqrt(m[0]*m[0] + m[1]*m[1])
        s_z[i] = m[2]
        m = bFFErelrot(alpha, m, TR_bFFE, T2, T1_input, M0)
        m = rotx(alpha, m)
        s_xy[i+1] = np.sqrt(m[0]*m[0] + m[1]*m[1])
        s_z[i+1] = m[2]
        m = bFFErelrot(alpha, m, TR_bFFE, T2, T1_input, M0)
    
    
    return m, s_xy, s_z


# Simulation for a 5(1)1(1)1 sequence
def signalsim51111(inv, m, t_small, split, T2, T1_input, M0, alpha, Mxy_to_plot, Mz_to_plot, T_for_mag, heart_rate, Acq_time, first_TI, second_TI, third_TI, delay1, delay2, delay3): 
    # recover
    t_small = first_TI/split
    m, M_xy_all, M_z_all = relrot(inv, m, t_small, split, T2, T1_input, M0)
    Mxy_to_plot = np.append(Mxy_to_plot, M_xy_all)
    Mz_to_plot = np.append(Mz_to_plot, M_z_all)
    T_for_mag = np.append(T_for_mag, np.linspace(delay1+t_small, delay1+first_TI, num=split))
    
    #images 1-4
    Mxy_only = 0
    Mz_only = 0
    T_for_bFFE = 0
    
    for i in range(1, 5):
        m, M_xy_all, M_z_all = fun_bffe(m, TR_bFFE, Total_shots, T2, T1_input, M0, alpha, readout_shots, reference_shots, startup_shots)
        Mxy_to_plot = np.append(Mxy_to_plot, M_xy_all)
        Mz_to_plot = np.append(Mz_to_plot, M_z_all)
        T_for_mag = np.append(T_for_mag, np.linspace((i*heart_rate-Acq_time+TR_bFFE), (i*heart_rate), num=Total_shots))
        Mxy_only = np.append(Mxy_only, M_xy_all)
        Mz_only = np.append(Mz_only, M_z_all)
        T_for_bFFE = np.append(T_for_bFFE, np.arange((i*heart_rate-Acq_time+TR_bFFE), (i*heart_rate), TR_bFFE))
        
        # recover
        t_small = Recov_time/split
        m, M_xy_all, M_z_all = relrot(inv, m, t_small, split, T2, T1_input, M0)
        Mxy_to_plot = np.append(Mxy_to_plot, M_xy_all)
        Mz_to_plot = np.append(Mz_to_plot, M_z_all)
        T_for_mag = np.append(T_for_mag, np.linspace((i*heart_rate+t_small), ((i+1)*heart_rate-Acq_time), num=split))
    
    #image 5
    m, M_xy_all, M_z_all = fun_bffe(m, TR_bFFE, Total_shots, T2, T1_input, M0, alpha, readout_shots, reference_shots, startup_shots)
    Mxy_to_plot = np.append(Mxy_to_plot, M_xy_all)
    Mz_to_plot = np.append(Mz_to_plot, M_z_all)
    T_for_mag = np.append(T_for_mag, np.linspace((5*heart_rate-Acq_time+TR_bFFE), (5*heart_rate), num=Total_shots))
    Mxy_only = np.append(Mxy_only, M_xy_all)
    Mz_only = np.append(Mz_only, M_z_all)
    T_for_bFFE = np.append(T_for_bFFE, np.arange((5*heart_rate-Acq_time+TR_bFFE), (5*heart_rate), TR_bFFE))
        
    #recovery period of 1 heart beats
    t_small = (heart_rate+delay2)/split
    m, M_xy_all, M_z_all = relrot(inv, m, t_small, split, T2, T1_input, M0)
    Mxy_to_plot = np.append(Mxy_to_plot, M_xy_all)
    Mz_to_plot = np.append(Mz_to_plot, M_z_all)
    T_for_mag = np.append(T_for_mag, np.linspace((5*heart_rate+t_small), (6*heart_rate+delay2), num=split))
    
    #apply a second 180 pulse
    m = rotx(inv, m)
    Mz_to_plot = np.append(Mz_to_plot, m[2])
    Mxy_to_plot = np.append(Mxy_to_plot, np.sqrt(m[0]*m[0]+m[1]*m[1]))
    T_for_mag = np.append(T_for_mag, 6*heart_rate+delay2)
    
    # recover
    t_small = second_TI/split
    m, M_xy_all, M_z_all = relrot(inv, m, t_small, split, T2, T1_input, M0)
    Mxy_to_plot = np.append(Mxy_to_plot, M_xy_all)
    Mz_to_plot = np.append(Mz_to_plot, M_z_all)
    T_for_mag = np.append(T_for_mag, np.linspace(6*heart_rate+delay2+t_small, 7*heart_rate-Acq_time, num=split))
    
    # image 6
    m, M_xy_all, M_z_all = fun_bffe(m, TR_bFFE, Total_shots, T2, T1_input, M0, alpha, readout_shots, reference_shots, startup_shots)
    Mxy_to_plot = np.append(Mxy_to_plot, M_xy_all)
    Mz_to_plot = np.append(Mz_to_plot, M_z_all)
    T_for_mag = np.append(T_for_mag, np.linspace((7*heart_rate-Acq_time+TR_bFFE), (7*heart_rate), num=Total_shots))
    Mxy_only = np.append(Mxy_only, M_xy_all)
    Mz_only = np.append(Mz_only, M_z_all)
    T_for_bFFE = np.append(T_for_bFFE, np.arange((7*heart_rate-Acq_time+TR_bFFE), (7*heart_rate), TR_bFFE))
    
    # recovery period of 1 heart beats
    t_small = (heart_rate+delay3)/split
    m, M_xy_all, M_z_all = relrot(inv, m, t_small, split, T2, T1_input, M0)
    Mxy_to_plot = np.append(Mxy_to_plot, M_xy_all)
    Mz_to_plot = np.append(Mz_to_plot, M_z_all)
    T_for_mag = np.append(T_for_mag, np.linspace((7*heart_rate+t_small), (8*heart_rate+delay3), num=split))
    
    # apply a third 180 pulse
    m = rotx(inv, m)
    Mz_to_plot = np.append(Mz_to_plot, m[2])
    Mxy_to_plot = np.append(Mxy_to_plot, np.sqrt(m[0]*m[0]+m[1]*m[1]))
    T_for_mag = np.append(T_for_mag, 8*heart_rate+delay3)
    
    # recover
    t_small = second_TI/split
    m, M_xy_all, M_z_all = relrot(inv, m, t_small, split, T2, T1_input, M0)
    Mxy_to_plot = np.append(Mxy_to_plot, M_xy_all)
    Mz_to_plot = np.append(Mz_to_plot, M_z_all)
    T_for_mag = np.append(T_for_mag, np.linspace(8*heart_rate+delay2+t_small, 9*heart_rate-Acq_time, num=split))
    
    # image 6
    m, M_xy_all, M_z_all = fun_bffe(m, TR_bFFE, Total_shots, T2, T1_input, M0, alpha, readout_shots, reference_shots, startup_shots)
    Mxy_to_plot = np.append(Mxy_to_plot, M_xy_all)
    Mz_to_plot = np.append(Mz_to_plot, M_z_all)
    T_for_mag = np.append(T_for_mag, np.linspace((9*heart_rate-Acq_time+TR_bFFE), (9*heart_rate), num=Total_shots))
    Mxy_only = np.append(Mxy_only, M_xy_all)
    Mz_only = np.append(Mz_only, M_z_all)
    T_for_bFFE = np.append(T_for_bFFE, np.arange((9*heart_rate-Acq_time+TR_bFFE), (9*heart_rate), TR_bFFE))

    return Mxy_to_plot, Mz_to_plot, T_for_mag, T_for_bFFE, Mxy_only, Mz_only

# defining function for fitting the points to a curve
def func(ti, A, B, t1_star):
    return A - B*(np.exp(-ti/t1_star))


def fitting51111(T_for_bFFE, Mxy_only, heart_rate, startup_shots, readout_shots, Total_shots):

    # Find center of k-space (partial fourier assumed)
    kspace_center = int(startup_shots + (readout_shots/3))

    # Center of bFFE shots for curvefitting 
    cen1 = T_for_bFFE[kspace_center] - delay1
    cen2 = T_for_bFFE[kspace_center + Total_shots] - delay1
    cen3 = T_for_bFFE[kspace_center + 2*Total_shots] - delay1
    cen4 = T_for_bFFE[kspace_center + 3*Total_shots] - delay1
    cen5 = T_for_bFFE[kspace_center + 4*Total_shots] - delay1
    cen6 = T_for_bFFE[kspace_center + 5*Total_shots] - delay2 -(6*heart_rate)
    cen7 = T_for_bFFE[kspace_center + 6*Total_shots] - delay3 -(8*heart_rate)
    T_to_fit = np.array([cen1, cen2, cen3, cen4, cen5, cen6, cen7])

    cen1 = Mxy_only[kspace_center]
    cen2 = Mxy_only[kspace_center + Total_shots]
    cen3 = Mxy_only[kspace_center + 2*Total_shots]
    cen4 = Mxy_only[kspace_center + 3*Total_shots]
    cen5 = Mxy_only[kspace_center + 4*Total_shots]
    cen6 = Mxy_only[kspace_center + 5*Total_shots]
    cen7 = Mxy_only[kspace_center + 6*Total_shots]
    Mxy_to_fit = np.array([cen1, cen2, cen3, cen4, cen5, cen6, cen7])

    # Fitting the plot and assigning the estimated variables
    plotlist5 = [0, 1, 2, 3, 4]
    popt, pcov = curve_fit(func, T_to_fit[plotlist5], Mxy_to_fit[plotlist5], maxfev=200000)#, p0=[0.5, 1.4, T1_input], maxfev=5000)
    A5, B5, t1_star5 = popt
    t1_corrected5 = t1_star5*((B5/A5) -1)
    
    # Calculating R^2
    y_pred = func(T_to_fit[plotlist5], *popt)
    r2_5 = r2_score(Mxy_to_fit[plotlist5], y_pred)
    
    # Calculating FE value
    FE5 = np.sqrt(sum((Mxy_to_fit[plotlist5] - func(T_to_fit[plotlist5], *popt))**2) / 4)
    #print(f'FE5: {FE5}')
    
    if t1_corrected5 > heart_rate:
        t1_star = t1_star5
        t1_corrected = t1_corrected5
        r2 = r2_5
        FE = FE5
        fit_type = 5
        A = A5
        B = B5
        plotlist = plotlist5
        #print('first five')
    
    else:
        
        #####################################################################
        #fitting for 6 points
        plotlist6 = [0, 5, 1, 2, 3, 4]
        
        # Fitting the plot and assigning the estimated variables
        popt, pcov = curve_fit(func, T_to_fit[plotlist6], Mxy_to_fit[plotlist6], maxfev=200000)#, p0=[0.5, 1.4, T1_input], maxfev=10000)
        A6, B6, t1_star6 = popt
        t1_corrected6 = t1_star6*((B6/A6) -1)
        
        # Calculating R^2
        y_pred = func(T_to_fit[plotlist6], *popt)
        r2_6 = r2_score(Mxy_to_fit[plotlist6], y_pred)
        
        # Calculating FE value
        FE6 = np.sqrt(sum((Mxy_to_fit[plotlist6] - func(T_to_fit[plotlist6], *popt)) **2) / 5)
        #print(f'FE6: {FE6}')
    
        
        #####################################################################
        #fitting for 7 points
        plotlist7 = [0, 5, 6, 1, 2, 3, 4]
        
        # Fitting the plot and assigning the estimated variables
        popt, pcov = curve_fit(func, T_to_fit[plotlist7], Mxy_to_fit[plotlist7], maxfev=200000)#, p0=[0.5, 1.4, T1_input], maxfev=15000)
        A7, B7, t1_star7 = popt
        t1_corrected7 = t1_star7*((B7/A7) -1)
        
        # Calculating R^2
        y_pred = func(T_to_fit[plotlist7], *popt)
        r2_7 = r2_score(Mxy_to_fit[plotlist7], y_pred)
        
        # Calculating FE value
        FE7 = np.sqrt(sum((Mxy_to_fit[plotlist7] - func(T_to_fit[plotlist7], *popt)) **2) / 6)
        #print(f'FE7: {FE7}')

        
        if (t1_corrected7*FE7) < (0.4*FE5*heart_rate):
            t1_star = t1_star7
            t1_corrected = t1_corrected7
            r2 = r2_7
            FE = FE7
            A = A7
            B = B7
            fit_type = 7
            plotlist = plotlist7
            #print('7')
            
        elif (t1_corrected6*FE6) < (FE5*heart_rate):
            t1_star = t1_star6
            t1_corrected = t1_corrected6
            r2 = r2_6
            FE = FE6
            A = A6
            B = B6
            fit_type = 6
            plotlist = plotlist6
            #print('6')
        
        else:
            t1_star = t1_star5
            t1_corrected = t1_corrected5
            r2 = r2_5
            FE = FE5
            A = A5
            B = B5
            fit_type = 5
            plotlist = plotlist5
            #print('5')
            
        
    return t1_star, t1_corrected, r2, FE, fit_type, T_to_fit, Mxy_to_fit, plotlist, A, B


####################################################
# Running functions
####################################################
#Input values
T2 = 100/1000

alpha = 30*np.pi/180
inv = np.pi # 180*pi/180

TR_bFFE = 0.0034
#TR_bFFE = 0.0048

M0 = 1

startup_shots = 18
readout_shots = 96
reference_shots = 24
Total_shots = startup_shots + readout_shots + reference_shots

split = 100

first_TI = 0.156
second_TI = 0.206
third_TI = 0.256
#first_TI = 0.17
#second_TI = 0.22
#third_TI = 0.270

################################################################################
'''
heart_rate = 0.6
T1_input = 0.8


# Empty arrays to be filled/ variables to recalculate
Acq_time = TR_bFFE*Total_shots + TR_bFFE/2
Recov_time = heart_rate - Acq_time
t_small = Recov_time/split
delay1 = heart_rate-first_TI-Acq_time
delay2 = heart_rate-second_TI-Acq_time
delay3 = heart_rate-third_TI-Acq_time
Mz_to_plot = np.array([M0, M0, -M0])
Mxy_to_plot = np.array([0.0, 0.0, 0.0])
T_for_mag = np.array([0.0, delay1, delay1])
m = np.array([0.0, 0.0, 0.0])
m[0] = 0 
m[1] = 0  
m[2] = -M0

# Running the simulation to gain the signal
Mxy_to_plot, Mz_to_plot, T_for_mag, T_for_bFFE, Mxy_only, Mz_only = signalsim51111(inv, m, t_small, split, T2, T1_input, M0, alpha, Mxy_to_plot, Mz_to_plot, T_for_mag, heart_rate, Acq_time, first_TI, second_TI, third_TI, delay1, delay2, delay3)

length = len(Mz_to_plot)
for i in range(0, length): 
    if Mz_to_plot[i] < 0:
        Mxy_to_plot[i] = -(Mxy_to_plot[i])

length = len(Mz_only)
for i in range(0, length): 
    if Mz_only[i] < 0:
        Mxy_only[i] = -(Mxy_only[i]) 

t1_star, t1_corrected, r2, FE, fit_type, T_to_fit, Mxy_to_fit, plotlist, A, B = fitting51111(T_for_bFFE, Mxy_only, heart_rate, startup_shots, readout_shots, Total_shots)


plt.figure()
plt.plot(T_for_mag, Mxy_to_plot,'b-')
plt.title('Mxy - Kidney')
plt.plot(T_for_mag, Mxy_to_plot,'b-')
plt.plot(T_for_bFFE[1:], Mxy_only[1:],'r.')
plt.plot(T_to_fit[plotlist], func(T_to_fit[plotlist], A, B, t1_star), 'r--')

plt.figure()
plt.plot(T_for_mag, Mz_to_plot,'b-')
plt.title('Mz - Kidney')
plt.plot(T_for_mag, Mz_to_plot,'b-')
plt.plot(T_for_bFFE[1:], Mz_only[1:],'r.')
#plt.plot(T_to_fit[plotlist], func(T_to_fit[plotlist], A, B, t1_star), 'r--')

'''
################################################################################


# T1 and RR lists for loop
hr_list = np.array([1.80, 1.81, 1.82, 1.83, 1.84, 1.85, 1.86, 1.87, 1.88, 1.89]) # List used in calculations
#hr_list = np.array([30, 45, 60, 75, 90])
#hr_list = 60/hr_list

t1_list = np.arange(0.4, 2.1, 0.1)

# Setting up empty dataframes with column names 
df_51111 = pd.DataFrame(columns = ['Input T1', 'RR = 1.80', 'RR = 1.81', 'RR = 1.82', 'RR = 1.83', 'RR = 1.84', 'RR = 1.85', 'RR = 1.86', 'RR = 1.87', 'RR = 1.88', 'RR = 1.89']) 
#df_51111 = pd.DataFrame(columns = ['Input T1', 'HR = 30', 'HR = 45', 'HR = 60', 'HR = 75', 'HR = 90'])

# Loop for T1 and RR values
for j in t1_list: # Range of t1 input values 
    T1_input = np.around(j, 1)

    #Setting empty lists to record results
    t1_star_list_51111 = []
    t1_corrected_list_51111 = []
    r2_list_51111 = []
    FE_51111 = []
    fit_type_51111 = []

    for i in hr_list: # This loop uses the 5(1)1(1)1 method
        heart_rate = i
        
        # Empty arrays to be filled/ variables to recalculate
        Acq_time = TR_bFFE*Total_shots + TR_bFFE/2
        Recov_time = heart_rate - Acq_time
        t_small = Recov_time/split
        delay1 = heart_rate-first_TI-Acq_time
        delay2 = heart_rate-second_TI-Acq_time
        delay3 = heart_rate-third_TI-Acq_time
        Mz_to_plot = np.array([M0, M0, -M0])
        Mxy_to_plot = np.array([0.0, 0.0, 0.0])
        T_for_mag = np.array([0.0, delay1, delay1])
        m = np.array([0.0, 0.0, 0.0])
        m[0] = 0 
        m[1] = 0  
        m[2] = -M0
        
        # Running the simulation to gain the signal
        Mxy_to_plot, Mz_to_plot, T_for_mag, T_for_bFFE, Mxy_only, Mz_only = signalsim51111(inv, m, t_small, split, T2, T1_input, M0, alpha, Mxy_to_plot, Mz_to_plot, T_for_mag, heart_rate, Acq_time, first_TI, second_TI, third_TI, delay1, delay2, delay3)

        length = len(Mz_to_plot)
        for i in range(0, length): 
            if Mz_to_plot[i] < 0:
                Mxy_to_plot[i] = -(Mxy_to_plot[i])

        length = len(Mz_only)
        for i in range(0, length): 
            if Mz_only[i] < 0:
                Mxy_only[i] = -(Mxy_only[i]) 
        
        t1_star, t1_corrected, r2, FE, fit_type, T_to_fit, Mxy_to_fit, plotlist, A, B = fitting51111(T_for_bFFE, Mxy_only, heart_rate, startup_shots, readout_shots, Total_shots)

        '''
        plt.figure()
        plt.plot(T_for_mag, Mxy_to_plot,'b-')
        plt.title('Mxy')
        plt.plot(T_for_mag, Mxy_to_plot,'b-')
        plt.plot(T_for_bFFE[1:], Mxy_only[1:],'r.')
        plt.plot(T_to_fit[plotlist], func(T_to_fit[plotlist], A, B, t1_star), 'r--')
        '''
        
        # Adding T1 values to a list for csv recording
        t1_star_list_51111.append(t1_star)
        t1_corrected_list_51111.append(t1_corrected*1000)
        r2_list_51111.append(r2)
        FE_51111.append(FE)
        fit_type_51111.append(fit_type)
        

    # Writing input T1, corrected T1s for each heart rate to dataframe
    T1_input = T1_input*1000
    t1_corrected_list_51111 = [T1_input] + t1_corrected_list_51111 
    df_51111.loc[len(df_51111)] = t1_corrected_list_51111


# Convert dataframes to csv file to record data   
df_51111.to_csv(r'C:\Users\ppxmr2\OneDrive - The University of Nottingham\Documents\Coding projects\MOLLI\Python\LUT_corrections\LUT_values\Kidney\30deg\18RR.csv', index=False)

