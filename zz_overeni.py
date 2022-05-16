import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm

file_woman = "meshes_weight_age_woman90.npy"
w = np.load(file_woman)
file_man= "meshes_weight_age_man120.npy"
m = np.load(file_man)

y1=w[:,0]
x1=[x[1] for x in w]

y2=m[:,0]
x2=[x[1] for x in m]

fs=18
fig_size =(9, 7)


Wmodel1 = np.poly1d(np.polyfit(x1, y1, 1))
Mmodel1 = np.poly1d(np.polyfit(x2, y2, 1))
polyline1 = np.linspace(20,84)
polyline2 = np.linspace(20,84)



#PLOT
fig = plt.figure("BMD_age_woman",figsize=fig_size, dpi=80)
fig = plt.scatter(x1,y1) 
plt.plot(polyline1, Wmodel1(polyline1), color='orange')
plt.ylim([112,133]) 
# Add Title
#plt.title("Závislost kostní minerální hustoty na věku u žen",fontsize=fs) 
# Add Axes Labels
plt.ylabel("minerální hustota kosti [g]",fontsize=20) 
plt.xlabel("věk [rok]",fontsize=20) 
plt.tick_params(axis='both', labelsize=14)
# Display
plt.show()

# PLot
fig = plt.figure("BMD_age_man",figsize=fig_size, dpi=80)
fig = plt.scatter(x2,y2) 
plt.plot(polyline2, Mmodel1(polyline2), color='orange')
plt.ylim([112,133]) 
# Add Title
#plt.title("Závislost kostní minerální hustoty na věku u mužů",fontsize=fs) 
# Add Axes Labels
plt.ylabel("minerální hustota kosti [g]",fontsize=20) 
plt.xlabel("věk [rok]",fontsize=20) 
plt.tick_params(axis='both', labelsize=14)
# Display
plt.show()




#HISTOGRAM
hm=y1
fig = plt.figure("histogram_woman",figsize=fig_size, dpi=80)
fig = plt.hist(hm,edgecolor="black")
plt.tick_params(axis='both', labelsize=16)
#plt.title("Histogram kostní minerální hustoty u žen",fontsize=fs) 
plt.show()

#plt.hist(y)
hm=y2
fig = plt.figure("histogram_man",figsize=fig_size, dpi=80)
fig = plt.hist(hm,edgecolor="black")
plt.tick_params(axis='both', labelsize=16)
#plt.title("Histogram kostní minerální hustoty u mužů",fontsize=fs) 
plt.show()

stredni_hodnota = y1.mean()
print(stredni_hodnota)
smerodatna_odchylka = y1.std()
print(smerodatna_odchylka)
print("----------------")

stredni_hodnota = y2.mean()
print(stredni_hodnota)
smerodatna_odchylka = y2.std()
print(smerodatna_odchylka)

