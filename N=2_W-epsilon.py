import numpy as np
import matplotlib.pyplot as plt
import sys

p1 = 0.79
p2 = 0.80
p3 = 0.81
Ic = 1/np.sqrt(2)
theta = np.pi/8
e = np.linspace(0,0.012,100)

W1 = p1/np.sqrt(2) *(np.sqrt(1+np.sin(2*theta)**2) *(1-2*e) + (1-np.sin(2*theta)**2) / np.sqrt(1+np.sin(2*theta)**2) * 2*np.sqrt(e*(1-e)))
W1_corrected = p1/np.sqrt(2) * np.sqrt(1+np.sin(2*theta)**2)*(1-2*e)
W2 = p2/np.sqrt(2) *(np.sqrt(1+np.sin(2*theta)**2) *(1-2*e) + (1-np.sin(2*theta)**2) / np.sqrt(1+np.sin(2*theta)**2) * 2*np.sqrt(e*(1-e)))
W2_corrected = p2/np.sqrt(2) * np.sqrt(1+np.sin(2*theta)**2)*(1-2*e)
W3 = p3/np.sqrt(2) *(np.sqrt(1+np.sin(2*theta)**2) *(1-2*e) + (1-np.sin(2*theta)**2) / np.sqrt(1+np.sin(2*theta)**2) * 2*np.sqrt(e*(1-e)))
W3_corrected = p3/np.sqrt(2) * np.sqrt(1+np.sin(2*theta)**2)*(1-2*e)
W_b = 1/np.sqrt(1+np.sin(2*theta)**2)/np.sqrt(2) *(np.sqrt(1+np.sin(2*theta)**2) *(1-2*e) + (1-np.sin(2*theta)**2) / np.sqrt(1+np.sin(2*theta)**2) * 2*np.sqrt(e*(1-e)))
plt.figure()
# plt.plot(e, W1,label = 'p = 0.79',color = 'red')
# plt.plot(e, W1_corrected,label = 'p = 0.79 corrected',linestyle='--',color = 'red')
plt.plot(e, W2,label = 'p = 0.80',color = 'orange')
plt.plot(e, W2_corrected,label = 'p = 0.80 corrected',linestyle='--',color = 'orange')
plt.plot(e, W3,label = 'p = 0.81',color = 'blue')
plt.plot(e, W3_corrected,label = 'p = 0.81 corrected',linestyle='--',color = 'blue')
plt.plot(e,W_b, label = 'W_bound',color = 'black',linestyle='-')

plt.plot([0,0.012],[np.sqrt(0.5),np.sqrt(0.5)],label ='detection bound',linestyle = '-.',color = 'grey')

plt.legend()
plt.xlabel('$\epsilon$')
plt.ylabel('W')
plt.show()