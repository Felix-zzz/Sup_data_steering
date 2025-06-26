import numpy as np
import matplotlib.pyplot as plt
import sys

thetalist = np.linspace(0,np.pi/4,100)

e_1 = 0.005
e_2 = 0.01
e_3 = 0.015

# For noisy, fixed and noiseless boundary p(\theta)
p2_noisy = 1/(np.sqrt(1+np.sin(2*thetalist)**2) *(1-2*e_1) + (1-np.sin(2*thetalist)**2) / np.sqrt(1+np.sin(2*thetalist)**2) * 2*np.sqrt(e_1*(1-e_1)))
p2_fixed = 1/(np.sqrt(1+np.sin(2*thetalist)**2) *(1-2*e_1))

p3_noisy = 1/(np.sqrt(1+2*np.sin(2*thetalist)**2) *(1-2*e_1) + np.sqrt(2)*(1-np.sin(2*thetalist)**2) / np.sqrt(1+2*np.sin(2*thetalist)**2) * 2*np.sqrt(e_1*(1-e_1)))
p3_fixed = 1/(np.sqrt(1+2*np.sin(2*thetalist)**2) *(1-2*e_1))

p2 = 1/(np.sqrt(1+np.sin(2*thetalist)**2))
p3 = 1/(np.sqrt(1+2*np.sin(2*thetalist)**2))

p_inf = 1/(1+0.5*np.sin(2*thetalist)**2/np.cos(2*thetalist)*np.log((1+np.cos(2*thetalist))/(1-np.cos(2*thetalist))))
p_inf[0]= 1
p_inf[-1]= 0.5

plt.figure(figsize=(6, 4))
plt.plot(thetalist, p2, label = r'$\mathcal{W}_2 = \mathcal{C}_2$',color = 'r')
plt.plot(thetalist, p2_noisy, label = r'$\tilde{\mathcal{W}}_2 = \mathcal{C}_2$',color = 'r',linestyle = ':')
plt.plot(thetalist, p2_fixed, label = r'$\overline{\mathcal{W}}_2 = \mathcal{C}_2$',color = 'r',linestyle = '--')
plt.plot(thetalist, p3, label = r'$\mathcal{W}_3 = \mathcal{C}_3$',color = 'b')
plt.plot(thetalist, p3_noisy, label = r'$\tilde{\mathcal{W}}_3 = \mathcal{C}_3$',color = 'b',linestyle = ':')
plt.plot(thetalist, p3_fixed, label = r'$\overline{\mathcal{W}}_3 = \mathcal{C}_3$',color = 'b',linestyle = '--')
plt.plot(thetalist, p_inf, label = r'$\mathcal{W}_\infty = \mathcal{C}_\infty$',color = 'k', alpha=1)

plt.fill_between(thetalist[np.where(p2>p2_noisy)[0].tolist()], p2[np.where(p2>p2_noisy)[0].tolist()], p2_noisy[np.where(p2>p2_noisy)[0].tolist()], color = 'r', alpha = 0.2)
plt.fill_between(thetalist[np.where(p3>p3_noisy)[0].tolist()], p3[np.where(p3>p3_noisy)[0].tolist()], p3_noisy[np.where(p3>p3_noisy)[0].tolist()], color = 'b', alpha = 0.2)
plt.fill_between(thetalist, [1]*len(thetalist), p_inf, color = 'k', alpha = 0.1)

plt.xlim(0,np.pi/5.5)
plt.ylim(0.5, 1)
plt.yticks([0.5, 0.75, 1],fontsize = 12)
plt.xticks([0,np.pi/24,np.pi/12,np.pi/8, np.pi/6], [0,r'${\pi}/{24}$', r'${\pi}/{12}$', r'${\pi}/{8}$', r'${\pi}/{6}$'], fontsize=12)
plt.legend()
plt.xlabel(r'$\theta$',fontsize = 14)
plt.ylabel('p', fontsize = 14)
plt.savefig('./Fig_p_theta.pdf', dpi=300, bbox_inches='tight')

plt.show()