import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bootstrap
from qutip import Bloch
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# RM_means=np.array([0.69282032, 0.69114347, 0.6894391,  0.68785612, 0.686395,   0.68457424, 0.6828699,  0.68099855, 0.67968087, 0.67753035, 0.67620559])
# RM_stds=np.array([2.44730529e-16, 7.99576431e-03, 1.13190636e-02, 1.39021798e-02, 1.57693490e-02, 1.78885860e-02, 1.97256493e-02, 2.11319061e-02,  2.26586822e-02, 2.39150227e-02, 2.52632908e-02])
# RM_noisy_means=np.array([0.6910675,  0.68960882, 0.68761859, 0.68603325, 0.68478593, 0.68289942, 0.6812139,  0.67947604, 0.67806593, 0.67599403, 0.67445696])
# RM_noisy_stds=np.array([0.00854967, 0.01271899, 0.01583659, 0.01806364, 0.01962089, 0.02155915, 0.02332099, 0.02448514, 0.02596668, 0.02712385, 0.02872178])

# prefix = 'N=2/'
# RM_means = np.load(prefix + 'RM_means.npy')
# RM_stds = np.load(prefix + 'RM_stds.npy')
# RM_noisy_means = np.load(prefix + 'RM_noisy_means.npy')
# RM_noisy_stds = np.load(prefix + 'RM_noisy_stds.npy')
# Wc_list = np.load(prefix + 'Wc_list.npy')
# data_ideal = np.load(prefix + 'data_ideal.npy', allow_pickle=True)
# data_noisy = np.load(prefix + 'data_noisy.npy', allow_pickle=True)
# elist = np.linspace(0, 0.012, 11)
# theta = np.pi/8
# p = 0.8
# epsilon = elist
# e_fine_list = np.linspace(0, 0.012, 500)
# # Wc_list = p/np.sqrt(2) * np.sqrt(1+np.sin(2*theta)**2) * (1 - 2*epsilon) + p/np.sqrt(2) * (1-np.sin(2*theta)**2) / np.sqrt(1+np.sin(2*theta)**2) * 2*np.sqrt(epsilon*(1-epsilon))
# W0 = p/np.sqrt(2) * np.sqrt(1+np.sin(2*theta)**2)
# W_c_fine = p/np.sqrt(2) * np.sqrt(1+np.sin(2*theta)**2) * (1 - 2*e_fine_list) + p/np.sqrt(2) * (1-np.sin(2*theta)**2) / np.sqrt(1+np.sin(2*theta)**2) * 2*np.sqrt(e_fine_list*(1-e_fine_list))
# # plt.errorbar(elist, RM_means, RM_stds, alpha = 0.5, label = 'Exact RM', color = 'orange')
# # plt.errorbar(elist, RM_noisy_means, RM_noisy_stds, alpha = 0.5, label = 'Noisy RM', color = 'purple')

# # plt.plot([0,0.012],[0.707,0.707], linestyle='dashed',label = r'$\mathcal{C}_2$', color = 'grey')
# plt.plot([0,0.012], [W0, W0], linestyle = 'dashed', label = r'$W_2$', color = 'grey')
# # plt.plot(elist, Wc_list, linestyle = 'solid', label = r'$\tilde{W}_2^{\epsilon}$', color = 'red')
# plt.plot(e_fine_list, W_c_fine, linestyle = 'dashed', color = 'orange', label = r'$\tilde{W}_2^{\epsilon}$')
# intval_ideal = []
# intval_noisy = []
# plt.plot(elist, RM_means, linestyle = 'solid', label = 'Exact RM', color = 'blue')
# plt.plot(elist, RM_noisy_means, linestyle = 'dotted', label = 'Noisy RM', color = 'blue')
# # Compute 99% confidence intervals using bootstrap
# for i in range(len(RM_means)):
#     res_ideal = bootstrap((data_ideal[i],), np.mean, confidence_level=0.99, n_resamples=1000, method='basic')
#     res_noisy = bootstrap((data_noisy[i],), np.mean, confidence_level=0.99, n_resamples=1000, method='basic')
#     print(f"e={elist[i]:.4f}: Ideal RM mean CI = {res_ideal.confidence_interval}, Noisy RM mean CI = {res_noisy.confidence_interval}")
#     intval_ideal.append(res_ideal.confidence_interval)
#     intval_noisy.append(res_noisy.confidence_interval)
# intval_ideal = np.array(intval_ideal)
# intval_noisy = np.array(intval_noisy)
# plt.fill_between(elist, intval_ideal[:,0], intval_ideal[:,1], color='blue', alpha=0.25)
# plt.fill_between(elist, intval_noisy[:,0], intval_noisy[:,1], color='blue', alpha=0.25)
# # plt.errorbar(elist, RM_means, yerr=[RM_means - intval_ideal[:,0], intval_ideal[:,1] - RM_means], fmt='o', color='orange', alpha=0.5)
# # plt.errorbar(elist, RM_noisy_means, yerr=[RM_noisy_means - intval_noisy[:,0], intval_noisy[:,1] - RM_noisy_means], fmt='o', color='purple', alpha=0.5)


# plt.xlabel(r'$\epsilon$ $/$ $10^{-3}$', fontsize=14)
# plt.ylabel(r'$\mathcal{W}$ Value', fontsize=14)
# plt.ylim([0.66, 0.74])
# plt.yticks(np.arange(0.66, 0.75, 0.02), fontsize=12)
# plt.xticks(np.arange(0, 0.013, 0.002),['0','2', '4','6','8','10','12'], fontsize=12)
# plt.legend()
# plt.savefig('WvsE_N2.pdf')
# # plt.show()

##################################
# # N = 2 Uniform
##################################
# prefix = 'N=2_uniform/'
# RM_means = np.load(prefix + 'RM_means.npy')
# RM_stds = np.load(prefix + 'RM_stds.npy')
# RM_noisy_means = np.load(prefix + 'RM_noisy_means.npy')
# RM_noisy_stds = np.load(prefix + 'RM_noisy_stds.npy')
# Wc_list = np.load(prefix + 'Wc_list.npy')
# data_ideal = np.load(prefix + 'data_ideal.npy', allow_pickle=True)
# data_noisy = np.load(prefix + 'data_noisy.npy', allow_pickle=True)
# elist = np.linspace(0, 0.012, 11)
# theta = np.pi/8
# p = 0.8
# epsilon = elist
# e_fine_list = np.linspace(0, 0.012, 500)
# # Wc_list = p/np.sqrt(2) * np.sqrt(1+np.sin(2*theta)**2) * (1 - 2*epsilon) + p/np.sqrt(2) * (1-np.sin(2*theta)**2) / np.sqrt(1+np.sin(2*theta)**2) * 2*np.sqrt(epsilon*(1-epsilon))
# W0 = p/np.sqrt(2) * np.sqrt(1+np.sin(2*theta)**2)
# W_c_fine = p/np.sqrt(2) * np.sqrt(1+np.sin(2*theta)**2) * (1 - 2*e_fine_list) + p/np.sqrt(2) * (1-np.sin(2*theta)**2) / np.sqrt(1+np.sin(2*theta)**2) * 2*np.sqrt(e_fine_list*(1-e_fine_list))
# # plt.errorbar(elist, RM_means, RM_stds, alpha = 0.5, label = 'Exact RM', color = 'orange')
# # plt.errorbar(elist, RM_noisy_means, RM_noisy_stds, alpha = 0.5, label = 'Noisy RM', color = 'purple')

# # plt.plot([0,0.012],[0.707,0.707], linestyle='dashed',label = r'$\mathcal{C}_2$', color = 'grey')
# plt.plot([0,0.012], [W0, W0], linestyle = 'dashed', label = r'$W_2$', color = 'grey')
# # plt.plot(elist, Wc_list, linestyle = 'solid', label = r'$\tilde{W}_2^{\epsilon}$', color = 'red')
# plt.plot(e_fine_list, W_c_fine, linestyle = 'dashed', color = 'orange', label = r'$\tilde{W}_2^{\epsilon}$')
# intval_ideal = []
# intval_noisy = []
# plt.plot(elist, RM_means, linestyle = 'solid', label = 'Exact RM', color = 'blue')
# plt.plot(elist, RM_noisy_means, linestyle = 'dotted', label = 'Noisy RM', color = 'blue')
# # Compute 99% confidence intervals using bootstrap
# for i in range(len(RM_means)):
#     res_ideal = bootstrap((data_ideal[i],), np.mean, confidence_level=0.99, n_resamples=1000, method='basic')
#     res_noisy = bootstrap((data_noisy[i],), np.mean, confidence_level=0.99, n_resamples=1000, method='basic')
#     print(f"e={elist[i]:.4f}: Ideal RM mean CI = {res_ideal.confidence_interval}, Noisy RM mean CI = {res_noisy.confidence_interval}")
#     intval_ideal.append(res_ideal.confidence_interval)
#     intval_noisy.append(res_noisy.confidence_interval)
# intval_ideal = np.array(intval_ideal)
# intval_noisy = np.array(intval_noisy)
# plt.fill_between(elist, intval_ideal[:,0], intval_ideal[:,1], color='blue', alpha=0.25)
# plt.fill_between(elist, intval_noisy[:,0], intval_noisy[:,1], color='blue', alpha=0.25)
# # plt.errorbar(elist, RM_means, yerr=[RM_means - intval_ideal[:,0], intval_ideal[:,1] - RM_means], fmt='o', color='orange', alpha=0.5)
# # plt.errorbar(elist, RM_noisy_means, yerr=[RM_noisy_means - intval_noisy[:,0], intval_noisy[:,1] - RM_noisy_means], fmt='o', color='purple', alpha=0.5)


# plt.xlabel(r'$\epsilon$ $/$ $10^{-3}$', fontsize=14)
# plt.ylabel(r'$\mathcal{W}$ Value', fontsize=14)
# plt.ylim([0.66, 0.74])
# plt.yticks(np.arange(0.66, 0.75, 0.02), fontsize=12)
# plt.xticks(np.arange(0, 0.013, 0.002),['0','2', '4','6','8','10','12'], fontsize=12)
# plt.legend()
# plt.savefig('WvsE_N2_uniform_0.0024.pdf')
# plt.show()


# ################################
# # N = 3
# ################################
# prefix = 'N=3/'
# RM_means = np.load(prefix + 'RM_means_3.npy')
# RM_stds = np.load(prefix + 'RM_stds_3.npy')
# RM_noisy_means = np.load(prefix + 'RM_noisy_means_3.npy')
# RM_noisy_stds = np.load(prefix + 'RM_noisy_stds_3.npy')
# Wc_list = np.load(prefix + 'Wc_list_3.npy')
# data_ideal = np.load(prefix + 'data_ideal_3.npy', allow_pickle=True)
# data_noisy = np.load(prefix + 'data_noisy_3.npy', allow_pickle=True)
# elist = np.linspace(0, 0.012, 11)
# theta = np.pi/8
# p = 0.8
# epsilon = elist
# e_fine_list = np.linspace(0, 0.012, 500)
# # Wc_list = p/np.sqrt(2) * np.sqrt(1+np.sin(2*theta)**2) * (1 - 2*epsilon) + p/np.sqrt(2) * (1-np.sin(2*theta)**2) / np.sqrt(1+np.sin(2*theta)**2) * 2*np.sqrt(epsilon*(1-epsilon))
# W0 = p/np.sqrt(3) * np.sqrt(1+2*np.sin(2*theta)**2)
# W_c_fine = p/np.sqrt(3) * np.sqrt(1+2* np.sin(2*theta)**2) * (1 - 2*e_fine_list) + p/np.sqrt(3) * (1-np.sin(2*theta)**2) / np.sqrt(1+2*np.sin(2*theta)**2) * 2*np.sqrt(2)*np.sqrt(e_fine_list*(1-e_fine_list))

# plt.figure()
# # plt.errorbar(elist, RM_means, RM_stds, alpha = 0.5, label = 'Exact RM', color = 'orange')
# # plt.errorbar(elist, RM_noisy_means, RM_noisy_stds, alpha = 0.5, label = 'Noisy RM', color = 'purple')

# # plt.plot([0,0.012],[0.707,0.707], linestyle='dashed',label = r'$\mathcal{C}_2$', color = 'grey')
# plt.plot([0,0.012], [W0, W0], linestyle = 'dashed', label = r'$W_3$', color = 'grey')
# # plt.plot(elist, Wc_list, linestyle = 'solid', label = r'$\tilde{W}_2^{\epsilon}$', color = 'red')
# plt.plot(e_fine_list, W_c_fine, linestyle = 'dashed', color = 'orange', label = r'$\tilde{W}_3^{\epsilon}$')
# intval_ideal = []
# intval_noisy = []
# plt.plot(elist, RM_means, linestyle = 'solid', label = 'Exact RM', color = 'blue')
# plt.plot(elist, RM_noisy_means, linestyle = 'dotted', label = 'Noisy RM', color = 'blue')
# # Compute 99% confidence intervals using bootstrap
# for i in range(len(RM_means)):
#     res_ideal = bootstrap((data_ideal[i],), np.mean, confidence_level=0.99, n_resamples=1000, method='basic')
#     res_noisy = bootstrap((data_noisy[i],), np.mean, confidence_level=0.99, n_resamples=1000, method='basic')
#     print(f"e={elist[i]:.4f}: Ideal RM mean CI = {res_ideal.confidence_interval}, Noisy RM mean CI = {res_noisy.confidence_interval}")
#     intval_ideal.append(res_ideal.confidence_interval)
#     intval_noisy.append(res_noisy.confidence_interval)
# intval_ideal = np.array(intval_ideal)
# intval_noisy = np.array(intval_noisy)
# plt.fill_between(elist, intval_ideal[:,0], intval_ideal[:,1], color='blue', alpha=0.25)
# plt.fill_between(elist, intval_noisy[:,0], intval_noisy[:,1], color='blue', alpha=0.25)
# # plt.errorbar(elist, RM_means, yerr=[RM_means - intval_ideal[:,0], intval_ideal[:,1] - RM_means], fmt='o', color='orange', alpha=0.5)
# # plt.errorbar(elist, RM_noisy_means, yerr=[RM_noisy_means - intval_noisy[:,0], intval_noisy[:,1] - RM_noisy_means], fmt='o', color='purple', alpha=0.5)


# plt.xlabel(r'$\epsilon$ $/$ $10^{-3}$', fontsize=14)
# plt.ylabel(r'$\mathcal{W}$ Value', fontsize=14)
# plt.ylim([0.62, 0.70])
# # plt.xlim([0, 0.012])
# plt.yticks(np.arange(0.62, 0.71, 0.02), fontsize=12)
# plt.xticks(np.arange(0, 0.013, 0.002),['0','2', '4','6','8','10','12'], fontsize=12)
# plt.legend()
# plt.savefig('WvsE_N3.pdf')
# # plt.show()

# ################################
# # Plot W vs N for fixed epsilon
# ################################

# prefix = 'W-M/'
# epsilon = 0.0024
# theta = np.pi/8
# sigma_infid = 0.0012     # 角度标准差 infid
# p = 0.8
# # N_list = np.load(prefix + 'N_list.npy')
# N_list = np.arange(10,1010,10)
# N_list = np.insert(N_list, 0, np.arange(1,10))
# N_list = np.append(N_list, np.arange(1100, 10100,100))

# RM_means = np.load(prefix + 'RM_means.npy')
# RM_stds = np.load(prefix + 'RM_stds.npy')
# RM_noisy_means = np.load(prefix + 'RM_noisy_means.npy')
# RM_noisy_stds = np.load(prefix + 'RM_noisy_stds.npy')
# ideal = np.array([np.cos(theta), 0, 0, np.sin(theta)])
# rho = p * np.outer(ideal,ideal) + (1-p) * np.eye(4)/4
# C2 = 1/np.sqrt(2)
# W0 = p/np.sqrt(2) * np.sqrt(1+np.sin(2*theta)**2)
# Wn = p/np.sqrt(2) * np.sqrt(1+np.sin(2*theta)**2) * (1 - 2*epsilon) + p/np.sqrt(2) * (1-np.sin(2*theta)**2) / np.sqrt(1+np.sin(2*theta)**2) * 2*np.sqrt(epsilon*(1-epsilon))
# WRm = p/np.sqrt(2) * np.sqrt(1+np.sin(2*theta)**2) * (1 - 2*epsilon)
# fig = plt.figure(figsize= (14,5))

# plt.axhline(W0, linestyle = 'dashed', label = r'$W_2$', color = 'grey')
# plt.axhline(Wn, linestyle = 'dashed', label = r'$\tilde{W}_2^{\epsilon}$', color = 'orange')
# plt.axhline(WRm, linestyle = 'dashed', label = r'$W_{RM}$', color = 'blue')
# plt.plot(N_list, RM_means, linestyle = 'solid', label = f'Exact RM',color = 'blue')

# for i in range(len(RM_noisy_means[0])):
#     plt.plot(N_list, RM_noisy_means[:,i], linestyle = 'solid', label = r'$\sigma = $'+f'{(i+1)*sigma_infid:.4f}',color = 'blue', alpha = 1-(i+1)*0.25)
# # plt.axhline(C2, linestyle='dashed',label = r'$\mathcal{C}_2$', color = 'grey')


# plt.xscale('log')
# plt.xlabel('Number of Random Unitaries', fontsize=14)
# plt.ylabel(r'$\mathcal{W}$ Value', fontsize=14)    
# plt.ylim([0.64,0.72])
# plt.xlim([1,10000])
# plt.xticks([1,10,100,1000,10000], ['1','10','100','1k','10k'], fontsize=12)
# plt.yticks(np.arange(0.64, 0.73, 0.02), fontsize=12)
# plt.legend()
# plt.savefig('WvsN.pdf')
# # plt.show()
# plt.xlim([1000,10000])
# plt.ylim([0.676,0.696])
# plt.yticks(np.arange(0.676, 0.696, 0.006), fontsize=12)
# plt.legend([], frameon=False) 
# fig.set_size_inches(3, 2)
# plt.subplots_adjust(left=0.2) 
# plt.subplots_adjust(bottom=0.13)
# plt.savefig('WvsN_zoom.pdf')
# plt.show()

# ############################
# # New Plot W vs N for fixed epsilon
# ############################

# prefix = 'New_W_M/'
# epsilon = 0.0024
# theta = np.pi/8
# sigma_infid = 0.0012     # 角度标准差 infid
# p = 0.8
# # N_list = np.load(prefix + 'N_list.npy')
# N_list = np.arange(10,1010,10)
# N_list = np.insert(N_list, 0, np.arange(1,10))
# N_list = np.append(N_list, np.arange(1100, 10100,100))
# X = np.array([[0,1],[1,0]])
# Y = np.array([[0,-1j],[1j,0]])
# Z = np.array([[1,0],[0,-1]])
# A1 = (1 / np.sqrt(1 + np.sin(2 * theta) ** 2)) * (Z - np.sin(2 * theta) * X)
# A2 = (1 / np.sqrt(1 + np.sin(2 * theta) ** 2)) * (Z + np.sin(2 * theta) * X)
# B10 = np.cos(np.pi/4) * Z - np.sin(np.pi/4) * X
# B20 = np.cos(np.pi/4) * Z + np.sin(np.pi/4) * X
# for trial in range(5):
#     UMY1 = np.load(prefix + str(trial) + '_UM1.npy')
#     UMY2 = np.load(prefix + str(trial) + '_UM2.npy')
#     effective_B1 = np.load(prefix + str(trial) + '_effective_B1.npy')
#     effective_B2 = np.load(prefix + str(trial) + '_effective_B2.npy')
#     n1 = [np.trace(effective_B1 @ X) / 2, np.trace(effective_B1 @ Y) / 2, np.trace(effective_B1 @ Z) / 2]
#     n2 = [np.trace(effective_B2 @ X) / 2, np.trace(effective_B2 @ Y) / 2, np.trace(effective_B2 @ Z) / 2]
#     RM_means = np.load(prefix + str(trial) + '_RM_means.npy')
#     RM_stds = np.load(prefix + str(trial) + '_RM_stds.npy')
#     RM_noisy_means = np.load(prefix + str(trial) + '_RM_noisy_means.npy')
#     RM_noisy_stds = np.load(prefix + str(trial) + '_RM_noisy_stds.npy')
#     ideal = np.array([np.cos(theta), 0, 0, np.sin(theta)])
#     rho = p * np.outer(ideal,ideal) + (1-p) * np.eye(4)/4
#     C2 = 1/np.sqrt(2)
#     W0 = p/np.sqrt(2) * np.sqrt(1+np.sin(2*theta)**2)
#     Wn = 0.5 * np.trace(rho @ (np.kron(A1, effective_B1) + np.kron(A2, effective_B2)))
#     # print(f'Trial {trial}: \nB1 effective=\n{effective_B1}\nB2 effective=\n{effective_B2}\nWn={Wn}')
#     print(f'Trial {trial}: \nn1={n1}, n2={n2}\nWn={Wn}')
#     cos_alpha_1 = np.trace(effective_B1 @ B10) / 2
#     cos_alpha_2 = np.trace(effective_B2 @ B20) / 2
#     print(f'cos_alpha_1={cos_alpha_1}, cos_alpha_2={cos_alpha_2}, e_1 = {(1-cos_alpha_1)/2}, e_2 = {(1-cos_alpha_2)/2}')
#     WRm = 0.5 * np.trace(rho @ (np.kron(A1, cos_alpha_1*B10) + np.kron(A2, cos_alpha_2*B20)))
#     # Wn = p/np.sqrt(2) * np.sqrt(1+np.sin(2*theta)**2) * (1 - 2*epsilon) + p/np.sqrt(2) * (1-np.sin(2*theta)**2) / np.sqrt(1+np.sin(2*theta)**2) * 2*np.sqrt(epsilon*(1-epsilon))
#     # WRm = p/np.sqrt(2) * np.sqrt(1+np.sin(2*theta)**2) * (1 - 2*epsilon)
#     fig = plt.figure(figsize= (14,5))

#     plt.axhline(W0, linestyle = 'dashed', label = r'$W_2$', color = 'grey')
#     plt.axhline(Wn, linestyle = 'dashed', label = r'$\tilde{W}_2^{\epsilon}$', color = 'orange')
#     plt.axhline(WRm, linestyle = 'dashed', label = r'$W_{RM}$', color = 'blue')
#     plt.plot(N_list, RM_means, linestyle = 'solid', label = f'Exact RM',color = 'blue')

#     for i in range(len(RM_noisy_means[0])):
#         plt.plot(N_list, RM_noisy_means[:,i], linestyle = 'solid', label = r'$\sigma = $'+f'{(i+1)*sigma_infid:.4f}',color = 'blue', alpha = 1-(i+1)*0.25)
# # plt.axhline(C2, linestyle='dashed',label = r'$\mathcal{C}_2$', color = 'grey')


#     plt.xscale('log')
#     plt.xlabel('Number of Random Unitaries', fontsize=14)
#     plt.ylabel(r'$\mathcal{W}$ Value', fontsize=14)    
#     plt.ylim([0.64,0.72])
#     plt.xlim([1,10000])
#     plt.xticks([1,10,100,1000,10000], ['1','10','100','1k','10k'], fontsize=12)
#     plt.yticks(np.arange(0.64, 0.73, 0.02), fontsize=12)
#     plt.title(f'Trial {trial}, n1={n1}, n2={n2}, Wn={Wn}', fontsize=16)
#     plt.legend()
#     plt.savefig(prefix+str(trial) + '_WvsN.pdf')
#     # plt.show()
#     # plt.xlim([1000,10000])
#     # plt.ylim([0.676,0.696])
#     # plt.yticks(np.arange(0.676, 0.696, 0.006), fontsize=12)
#     # plt.legend([], frameon=False) 
#     # fig.set_size_inches(3, 2)
#     # plt.subplots_adjust(left=0.2) 
#     # plt.subplots_adjust(bottom=0.13)
#     # plt.savefig('WvsN_zoom.pdf')
#     # plt.show()

############################
# New Plot W vs delta for different B1, B2
############################

prefix = 'New_W_M/'
epsilon = 0.0024
theta = np.pi/8
sigma_infid = 0.0012     # 角度标准差 infid
sigma_infid_list = np.linspace(0.01, 3, 300)*sigma_infid
p = 0.8
# N_list = np.load(prefix + 'N_list.npy')
N_list = np.arange(10,1010,10)
N_list = np.insert(N_list, 0, np.arange(1,10))
N_list = np.append(N_list, np.arange(1100, 10100,100))
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])
A1 = (1 / np.sqrt(1 + np.sin(2 * theta) ** 2)) * (Z - np.sin(2 * theta) * X)
A2 = (1 / np.sqrt(1 + np.sin(2 * theta) ** 2)) * (Z + np.sin(2 * theta) * X)
B10 = np.cos(np.pi/4) * Z - np.sin(np.pi/4) * X
B20 = np.cos(np.pi/4) * Z + np.sin(np.pi/4) * X
C2 = 1/np.sqrt(2)
W0 = p/np.sqrt(2) * np.sqrt(1+np.sin(2*theta)**2)
# fig = plt.figure(figsize= (14,5))
# plt.plot(sigma_infid_list, np.full_like(sigma_infid_list, W0), linestyle = 'dashed', label = r'$W_2$', color = 'grey')
    

# We only take N=500 for different trials
for trial in range(5,30):
    UMY1 = np.load(prefix + str(trial) + '_UM1.npy')
    UMY2 = np.load(prefix + str(trial) + '_UM2.npy')
    effective_B1 = np.load(prefix + str(trial) + '_effective_B1.npy')
    effective_B2 = np.load(prefix + str(trial) + '_effective_B2.npy')
    n1 = [np.trace(effective_B1 @ X) / 2, np.trace(effective_B1 @ Y) / 2, np.trace(effective_B1 @ Z) / 2]
    n2 = [np.trace(effective_B2 @ X) / 2, np.trace(effective_B2 @ Y) / 2, np.trace(effective_B2 @ Z) / 2]
    RM_means = np.load(prefix + str(trial) + '_RM_means.npy')
    RM_stds = np.load(prefix + str(trial) + '_RM_stds.npy')
    RM_noisy_means = np.load(prefix + str(trial) + '_RM_noisy_means.npy')
    RM_noisy_stds = np.load(prefix + str(trial) + '_RM_noisy_stds.npy')
    # if trial<5:
    #     RM_noisy_means = RM_noisy_means[58,:]
    #     print(N_list[58])
    #     RM_noisy_stds = RM_noisy_stds[58,:]
    ideal = np.array([np.cos(theta), 0, 0, np.sin(theta)])
    rho = p * np.outer(ideal,ideal) + (1-p) * np.eye(4)/4
    
    Wn = 0.5 * np.trace(rho @ (np.kron(A1, effective_B1) + np.kron(A2, effective_B2)))
    # print(f'Trial {trial}: \nB1 effective=\n{effective_B1}\nB2 effective=\n{effective_B2}\nWn={Wn}')
    print(f'Trial {trial}: \nn1={n1}, n2={n2}\nWn={Wn}')
    cos_alpha_1 = np.trace(effective_B1 @ B10) / 2
    cos_alpha_2 = np.trace(effective_B2 @ B20) / 2
    print(f'cos_alpha_1={cos_alpha_1}, cos_alpha_2={cos_alpha_2}, e_1 = {(1-cos_alpha_1)/2}, e_2 = {(1-cos_alpha_2)/2}')
    WRm = 0.5 * np.trace(rho @ (np.kron(A1, cos_alpha_1*B10) + np.kron(A2, cos_alpha_2*B20)))
    # Wn = p/np.sqrt(2) * np.sqrt(1+np.sin(2*theta)**2) * (1 - 2*epsilon) + p/np.sqrt(2) * (1-np.sin(2*theta)**2) / np.sqrt(1+np.sin(2*theta)**2) * 2*np.sqrt(epsilon*(1-epsilon))
    # WRm = p/np.sqrt(2) * np.sqrt(1+np.sin(2*theta)**2) * (1 - 2*epsilon)

    fig, ax = plt.subplots(figsize= (8,5))
    # ax = fig.add_subplot(111)
    # ax = fig.add_subplot(121)  # 主图
    # axins = fig.add_axes([0.6, 0.15, 0.25, 0.7], projection='3d')
    ax.plot(sigma_infid_list, np.full_like(sigma_infid_list, W0), linestyle = 'dashed', label = r'$W_2$', color = 'grey')
    
    ax.plot(sigma_infid_list, np.full_like(sigma_infid_list, Wn), linestyle = 'dashed', label = r'$\tilde{W}_2^{\epsilon}$', color = 'orange')#, alpha = 1-(trial+1)*0.175 + 5*0.175)
    # plt.plot(sigma_infid_list, np.full_like(sigma_infid_list, WRm), linestyle = 'dashed', label = r'$W_{RM}$', color = 'blue')#, alpha = 1-(trial+1)*0.175 + 5*0.175)
    ax.plot(sigma_infid_list, np.full_like(sigma_infid_list, RM_means), linestyle = 'dashed', label = f'Exact RM',color = 'blue')#, alpha = 1-(trial+1)*0.175 + 5*0.175)
    ax.plot(sigma_infid_list, np.full_like(sigma_infid_list, RM_noisy_means), linestyle = 'solid', label = f'Noisy RM',color = 'blue')#, alpha = 1-(trial+1)*0.15 + 5*0.15)


#     plt.axhline(Wn, linestyle = 'dashed', label = r'$\tilde{W}_2^{\epsilon}$', color = 'orange')
#     plt.axhline(WRm, linestyle = 'dashed', label = r'$W_{RM}$', color = 'blue')
#     plt.plot(N_list, RM_means, linestyle = 'solid', label = f'Exact RM',color = 'blue')

#     for i in range(len(RM_noisy_means[0])):
#         plt.plot(N_list, RM_noisy_means[:,i], linestyle = 'solid', label = r'$\sigma = $'+f'{(i+1)*sigma_infid:.4f}',color = 'blue', alpha = 1-(i+1)*0.25)
# # plt.axhline(C2, linestyle='dashed',label = r'$\mathcal{C}_2$', color = 'grey')


    # plt.xscale('log')
    # plt.xlabel('Number of Random Unitaries', fontsize=14)
    ax.set_xlabel('Max infidelity of random unitaries $\sigma$', fontsize=20)
    ax.set_ylabel(r'$\mathcal{W}$ Value', fontsize=20)    
    ax.set_ylim([0.645,0.72])
    ax.set_xlim([0, 0.0036])
    ax.set_xticks([0,0.0012,0.0024,0.0036])
    ax.set_xticklabels(['0',r'$1.2\times 10^{-3}$', r'$2.4\times 10^{-3}$', r'$3.6\times 10^{-3}$'], fontsize=22)
    ax.set_yticks([0.645, 0.67, 0.695, 0.72])
    ax.set_yticklabels([0.645, 0.67, 0.695, 0.72], fontsize=22)
    # plt.title(f'Trial {trial}, n1={np.real(n1)}, n2={np.real(n2)}, e_1 = {(1-np.real(cos_alpha_1))/2}, e_2 = {(1-np.real(cos_alpha_2))/2}', fontsize=13)
    ax.legend(fontsize=14, loc='upper right')

    inset_pos = [0.65, 0.29, 0.24, 0.28]  # [left, bottom, width, height]
    axins = fig.add_axes(inset_pos, projection='3d')
    axins.patch.set_alpha(0)          # 背景框透明
    axins.set_facecolor("none")       # 绘图区透明
    # 隐藏 3D pane（关键）
    axins.xaxis.pane.set_edgecolor('none')
    axins.xaxis.pane.set_facecolor((1,1,1,0))   # RGBA: alpha=0

    axins.yaxis.pane.set_edgecolor('none')
    axins.yaxis.pane.set_facecolor((1,1,1,0))

    axins.zaxis.pane.set_edgecolor('none')
    axins.zaxis.pane.set_facecolor((1,1,1,0))

    # inset = inset_axes(ax, width="38%", height="38%", loc='lower right', borderpad=2)
    # 两个三维向量（需归一化）
    v1 = np.array(np.real(n1))
    v2 = np.array(np.real(n2))

    b = Bloch()
    b.fig = fig
    b.axes = axins
    b.sphere_alpha = 0.05
    # b.vector_alpha = 0.5
    b.point_alpha = 0.3
    b.frame_alpha = 0.3
    b.view = [90, 30]
    b.add_vectors(v1)
    b.add_vectors(v2)
    # axins.text(v1[0], v1[1], v1[2], f'({v1[0]:.2f}, {v1[1]:.2f}, {v1[2]:.2f})', color='black', fontsize=10)
    # axins.text(v2[0], v2[1], v2[2], f'({v2[0]:.2f}, {v2[1]:.2f}, {v2[2]:.2f})', color='black', fontsize=10)
    # 显示坐标字符串
    # coord_text = rf"$\vec{{n}}_1=$({v1[0]:.2f}, {v1[1]:.2f}, {v1[2]:.2f}), $\epsilon_1$ = {(1-np.real(cos_alpha_1))/2}"+"\n"+rf"$\vec{{n}}_2=$({v2[0]:.2f}, {v2[1]:.2f}, {v2[2]:.2f}), $\epsilon_2$ = {(1-np.real(cos_alpha_2))/2}"

    eps1 = (1 - np.real(cos_alpha_1)) / 2
    eps2 = (1 - np.real(cos_alpha_2)) / 2

    coord_text = (
    rf"$\vec{{n}}_1$ = ({v1[0]:.2f}, {v1[1]:.2f}, {v1[2]:.2f}), "
    rf"$\epsilon_1$ = {eps1:.2e}"
    "\n"
    rf"$\vec{{n}}_2$ = ({v2[0]:.2f}, {v2[1]:.2f}, {v2[2]:.2f}), "
    rf"$\epsilon_2$ = {eps2:.2e}"
)
    ax.text(
    0.98, 0.02,              # 主图内的相对坐标(0~1)
    coord_text,
    transform=ax.transAxes, # 使用轴坐标系，不随数据变化
    fontsize=15,
    ha='right', va='bottom'
)

    # inset.text(0.1,0.9, rf"$B_1$={np.round(v1,3)}", transform=inset.transAxes, fontsize=10, verticalalignment='top')
    # inset.text(0.1,0.8, rf"$B_2$={np.round(v2,3)}", transform=inset.transAxes, fontsize=10, verticalalignment='top')
    # --- 读取当前视角 ---
    curr_elev = getattr(b.axes, 'elev', 30)   # 如果没有就用默认 30
    curr_azim = getattr(b.axes, 'azim', 30)   # 如果没有就用默认 30

    # --- 绕 z 轴加 90 度 ---
    b.axes.view_init(elev=curr_elev, azim=curr_azim - 90)

    b.make_sphere()
    
    # plt.subplots_adjust(left=0.15, bottom=0.15)
    plt.tight_layout()
    plt.savefig(prefix+'trial'+str(trial)+'_Wvsdelta.pdf')
    # plt.show()
    # plt.xlim([1000,10000])
    # plt.ylim([0.676,0.696])
    # plt.yticks(np.arange(0.676, 0.696, 0.006), fontsize=12)
    # plt.legend([], frameon=False) 
    # fig.set_size_inches(3, 2)
    # plt.subplots_adjust(left=0.2) 
    # plt.subplots_adjust(bottom=0.13)
    # plt.savefig('WvsN_zoom.pdf')
    # plt.show()