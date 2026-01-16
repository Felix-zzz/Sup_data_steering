import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

prefix = 'New_W_M/'

def random_direction_gaussian(n, sigma, size=1):
    """
    生成围绕方向 n 的随机方向向量
    角度分布 ~ 高斯分布 N(0, sigma^2)

    参数:
        n : array-like, shape (3,)
            给定的单位向量方向
        sigma : float
            高斯分布的标准差（弧度）
        size : int
            生成多少个随机方向

    返回:
        dirs : ndarray, shape (size, 3)
            随机生成的方向向量
    """
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)  # 归一化

    # 构造局部坐标系: n 作为 z 轴
    # 找一个不平行的向量来做叉积
    if abs(n[0]) < 0.9:
        ref = np.array([1,0,0])
    else:
        ref = np.array([0,1,0])
    x_axis = np.cross(ref, n)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(n, x_axis)

    # 生成随机角度
    theta = np.random.normal(loc=0.0, scale=sigma, size=size)  # 极角
    phi = np.random.uniform(0, 2*np.pi, size=size)             # 方位角

    # 局部坐标系下的向量
    dirs_local = (np.sin(theta)[:,None] * np.cos(phi)[:,None] * x_axis +
                  np.sin(theta)[:,None] * np.sin(phi)[:,None] * y_axis +
                  np.cos(theta)[:,None] * n)

    return dirs_local

def random_direction_uniform(n, angle_range, size=1):
    """
    生成围绕方向 n 的随机方向向量
    角度分布 ~ 均匀分布 U(-angle_range, angle_range)

    参数:
        n : array-like, shape (3,)
            给定的单位向量方向
        angle_range : float
            角度范围（弧度）
        size : int
            生成多少个随机方向
    返回:
        dirs : ndarray, shape (size, 3)
            随机生成的方向向量
    """
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)  # 归一化

    # 构造局部坐标系: n 作为 z 轴
    # 找一个不平行的向量来做叉积
    if abs(n[0]) < 0.9:
        ref = np.array([1,0,0])
    else:
        ref = np.array([0,1,0])
    x_axis = np.cross(ref, n)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(n, x_axis)

    # 生成随机角度
    theta = np.random.uniform(low=-angle_range, high=angle_range, size=size)  # 极角
    phi = np.random.uniform(0, 2*np.pi, size=size)             # 方位角

    # 局部坐标系下的向量
    dirs_local = (np.sin(theta)[:,None] * np.cos(phi)[:,None] * x_axis +
                  np.sin(theta)[:,None] * np.sin(phi)[:,None] * y_axis +
                  np.cos(theta)[:,None] * n)

    return dirs_local

# n = [0.707, 0, 0.707]
# sigma = 0.05     # 角度标准差 (弧度)
# samples = random_direction_gaussian(n, sigma, size=1)
# print(samples[0].tolist())
# noisy_direction = samples[0]
# elist = np.linspace(0, 0.012, 11)
# epsilon = 0.0036
elist = [ 0.0024]
epsilon = 0.0024
theta = np.pi/8
sigma_infid = 0.0012     # 角度标准差 infid
p = 0.8
Z = np.array([[1,0],[0,-1]])
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])

N_list = np.arange(10,1010,10)
N_list = np.insert(N_list, 0, np.arange(1,10))
N_list = np.append(N_list, np.arange(1100, 10100,100))

# Only keep N=500
N_list = [500]

plt.figure()
# Randomly generate 25 trials of different coherent measurement noise
for trial in range(5, 30):
    alpha = np.arccos(1-2*epsilon)
    A1 = (1 / np.sqrt(1 + np.sin(2 * theta) ** 2)) * (Z - np.sin(2 * theta) * X)
    A2 = (1 / np.sqrt(1 + np.sin(2 * theta) ** 2)) * (Z + np.sin(2 * theta) * X)
    B10 = np.cos(np.pi/4) * Z - np.sin(np.pi/4) * X
    B20 = np.cos(np.pi/4) * Z + np.sin(np.pi/4) * X
    # B1 = np.cos(np.pi/4-alpha) * Z - np.sin(np.pi/4-alpha) * X
    # B2 = np.cos(np.pi/4-alpha) * Z + np.sin(np.pi/4-alpha) * X

    
    n10 = [-np.cos(np.pi/4), 0, np.sin(np.pi/4)]
    # U10 rotates n10 to Z direction
    UM10 = la.expm(-0.5j * np.pi/4 * Y)
   
    n20 = [np.cos(np.pi/4), 0, np.sin(np.pi/4)]
    UM20 = la.expm(0.5j * np.pi/4 * Y)

    nY0 = np.array([0, 1, 0])
    alpha = np.arccos(1-2*epsilon)
    nY1 = random_direction_gaussian(nY0, alpha, size=1)[0]
    nY2 = random_direction_gaussian(nY0, alpha, size=1)[0]
    thetaY1 = np.random.normal(loc=np.pi/4, scale=alpha, size=1)[0]
    thetaY2 = np.random.normal(loc=-np.pi/4, scale=alpha, size=1)[0]
    UMY1 = la.expm(-0.5j * thetaY1 * (nY1[0] * X + nY1[1] * Y + nY1[2] * Z))
    UMY2 = la.expm(-0.5j * thetaY2 * (nY2[0] * X + nY2[1] * Y + nY2[2] * Z))
    effective_B1 = UMY1.conj().T @ Z @ UMY1
    effective_B2 = UMY2.conj().T @ Z @ UMY2

    ideal = np.array([np.cos(theta), 0, 0, np.sin(theta)])
    rho = p * np.outer(ideal,ideal) + (1-p) * np.eye(4)/4
    W0 = 0.5 * np.trace(rho @ (np.kron(A1, B10) + np.kron(A2, B20)))
    RM_means = []
    RM_stds = []
    RM_noisy_means = []
    RM_noisy_stds = []
    Wc_list = []
    Wc = 0.5 * np.trace(rho @ (np.kron(A1, effective_B1) + np.kron(A2, effective_B2)))
    Wc_list.append(Wc)

    for N in N_list:
        RM_list=[]
        RM_list_noisy=[]
        for smp in range(N):
            phi_rot1 = np.random.uniform(0, 2 * np.pi)
            phi_rot2 = np.random.uniform(0, 2 * np.pi)
            U10 = la.expm(-0.5j * phi_rot1 * B10)
            U20 = la.expm(-0.5j * phi_rot2 * B20)

            W_RM = np.real(0.5 * np.trace(rho @ (np.kron(A1, U10.conj().T @ effective_B1 @ U10) + np.kron(A2, U20.conj().T @ effective_B2 @ U20))))

            RM_list.append(W_RM)
            W_RM_noisy = []
            # alpha_list = []

            for sigma_mul in np.linspace(0.01,3,300):

                infid = sigma_mul * sigma_infid
                alpha = np.arccos(1-2*infid)
                # alpha_list.append(alpha)

                noisy_direction1 = random_direction_uniform(n10, alpha, size=1)[0]
                noisy_direction2 = random_direction_uniform(n20, alpha, size=1)[0]
                Un1 = noisy_direction1[0] * X + noisy_direction1[1] * Y + noisy_direction1[2] * Z
                Un2 = noisy_direction2[0] * X + noisy_direction2[1] * Y + noisy_direction2[2] * Z
                U1 = la.expm(-0.5j * phi_rot1 * Un1)
                U2 = la.expm(-0.5j * phi_rot2 * Un2)
                # W = p/np.sqrt(2) * np.sqrt(1+np.sin(2*theta)**2) * (1 - 2*epsilon) + p/np.sqrt(2) * (1-np.sin(2*theta)**2) / np.sqrt(1+np.sin(2*theta)**2) * 2*np.sqrt(epsilon*(1-epsilon))
                # print(W, Wc)
                W_RM_noisy.append(np.real(0.5 * np.trace(rho @ (np.kron(A1, U1.conj().T @ effective_B1 @ U1) + np.kron(A2, U2.conj().T @ effective_B2 @ U2)))))
            W_RM_noisy = np.array(W_RM_noisy)
            RM_list_noisy.append(W_RM_noisy)
            # print(f"SMP {smp}: W0={W0:.4f}, Wc={Wc:.4f}, W_RM={W_RM:.4f}, W_RM_noisy={W_RM_noisy}")
        # print(len(RM_list), len(RM_list_noisy))
        # plt.figure()
        # bins = np.linspace(0.52, 0.74, 50)
        # plt.hist(RM_list, bins=bins, alpha=0.5, label='Ideal RM')
        # plt.hist(RM_list_noisy, bins=bins, alpha=0.5, label='Noisy RM')
        # plt.axvline(np.mean(RM_list), color='blue', linestyle='dashed', linewidth=1)
        # plt.axvline(np.mean(RM_list_noisy), color='orange', linestyle='dashed', linewidth=1)
        # plt.axvline(W0, color='green', linestyle='dashed', linewidth=1, label='W0')
        # plt.axvline(Wc, color='red', linestyle='dashed', linewidth=1, label='Wc')
        # plt.legend()
        # plt.xlabel('RM Value')
        # plt.ylabel('Frequency')
        # plt.title('Histogram of RM Values')
        # plt.savefig(f'e={epsilon}, N={N}.pdf')
        # plt.show()
        RM_list_noisy = np.array(RM_list_noisy)
        RM_means.append(np.mean(RM_list))
        RM_stds.append(np.std(RM_list))
        RM_noisy_means.append(np.mean(RM_list_noisy, axis = 0))
        RM_noisy_stds.append(np.std(RM_list_noisy, axis = 0))
        print(f"trial={trial}, e={epsilon}, N={N}: W0={W0:.4f}, Wc={Wc:.4f}")#, RM_mean={np.mean(RM_list):.4f}±{np.std(RM_list):.4f}, RM_noisy_mean={np.mean(RM_list_noisy, axis=0)}±{np.std(RM_list_noisy, axis=0)}")
    # plt.show()
    # print(f'RM_means={RM_means}')
    # print(f'RM_stds={RM_stds}')
    # print(f'RM_noisy_means={RM_noisy_means}')
    # print(f'RM_noisy_stds={RM_noisy_stds}')
    RM_means = np.array(RM_means)
    RM_stds = np.array(RM_stds)
    RM_noisy_means = np.array(RM_noisy_means)
    RM_noisy_stds = np.array(RM_noisy_stds)
    # plt.plot([0,0.012],[0.707,0.707], linestyle='dashed',label = r'$\mathcal{C}_2$', color = 'grey')
    # plt.plot([0,0.012], [W0, W0], linestyle = 'solid', label = r'$W_2$', color = 'blue')
    # plt.plot(elist, Wc_list, linestyle = 'solid', label = r'\tilde{W}_2^{\epsilon}', color = 'red')
    # plt.errorbar(elist, RM_means, RM_stds, alpha = 0.5, label = 'Exact RM', color = 'orange')
    # plt.errorbar(elist, RM_noisy_means, RM_noisy_stds, alpha = 0.5, label = 'Noisy RM', color = 'purple')
    # plt.savefig('WvsE.pdf')
    # plt.plot([0,100000],[0.707,0.707], linestyle='dashed',label = r'$\mathcal{C}_2$', color = 'grey')
    # plt.plot([0,100000], [W0, W0], linestyle = 'solid', label = r'$W_2$', color = 'blue')
    # np.save('RM_means.npy', RM_means)
    # np.save('RM_stds.npy', RM_stds)
    # np.save('RM_noisy_means.npy', RM_noisy_means)
    # np.save('RM_noisy_stds.npy', RM_noisy_stds)
    # np.save('N_list.npy', N_list)
    # np.save('Wc_list.npy', Wc_list)

    np.save(prefix + str(trial) + '_UM1.npy', UMY1)
    np.save(prefix + str(trial) + '_UM2.npy', UMY2)
    np.save(prefix + str(trial) + '_effective_B1.npy', effective_B1)
    np.save(prefix + str(trial) + '_effective_B2.npy', effective_B2)

    np.save(prefix + str(trial) + '_RM_means.npy', RM_means)
    np.save(prefix + str(trial) + '_RM_stds.npy', RM_stds)
    np.save(prefix + str(trial) + '_RM_noisy_means.npy', RM_noisy_means)
    np.save(prefix + str(trial) + '_RM_noisy_stds.npy', RM_noisy_stds)
    # np.save('N_list.npy', np.array([10000]*len(elist)))
    np.save(prefix + str(trial) + '_N_list.npy', N_list)
    np.save(prefix + str(trial) + '_Wc_list.npy', Wc_list)

    # np.save(prefix + 'data_ideal.npy', np.array(data_ideal))
    # np.save(prefix + 'data_noisy.npy', np.array(data_noisy))
    plt.plot(N_list, RM_means, linestyle = 'solid', label = f'Exact e = {epsilon}')
    for i in range(len(RM_noisy_means[0])):
        plt.plot(N_list, RM_noisy_means[:,i], linestyle = 'dashed', label = f'Noisy e = {epsilon}, sigma = {(i+1)*sigma_infid:.4f}')
    # plt.plot(N_list, RM_noisy_means, linestyle = 'dashed', label = f'Noisy e = {epsilon}')
    

plt.xscale('log') 
plt.xlabel('Number of Samples')
plt.ylabel('W Value')    
plt.legend()
plt.savefig('WvsN.pdf')
plt.show()
