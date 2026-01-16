import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

prefix = 'N=2_uniform/'
def random_direction_uniform(n, sigma, size=1):
    """
    生成围绕方向 n 的随机方向向量
    角度分布 ~ 均匀分布 U(-sigma, sigma)

    参数:
        n : array-like, shape (3,)
            给定的单位向量方向
        sigma : float
            均匀分布的范围（弧度）
        size : int
            生成多少个随机方向
    """
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)  # 归一化

    # 生成随机角度
    theta = np.random.uniform(-sigma, sigma, size=size)  # 极角
    phi = np.random.uniform(0, 2*np.pi, size=size)        # 方位角

    # 构造局部坐标系: n 作为 z 轴
    # 找一个不平行的向量来做叉积
    if abs(n[0]) < 0.9:
        ref = np.array([1,0,0])
    else:
        ref = np.array([0,1,0])
    x_axis = np.cross(ref, n)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(n, x_axis)
    # 局部坐标系下的向量
    dirs_local = (np.sin(theta)[:,None] * np.cos(phi)[:,None] * x_axis +
                  np.sin(theta)[:,None] * np.sin(phi)[:,None] * y_axis +
                  np.cos(theta)[:,None] * n)
    return dirs_local

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

# n = [0.707, 0, 0.707]
# sigma = 0.05     # 角度标准差 (弧度)
# samples = random_direction_gaussian(n, sigma, size=1)
# print(samples[0].tolist())
# noisy_direction = samples[0]
elist = np.linspace(0, 0.012, 11)
theta = np.pi/8
# sigma = 0.05     # 角度标准差 (弧度)
sigma_infid = 0.0024
sigma = np.arccos(1-2*sigma_infid)
p = 0.8
Z = np.array([[1,0],[0,-1]])
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
RM_means = []
RM_stds = []
RM_noisy_means = []
RM_noisy_stds = []
Wc_list = []
data_ideal = []
data_noisy = []
for epsilon in elist:
    alpha = np.arccos(1-2*epsilon)
    A1 = (1 / np.sqrt(1 + np.sin(2 * theta) ** 2)) * (Z - np.sin(2 * theta) * X)
    A2 = (1 / np.sqrt(1 + np.sin(2 * theta) ** 2)) * (Z + np.sin(2 * theta) * X)
    B10 = np.cos(np.pi/4) * Z - np.sin(np.pi/4) * X
    B20 = np.cos(np.pi/4) * Z + np.sin(np.pi/4) * X
    B1 = np.cos(np.pi/4-alpha) * Z - np.sin(np.pi/4-alpha) * X
    B2 = np.cos(np.pi/4-alpha) * Z + np.sin(np.pi/4-alpha) * X

    RM_list=[]
    RM_list_noisy=[]
    n10 = [-np.sin(np.pi/4), 0, np.cos(np.pi/4)]
    n20 = [np.sin(np.pi/4), 0, np.cos(np.pi/4)]

    ideal = np.array([np.cos(theta), 0, 0, np.sin(theta)])
    rho = p * np.outer(ideal,ideal) + (1-p) * np.eye(4)/4
    W0 = 0.5 * np.trace(rho @ (np.kron(A1, B10) + np.kron(A2, B20)))
    Wc = 0.5 * np.trace(rho @ (np.kron(A1, B1) + np.kron(A2, B2)))
    Wc_list.append(Wc)

    for smp in range(10000):
        phi_rot1 = np.random.uniform(0, 2 * np.pi)
        phi_rot2 = np.random.uniform(0, 2 * np.pi)
        U10 = la.expm(-0.5j * phi_rot1 * B10)
        U20 = la.expm(-0.5j * phi_rot2 * B20)
        noisy_direction1 = random_direction_uniform(n10, sigma, size=1)[0]
        noisy_direction2 = random_direction_uniform(n20, sigma, size=1)[0]
        Un1 = noisy_direction1[0] * X + noisy_direction1[1] * Y + noisy_direction1[2] * Z
        Un2 = noisy_direction2[0] * X + noisy_direction2[1] * Y + noisy_direction2[2] * Z
        U1 = la.expm(-0.5j * phi_rot1 * Un1)
        U2 = la.expm(-0.5j * phi_rot2 * Un2)
        # W = p/np.sqrt(2) * np.sqrt(1+np.sin(2*theta)**2) * (1 - 2*epsilon) + p/np.sqrt(2) * (1-np.sin(2*theta)**2) / np.sqrt(1+np.sin(2*theta)**2) * 2*np.sqrt(epsilon*(1-epsilon))
        # print(W, Wc)
        W_RM = np.real(0.5 * np.trace(rho @ (np.kron(A1, U10.conj().T @ B1 @ U10) + np.kron(A2, U20.conj().T @ B2 @ U20))))
        W_RM_noisy = np.real(0.5 * np.trace(rho @ (np.kron(A1, U1.conj().T @ B1 @ U1) + np.kron(A2, U2.conj().T @ B2 @ U2))))
        RM_list.append(W_RM)
        RM_list_noisy.append(W_RM_noisy)
        print(f"SMP {smp}: W0={W0:.4f}, Wc={Wc:.4f}, W_RM={W_RM:.4f}, W_RM_noisy={W_RM_noisy:.4f}")
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
    # plt.savefig(f'e={epsilon}.pdf')
    # plt.show()
    RM_means.append(np.mean(RM_list))
    RM_stds.append(np.std(RM_list))
    RM_noisy_means.append(np.mean(RM_list_noisy))
    RM_noisy_stds.append(np.std(RM_list_noisy))
    data_ideal.append(RM_list)
    data_noisy.append(RM_list_noisy)
    # plt.show()
print(f'RM_means={RM_means}')
print(f'RM_stds={RM_stds}')
print(f'RM_noisy_means={RM_noisy_means}')
print(f'RM_noisy_stds={RM_noisy_stds}')
RM_means = np.array(RM_means)
RM_stds = np.array(RM_stds)
RM_noisy_means = np.array(RM_noisy_means)
RM_noisy_stds = np.array(RM_noisy_stds)
np.save(prefix + 'RM_means.npy', RM_means)
np.save(prefix + 'RM_stds.npy', RM_stds)
np.save(prefix + 'RM_noisy_means.npy', RM_noisy_means)
np.save(prefix + 'RM_noisy_stds.npy', RM_noisy_stds)
# np.save('N_list.npy', np.array([10000]*len(elist)))
np.save(prefix + 'Wc_list.npy', np.array(Wc_list))
np.save(prefix + 'data_ideal.npy', np.array(data_ideal))
np.save(prefix + 'data_noisy.npy', np.array(data_noisy))

# plt.figure()
# plt.plot([0,0.012],[0.707,0.707], linestyle='dashed',label = r'$\mathcal{C}_2$', color = 'grey')
# plt.plot([0,0.012], [W0, W0], linestyle = 'solid', label = r'$W_2$', color = 'blue')
# plt.plot(elist, Wc_list, linestyle = 'solid', label = r'\tilde{W}_2^{\epsilon}', color = 'red')
# plt.errorbar(elist, RM_means, RM_stds, alpha = 0.5, label = 'Exact RM', color = 'orange')
# plt.errorbar(elist, RM_noisy_means, RM_noisy_stds, alpha = 0.5, label = 'Noisy RM', color = 'purple')
# plt.savefig('WvsE.pdf')