import numpy as np
import matplotlib.pyplot as plt

# this a workaround for a numpy bug in file io (http://github.com/numpy/numpy#5655)
def my_loadtxt(*args, **kw):
    if 'dtype' in kw and (kw['dtype'] == 'complex' or kw['dtype'] is np.complex):
        return np.genfromtxt(*args, **kw)
    else:
        return unpatched_loadtxt(*args, **kw)

unpatched_loadtxt = np.loadtxt
np.loadtxt = my_loadtxt

dat = np.loadtxt('out.txt', dtype=np.complex)
t = np.real(dat[:, 0])
S_load = dat[:, 1]
S_norm_gen = dat[:, 2]
S_inv_gen = dat[:, 3]
V_min = np.abs(dat[:, 4])
V_max = np.abs(dat[:, 5])
cloud = dat[:, 6]

dat_ctrl = np.loadtxt('out_ctrl.txt', dtype=np.complex)
V_min_ctrl = np.abs(dat_ctrl[:, 4])
V_max_ctrl = np.abs(dat_ctrl[:, 5])

(fig, (ax1, ax2)) = plt.subplots(2, 1)
ax1.hold(True)
ax2.hold(True)

ax1.fill_between(t, V_min_ctrl, V_max_ctrl, where=V_max_ctrl >= V_min_ctrl,
                facecolor='green', alpha=0.5, linewidth=0.0)
ax1.fill_between(t, V_min, V_max, where=V_max >= V_min,
                facecolor='red', alpha=0.5, linewidth=0.0)
ax1.axhline(0.96, color='black')
ax1.axhline(1.04, color='black')
ax1.set_xlim([0, 72])
ax1.set_ylim([0.9, 1.1])

ax2.plot(t, np.real(S_load))
ax2.plot(t, np.real(S_inv_gen), color='red')
ax2.plot(t, np.imag(S_inv_gen))
ax2.set_xlim([0, 72])

plt.show()