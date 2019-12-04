import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

n_particles = [10, 100, 1000, 10000, 100000]
cpu_10steps = [5.212963104248047, 5.212963104248047, 7.242398023605347, 38.84834694862366, 257.1643121242523]
gpu_10steps = [9.441170930862427, 9.652847051620483, 10.13875412940979, 16.774540901184082, 94.53944396972656]
cpu_100steps = [4.895235061645508, 6.571429014205933, 23.400748014450073, 294.57816100120544, 2106.7643020153046]
gpu_100steps = [8.994035959243774, 9.310070991516113, 15.541818141937256, 75.73616981506348, 795.0979599952698]

colors = sns.color_palette('colorblind')

fig, ax = plt.subplots()

for times, steps, \
    is_gpu in zip([cpu_10steps, gpu_10steps,
                   cpu_100steps, gpu_100steps],
                  [10, 10, 100, 100],
                  [False, True, False, True]):
    x = n_particles[:len(times)]
    ax.plot(x, times,
            label=f"{'GPU' if is_gpu else 'CPU'}, {steps} steps",
            color=colors[2 if steps==100 else 4],
            lw=2,
            ls='dotted' if is_gpu else 'dashdot')

ax.set_xlabel('Number of particles')
ax.set_xscale('log')
ax.set_xlim(10, 1e5)
ax.set_ylabel('Time (seconds)')
ax.set_yscale('log')

ax.legend()

# put on nice grid
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

fig.savefig('gpu_comparison.pdf', bbox_inches='tight')
