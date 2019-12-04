import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

particleses = {1: [10, 100, 1000, 10000, 100000],  # per process particleses
               2: [5, 50, 500, 5000, 50000, 500000, 5000000],
               4: [3, 25, 250, 2500, 25000, 250000]}

timeses = {(1, 10): [9.441170930862427, 9.652847051620483, 10.13875412940979,
                     16.774540901184082, 94.53944396972656],
           (1, 100): [8.994035959243774, 9.310070991516113, 15.541818141937256,
                      75.73616981506348, 795.0979599952698],
           (2, 10): [10.06730604171753, 10.14134407043457, 10.558981895446777,
                     14.56136703491211, 61.70177102088928, 410.12130999565125],
           (2, 100): [10.611931800842285, 10.768099069595337, 14.296647071838379,
                      48.765763998031616, 528.5079121589661],
           (4, 10): [9.886150121688843, 9.948958158493042, 10.40027904510498,
                     12.09963607788086, 34.06696391105652],
           (4, 100): [11.691739082336426, 10.641443967819214, 12.713840961456299,
                      28.736865997314453, 247.11202716827393, 2001.2320849895477]}

colors = sns.color_palette('colorblind')
linestyles = {1: 'solid', 2: 'dashed', 4: 'dotted'}

fig, ax = plt.subplots()

for (processes, steps), times in timeses.items():
    particles = particleses[processes]
    x = processes * np.array(particles[:len(times)])
    ax.plot(x, times,
            label=f"{processes} processes, {steps} steps",
            color=colors[2 if steps==100 else 4],
            lw=2,
            ls=linestyles[processes])

ax.set_xlabel('Number of particles')
ax.set_xscale('log')
ax.set_ylabel('Time (seconds)')
ax.set_yscale('log')
ax.set_xlim(10)

ax.legend()

# put on nice grid
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

fig.savefig('multiprocessing_comparison.pdf', bbox_inches='tight')
