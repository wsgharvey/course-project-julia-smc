import os
from glob import glob
import pickle

import matplotlib
import matplotlib.pyplot as plt

from wasserstein import wasserstein

def get_fname(particles, steps):
    return f"empirical-posteriors/posterior-{particles}-{steps}.csv"

def get_cache_name(steps):
    return f"cache/errors_{steps}.p"

stepses = [10, 100, 1000]
def get_particleses(steps):
    template = get_fname("*", str(steps))
    files = glob(template)
    def get_particles(fname):
        return int(fname.replace('-', '.').split('.')[-3])
    return [get_particles(fname) for fname in files if get_particles(fname)<80000]

empirical_distributions = {
    steps: get_particleses(steps)
    for steps in stepses
}

ground_truth = "empirical-posteriors/ground-truth-40000-200.csv"

fig, ax = plt.subplots()
cmap = matplotlib.cm.get_cmap('cool')
colors = [cmap(r/(len(stepses)-1)) for r in range(0, len(stepses))]

for i, (steps, particleses) in enumerate(empirical_distributions.items()):

    if os.path.exists(get_cache_name(steps)):
        errors, particleses = \
            pickle.load(open(get_cache_name(steps), 'rb'))
        print("loaded from cache")
    else:
        errors = [wasserstein(ground_truth,
                              get_fname(particles, steps))
                  for particles in particleses]
        # save to cache for later
        pickle.dump((errors, particleses,),
                    open(get_cache_name(steps), 'wb'))
        print("saved to cache")

    # sort stuff so plotting looks reasonable
    particleses, errors = zip(*list(sorted(zip(particleses, errors))))
    plt.plot(particleses, errors,
             label=f"{steps} steps",
             color=colors[i],
             lw=2)
    print(f"plotted {steps}")

ax.set_xlim(5)
ax.set_xscale('log')
ax.set_xlabel('Number of particles')
ax.set_ylim(0.1)
ax.set_yscale('log')
ax.set_ylabel('Wasserstein-2 distance from ground truth')
ax.legend()

# put on nice grid
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

plt.savefig('posterior-qualities.pdf', bbox_inches='tight')
