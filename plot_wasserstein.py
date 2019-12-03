import os
from glob import glob
import pickle

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
    return [get_particles(fname) for fname in files]

empirical_distributions = {
    steps: get_particleses(steps)
    for steps in stepses
}

ground_truth = "empirical-posteriors/ground-truth-40000-200.csv"

fig, ax = plt.subplots()
for steps, particleses in empirical_distributions.items():

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

    plt.plot(particleses, errors, label=steps)
    print(f"plotted {steps}")

plt.xscale('log')
plt.legend()
plt.savefig('posterior-qualities.pdf', bbox_inches='tight')
