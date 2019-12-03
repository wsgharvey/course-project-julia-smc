"""
Computes Wasserstein distance between two empirical distributions.
"""
import numpy as np
import ot
import argparse


def wasserstein(path1, path2):

    # load arrays-------------------------------------------------
    def load_csv(path):
        return np.loadtxt(open(path, "rb"), delimiter=",", skiprows=1)
    empirical1 = load_csv(path1)
    empirical2 = load_csv(path2)

    # create cost matrix------------------------------------------
    def cost(p1, p2):
        return ((p1-p2)**2).sum()
    M = np.array([[cost(sample1, sample2)
                   for sample2 in empirical2]
                  for sample1 in empirical1])

    # compute Wasserstein distance--------------------------------
    return ot.emd2([], [], M)


if __name__ == '__main__':
    # parse command line arguments (file names)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dist1")
    parser.add_argument("dist2")
    args = parser.parse_args()

    print(wasserstein(
        f'empirical-posteriors/{args.dist1}.csv',
        f'empirical-posteriors/{args.dist2}.csv'
    ))
