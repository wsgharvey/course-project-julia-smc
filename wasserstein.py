"""
Computes Wasserstein distance between two empirical distributions.
"""
import numpy as np
import ot
import argparse


def wasserstein(path1, path2):
    import time; start = time.time()

    # load arrays-------------------------------------------------
    def load_csv(path):
        return np.loadtxt(open(path, "rb"), delimiter=",", skiprows=1)
    empirical1 = load_csv(path1)
    empirical2 = load_csv(path2)

    # create cost matrix------------------------------------------
    empirical1 = np.expand_dims(empirical1, axis=0)
    empirical2 = np.expand_dims(empirical2, axis=1)
    M = ((empirical1-empirical2)**2).sum(axis=2)

    # compute Wasserstein distance--------------------------------
    w = ot.emd2([], [], M)
    return w


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
