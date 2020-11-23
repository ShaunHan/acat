from cabins.adsorption_sites import *
from cabins.adsorbate_coverage import *
from dscribe.descriptors import EwaldSumMatrix
from ase.io import read, write, Trajectory
from itertools import combinations
import numpy as np
import random
import pickle


adsorbate_elements = 'SCHON'


def ewald_sum(atoms):
    # Setting up the Ewald sum matrix descriptor
    esm = EwaldSumMatrix(n_atoms_max=len(atoms), flatten=False)
    ewald_matrix = esm.create(atoms, n_jobs=1)
    return ewald_matrix

def fingerprint(atoms, site_list):
    fsl = site_list
    nst = len(fsl)
    finger = np.zeros((nst, nst)))
    ads_ids = fsl.ads_ids 
    ads_atoms = atoms[ads_ids]
    ewald_matrix = ewald_sum(ads_atoms)

    frag_dict = {}
    for i, st in enumerate(fsl):
        if st['occupied'] == 1:
            frag_dict[i] = st['fragment_indices']

    combs = combinations(frag_dict.keys(), 2)
    for comb in combs:
        stid1 = comb[0]
        stid2 = comb[1]
        frag1 = frag_dict[stid1]
        frag2 = frag_dict[stid2]
        ess12, ess21 = [], []
        for i in frag1:
            fid2 = np.where(np.isin(ads_ids, frag2))[0]
            es12 = np.sum(ewald_matrix[ads_ids.index(i), fid2])
            ess12.append(es12)
        for j in frag2:
            fid1 = np.where(np.isin(ads_ids, frag1))[0]
            es21 = np.sum(ewald_matrix[ads_ids.index(j), fid1])
            ess21.append(es21)
        e12, e21 = np.sum(ess12), np.sum(ess21)
        finger[stid1,stid2] = e12
        finger[stid2,stid1] = e21

    return finger

def get_Elat(atoms, Eads_dict)
    labels = atoms.info['data']['labels']
    Eads = atoms.info['data']['Eads']
    Elabs = [Eads_dict[lab] for lab in labels]
    Elat = Eads - np.sum(Elabs)

    return Elat                

def squared_exponential_kernel(x1, x2, w):
    return np.exp(-np.linalg.norm([x1 - x2], 2)**2/(2*w**2))

def kernel(X1, X2, w):
    K = np.empty((len(X1), len(X2)))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            K[i, j] = squared_exponential_kernel(x1, x2, w)
    return K

def train(X, y, test_size=0.2):
    """Train a GPR model and returns the mean and covariance matrix"""
    train_index = int((1-test_size) * len(X))
    X_train, X_test = X[:train_index], X[train_index:]
    y_train, y_test = y[:train_index], y[train_index:]

    Kst = kernel(X_test,  X_train)
    Ktt = kernel(X_train, X_train)
    Kss = kernel(X_test,  X_test)
    Kts = kernel(X_train, X_test)
    KstKtt = np.dot(Kst, np.linalg.inv(Ktt))
    mean = np.dot(KstKtt, y_train)
    cov = Kss - np.dot(KstKtt, Kts)

    return mean, cov

def predict(x, mean, cov)
    """Predict the mean and std of a new data point x"""
    pred = np.random.multivariate_normal(mean, cov)
    std = np.sqrt(np.diag(cov))

    return pred, std

def process_data(images, adsorpition_sites, old_data=None):
    sas = adsorption_sites
    if not old_data:
        xs, ys = [], []
    else:
        None #TODO: Read old data

    for atoms in images:
        sas.update_positions(atoms)
        sac = SlabAdsorbateCoverage(atoms, sas)
        fsl = sac.full_site_list
        finger = fingerprint(atoms, fsl)
        Elat = get_Elat(atoms, Eads_dict)
        atoms.info['data']['Elat'] = Elat
        xs.append(finger)
        ys.append(Elat)

    return np.array(xs), np.array(ys)

def main():
    X, y = process_data(old, sas, old_data=None)
    mean, cov = train(X, y)
    new_images = read('xxx.traj', index=':')
    for atoms in new_images:
        pred, std = predict(finger(x), mean, cov)
        eps = pred + 2 * std
        

if __name__ == "__main__":
    main()
