from acat.utilities import (neighbor_shell_list, 
                            get_adj_matrix, 
                            hash_composition)
from multiprocessing import Pool
from itertools import chain
from functools import partial
import networkx as nx
import numpy as np
import os


class Neighborhood(object):

    def __init__(self,
                 hp={'alpha': np.array([.75]),
                     'length': np.array([200.]), 
                     'kmax': 1},
                 n_jobs=os.cpu_count()):
        self.hp=hp
        self.set_hyperparams(hp)
        self.n_jobs = n_jobs

    def set_hyperparams(self, new_params):
        """Set or update the hyperparameters for the Kernel.
            Parameters:
                new_params: dictionary
                    A dictionary of hyperparameters that are added or updated.
        """
        self.hp.update(new_params)
        if 'alpha' not in self.hp:
            self.hp['alpha'] = np.array([.75])
        if 'length' not in self.hp:
            self.hp['length'] = np.array([200.])

        # Lower and upper machine precision
        eps_mach_lower = np.sqrt(1.01 * np.finfo(float).eps)
        eps_mach_upper = 1 / eps_mach_lower
        self.hp['length'] = np.abs(self.hp['length']).reshape(-1)
        self.hp['alpha'] = np.abs(self.hp['alpha']).reshape(-1)
        self.hp['length'] = np.where(self.hp['length'] < eps_mach_upper, 
                                     np.where(self.hp['length'] > eps_mach_lower, 
                                              self.hp['length'], eps_mach_lower), 
                                     eps_mach_upper)
        self.hp['alpha'] = np.where(self.hp['alpha'] < eps_mach_upper, 
                                    np.where(self.hp['alpha'] > eps_mach_lower, 
                                             self.hp['alpha'], eps_mach_lower), 
                                    eps_mach_upper)
        return self.hp

    def dist_m(self, train_images, test_images=None):
        dists = self.__call__(train_images, test_images, get_dists=True)

        return dists

    def __call__(self, train_images, test_images=None, get_derivatives=False, get_dists=False, dists=None):
        if test_images is None:
            images = list(train_images)
        else:
            images = list(train_images) + list(test_images)
        if dists is None:
            pool = Pool(self.n_jobs)
            dicts = pool.map(self.get_dict, images)
#            for d in dicts:
#                all_keys = set(chain.from_iterable(dicts))   
#                old_keys = d.keys()
#                d.update((k, 0) for k in all_keys - old_keys)
            dicts = np.asarray(dicts, dtype=object)
            if get_dists:
                dists = dicts
        else:
            dicts = dists
            print('length: {}'.format(self.hp['length']))
            print('alpha: {}'.format(self.hp['alpha']))
            print('noise: {}'.format(self.hp['noise']))
        if get_dists:
            return dists

        # Initialize the similarity kernel
        K = np.zeros(shape=(len(images), len(images)))
        # Iterate through upper triangular matrix indices
        idx0, idx1 = np.triu_indices_from(K)
        # Perform similarity calculations in parallel.
        pool = Pool(self.n_jobs)
        po = partial(self.pairwise_operation)
        sims = pool.starmap(po, zip(dicts[idx0], dicts[idx1]))

        # Insert it into K[i,j] and K[j,i]
        for i, sim in enumerate(sims):
            K[idx0[i],idx1[i]] = sim
        # Symmetrize
        K = K + K.T - np.diag(np.diag(K))
        # Take the upper right rectangle if there are test images
        if test_images is not None:
            K = K[:len(train_images),len(train_images):len(images)] 

        return K

    def get_hyperparameters(self):
        "Get all the hyperparameters"
        return self.hp 

    def get_dict(self, atoms):
        d = {} 
        symbols = atoms.symbols
        nblist = neighbor_shell_list(atoms, dx=0.3, neighbor_number=1, mic=True)
        A = get_adj_matrix(nblist)
        G = nx.from_numpy_matrix(A) 
        if 'kmax' in self.hp:
            kmax = self.hp['kmax']
        else:
            kmax = 1
        for i in range(len(A)):
            lab0 = symbols[i]
            if lab0 in d:
                d[lab0] += 1.
            else:
                d[lab0] = 1.
            kd = nx.single_source_shortest_path_length(G, i, cutoff=kmax)
            for k in range(kmax):
                nbrs = [j for j, v in kd.items() if v == k + 1]
                if len(nbrs) == 0:
                    continue
                lab = lab0 + '-' + str(k+1) + '-' + ''.join(sorted(symbols[nbrs]))
                if lab in d:
                    d[lab] += self.hp['alpha'][k] 
                else:
                    d[lab] = self.hp['alpha'][k]
        return d

    def pairwise_operation(self, dx, dy):
        """Compute pairwise similarity between every two dicts."""
        dxy = {k: [d[k] if k in d else 0 for d in (dx, dy)] 
               for k in set(dx.keys()) | set(dy.keys())}
        X = np.asarray(list(dxy.values()))
#        k = X[:,0] @ X[:,1].T
#        k = np.exp(-np.linalg.norm(X[:,0] - X[:,1]) / self.hp['length'][0]**2)
        k = np.exp(-np.sum((X[:,0] - X[:,1])**2) / (2 * self.hp['length'][0]**2))

        return k

    def diag(self, X, get_derivatives=False):
        """Get the diagonal kernel vector.

        Parameters:
            X : (N,D) array
                Features with N data points and D dimensions.
        """
        return np.ones(len(X))

    def __repr__(self):
        return 'Neighborhood(hp={})'.format(self.hp)
