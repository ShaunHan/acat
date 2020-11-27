from ase.io import read, write
from dscribe.descriptors import ACSF
import numpy as np
import pickle


class GaussianProcess(object):
    def __init__(self, X, y, load_kernel=None, lmbda=0):                 
        super().__init__()
        self.X = X
        self.y = y
        self.N = len(self.X)
        self.lmbda = lmbda
        self.sigma = []
        self.mean = []
        self.load_kernel = load_kernel if load_kernel else \
                           self.squared_exponential_kernel
        self.setup_sigma()

    # Different types of kernels can be added as @classmethod
    @classmethod
    def squared_exponential_kernel(cls, x1, x2, w=0.5):
        return np.exp(-np.linalg.norm([x1 - x2], 2)**2 / (2*w**2))

    @classmethod
    def generate_kernel(cls, kernel, w=0.5):
        def wrapper(*args, **kwargs):
            kwargs.update({'width': w})
            return kernel(*args, **kwargs)
        return wrapper

    @classmethod
    def calculate_sigma(cls, x, load_kernel, lmbda=0):
        N = len(x)
        sigma = np.ones((N, N))
        for i in range(N):
            for j in range(i+1, N):
                cov = load_kernel(x[i], x[j])
                sigma[i][j] = cov
                sigma[j][i] = cov

        sigma = sigma + lmbda * np.eye(N)
        return sigma

    def setup_sigma(self):
        self.sigma = self.calculate_sigma(self.X, self.load_kernel, self.lmbda)

    def predict(self, x):
        cov = 1 + self.lmbda * self.load_kernel(x, x)
        sigma_1_2 = np.zeros((self.N, 1))
        for i in range(self.N):
            sigma_1_2[i] = self.load_kernel(self.X[i], x)

        K = sigma_1_2.T * np.mat(self.sigma).I
        mu = K * np.mat(self.y).T
        sigma_pred = cov + self.lmbda - K * sigma_1_2
        std = np.sqrt(np.diag(sigma_pred))
        return mu, std

    @staticmethod
    def get_probability(sigma, y, lmbda):
        multiplier = np.power(np.linalg.det(2 * np.pi * sigma), -1/2)
        return multiplier * np.exp((-1/2 * (np.mat(y) * np.dot(np.mat(sigma).I, y).T)))

    def optimize(self, lmbda_list, beta_list):
        def load_kernel_proxy(w, f):
            def wrapper(*args, **kwargs):
                kwargs.update({'width': w})
                return f(*args, **kwargs)
            return wrapper
        best = (0, 0, 0)
        history = []
        for l in lmbda_list:
            best_beta = (0, 0)
            for b in beta_list:
                sigma = gaus.calculate_sigma(self.X, load_kernel_proxy(b, self.load_kernel), l)
                marginal = b* float(self.get_probability(sigma, self.y, l))
                if marginal > best_beta[0]:
                    best_beta = (marginal, b)
            history.append((best_beta[0], l, best_beta[1]))
        return sorted(history)[-1], np.mat(history)


def normalize(X, centering=False, scaling_params=None):
    X_min = scaling_params[0] if scaling_params else X.min(axis=0)
    ptp = scaling_params[1] if scaling_params else X.ptp(axis=0)                         
    X_norm = (X - X_min) / ptp
    scaling_params = [X_min, ptp]
    if centering:
        mu = scaling_params[2] if scaling_params else X_norm.mean(axis=0)
        X_norm -= mu
        scaling_params.append(mu)
        
    return X_norm, scaling_params

        
def fingerprint(atoms, surf_ids, acsf):
    surf_acsfs = acsf.create_single(atoms, positions=surf_ids)
    surf_nums = atoms.numbers[surf_ids]
    res = np.hstack([surf_nums[:,None], surf_acsfs]) 

    return res.ravel() 
    

def collate_data(structures, surf_ids, acsf, 
                 load_pkl_data=None, 
                 save_pkl_data=None):
    if load_pkl_data:
        with open(load_pkl_data, 'rb') as input:
            data = pickle.load(input)
        xs, ys = data[0], data[1]
    else:
        xs, ys = [], []

    for atoms in structures:
        finger = fingerprint(atoms, surf_ids, acsf) 
        Eads = atoms.info['data']['Eads']
        xs.append(finger)
        ys.append(Eads)
    if save_pkl_data:
        with open(save_pkl_data, 'wb') as output:
            pickle.dump((xs, ys), output) 

    X, y = np.asarray(xs), np.asarray(ys)
    X_norm, scaling_params = normalize(X, centering=True)

    return X_norm, y, scaling_params


def main():
    dft_structures = read('NiPt3_311_1_reax.traj', index=':')
    with open('adsorption_sites_NiPt3_311.pkl', 'rb') as f:
        sas = pickle.load(f)
    surf_ids = sas.surf_ids
    acsf = ACSF(species=['H','C','O','Ni','Pt'],                   
                rcut=6.5, 
                g2_params=[[0.007,0],[0.011,0],[0.018,0],[0.029,0],
                           [0.047,0],[0.076,0],[0.124,0],[0.202,0],
                           [0.329,0],[0.996,4.626],[1.623,5.905],
                           [2.644,7.538],[4.309,9.622]],
                g4_params=[[0.007,1,1],[0.007,2,-1],[0.014,1,-1],
                           [0.014,2,-1],[0.029,1,-1],[0.029,2,-1],
                           [0.029,2,1],[0.06,1,1],[0.06,2,-1]])    

    X, y, params = collate_data(dft_structures, surf_ids, acsf,
                                load_pkl_data=None,
                                save_pkl_data='training_data_NiPt3_311.pkl')
    gpr = GaussianProcess(X, y)
    new_structures = read('NiPt2_311_2_reax.traj', index=':')
    low_energy_structures = []
    for atoms in new_structures:
        x = fingerprint(atoms, surf_ids, acsf)
        x_norm = normalize(x, centering=True, 
                           scaling_params=params)
        mu, std = gpr.predict(x_norm)
        upper = mu + 2 * std
        if upper < Ecut:
            low_energy_structures.append(atoms)
    write('NiPt3_311_3_reax.traj', low_energy_structures)                


if __name__ == "__main__":
    main()
