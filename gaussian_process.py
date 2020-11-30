from allocat.adsorbate_coverage import SlabAdsorbateCoverage
from ase.io import read, write
from ase.units import kB
import numpy as np
import pickle

Eads_dict = {'1H': -1.1,'1C': -1.2,'1O': -1.3,'1CH': -1.0,'1CH2': -1.4,'1CH3': -1.2,'1OH': -0.9,'1CO': -1.4,'1COH': -0.7,
             '12H': -1.1,'12C': -1.2,'12O': -1.3,'12CH': -1.0,'12CH2': -1.4,'12CH3': -1.2,'12OH': -0.9,'12CO': -1.4,'12COH': -0.7,
             '23H': -1.1,'23C': -1.2,'23O': -1.3,'23CH': -1.0,'23CH2': -1.4,'23CH3': -1.2,'23OH': -0.9,'23CO': -1.4,'23COH': -0.7,}
Edis_dict = {'NiNi': 2.0422, 'NiPt': 2.7983, 'PtNi': 2.7983, 'PtPt': 3.142}

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

        
def fingerprint(atoms, adsorption_sites, alpha=1.):
    """alpha: Importance factor of metal-metal interactions
    compared to metal-adsorbate interactions
    """
    sac = SlabAdsorbateCoverage(atoms, adsorption_sites)
    fsl = sac.full_site_list
    cm = sac.connectivity_matrix
    surf_ids = sac.surf_ids
    symbols = atoms.symbols
    finger = np.zeros(len(atoms))
    for st in fsl:
        Eads = Eads_dict[st['label']]
        indices = st['indices'][:3] if st['site'] == 'subsurf' else st['indices']
        for i in indices:
            finger[i] += Eads / len(indices)
    for si in surf_ids:
        nbids = np.where(cm[si]==1)[0]
        Edis = np.sum([Edis_dict[symbols[si]+symbols[i]] for i in nbids])
        finger[si] += alpha * Edis

    return finger[surf_ids]
    

def collate_data(structures, 
                 adsorption_sites, 
                 alpha=1.,
                 load_pkl_data=None, 
                 save_pkl_data=None):
    if load_pkl_data:
        with open(load_pkl_data, 'rb') as input:
            data = pickle.load(input)
        xs, ys = data[0], data[1]
    else:
        xs, ys = [], []

    for atoms in structures:
        sas = adsorption_sites
        finger = fingerprint(atoms, sas, alpha) 
        Eads = atoms.info['data']['Eads_dft']
        xs.append(finger)
        ys.append(Eads)
    if save_pkl_data:
        with open(save_pkl_data, 'wb') as output:
            pickle.dump((xs, ys), output) 

    X, y = np.asarray(xs), np.asarray(ys)
    X_norm, scaling_params = normalize(X, centering=True)

    return X_norm, y, scaling_params


def main():
    # Temperature for MC step(K)
    T = 300
    dft_structures = read('NiPt3_311_1_reax.traj', index=':')
    with open('adsorption_sites_NiPt3_311.pkl', 'rb') as f:
        sas = pickle.load(f)
    X, y, params = collate_data(dft_structures, sas,
                                alpha=1.,
                                load_pkl_data=None,
                                save_pkl_data='training_data_NiPt3_311.pkl')
    gpr = GaussianProcess(X, y)
    new_structures = read('NiPt2_311_2_reax.traj', index=':')
    low_energy_structures = []
    for atoms in new_structures:
        x = fingerprint(atoms, sas)
        x_norm = normalize(x, centering=True, 
                           scaling_params=params)
        mu, std = gpr.predict(x_norm)
        Eupper = mu + 2 * std
        if Eupper < Ecut:
            # Metropolis MC step
            Eprev = atoms.info['data']['Eads_dft']
            p_normal = np.minimum(1, (np.exp(-(Eupper - Eprev) / (kB * T))))
            if np.random.rand() < p_normal:
                atoms.info['data']['Eads_gpr'] = (mu, std)                     
                low_energy_structures.append(atoms)
    write('NiPt3_311_3_reax.traj', low_energy_structures)                


if __name__ == "__main__":
    main()