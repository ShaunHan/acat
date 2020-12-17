from ase.data import covalent_radii
from ase.geometry import find_mic
from collections import defaultdict
from itertools import product, permutations
import networkx as nx
import numpy as np
import scipy
import math


def neighbor_shell_list(atoms, dx=0.3, neighbor_number=1, 
                        different_species=False, mic=False,
                        radius=None, span=False):
    """Make dict of neighboring shell atoms for both periodic and 
    non-periodic systems.

    Possible to return neighbors from defined neighbor shell e.g. 1st, 2nd,
    3rd by changing the neighbor number.
    Parameters
    ----------
    dx : float
        Buffer to calculate nearest neighbor pairs.
    neighbor_number : int
        Neighbor shell.
    different_species : boolean
        Whether each neighbor pair are different species or not
    mic: boolean
        Whether to apply minimum image convention
    radius: float
        The radius of each shell. If not specified, use covalent radii 
    span: boolean
        Whether to include all neighbors spanned within the shell
    """

    atoms = atoms.copy()
    natoms = len(atoms)
    cell = atoms.cell
    nums = set(atoms.numbers)
    pairs = product(nums, nums)
    res = np.asarray(list(permutations(np.asarray(range(natoms)),2)))
    indices1, indices2 = res[:,0], res[:,1]
    positions = atoms.positions
    p1, p2 = positions[indices1], positions[indices2]

    if mic:
        _, distances = find_mic(p2 - p1, cell, pbc=True)
    else:
        distances = np.linalg.norm(p2 - p1)
    ds = np.insert(distances, np.arange(0, distances.size, natoms), 0.)

    if not radius:
        cr_dict = {(i, j): (covalent_radii[i]+covalent_radii[j]) for i, j in pairs}
    
    conn = {k: [] for k in range(natoms)}
    for atomi in atoms:
        for atomj in atoms:
            i, j = atomi.index, atomj.index
            if i != j:
                if not (different_species & (atomi.symbol == atomj.symbol)):
                    d = ds[i*natoms:(i+1)*natoms][j]
                    crij = 2 * radius if radius else cr_dict[(atomi.number, atomj.number)] 

                    if neighbor_number == 1 or span:
                        d_max1 = 0.
                    else:
                        d_max1 = (neighbor_number - 1) * crij + dx

                    d_max2 = neighbor_number * crij + dx

                    if d > d_max1 and d < d_max2:
                        conn[atomi.index].append(atomj.index)

    return conn


def get_connectivity_matrix(neighborlist):
    """Generate a connections matrix from a neighborlist object.""" 

    conn_mat = []
    index = range(len(neighborlist.keys()))
    # Create binary matrix denoting connections.
    for index1 in index:
        conn_x = []
        for index2 in index:
            if index2 in neighborlist[index1]:
                conn_x.append(1.)
            else:
                conn_x.append(0.)
        conn_mat.append(conn_x)

    return np.asarray(conn_mat)

def expand_cell(atoms, cutoff=None, padding=None):
    """Return Cartesian coordinates atoms within a supercell
    which contains repetitions of the unit cell which contains
    at least one neighboring atom.

    Parameters
    ----------
    atoms : Atoms object
        Atoms with the periodic boundary conditions and unit cell
        information to use.
    cutoff : float
        Radius of maximum atomic bond distance to consider.
    padding : ndarray (3,)
        Padding of repetition of the unit cell in the x, y, z
        directions. e.g. [1, 0, 1].

    Returns
    -------
    index : ndarray (N,)
        Indices associated with the original unit cell positions.
    coords : ndarray (N, 3)
        Cartesian coordinates associated with positions in the
        supercell.
    offsets : ndarray (M, 3)
        Integer offsets of each unit cell.
    """
    cell = atoms.cell
    pbc = atoms.pbc
    pos = atoms.positions

    if padding is None and cutoff is None:
        diags = np.sqrt((np.dot([[1, 1, 1],
                                [-1, 1, 1],
                                [1, -1, 1],
                                [-1, -1, 1]],
                                cell)**2).sum(1))

        if pos.shape[0] == 1:
            cutoff = max(diags) / 2.
        else:
            dpos = (pos - pos[:, None]).reshape(-1, 3)
            Dr = np.dot(dpos, np.linalg.inv(cell))
            D = np.dot(Dr - np.round(Dr) * pbc, cell)
            D_len = np.sqrt((D**2).sum(1))

            cutoff = min(max(D_len), max(diags) / 2.)

    latt_len = np.sqrt((cell**2).sum(1))
    V = abs(np.linalg.det(cell))
    padding = pbc * np.array(np.ceil(cutoff * np.prod(latt_len) /
                                     (V * latt_len)), dtype=int)

    offsets = np.mgrid[-padding[0]:padding[0] + 1,
                       -padding[1]:padding[1] + 1,
                       -padding[2]:padding[2] + 1].T
    tvecs = np.dot(offsets, cell)
    coords = pos[None, None, None, :, :] + tvecs[:, :, :, None, :]

    ncell = np.prod(offsets.shape[:-1])
    index = np.arange(len(atoms))[None, :].repeat(ncell, axis=0).flatten()
    coords = coords.reshape(np.prod(coords.shape[:-1]), 3)
    offsets = offsets.reshape(ncell, 3)

    return index, coords, offsets


def get_mic(p1, p2, cell, pbc=[1,1,0], max_cell_multiples=1e5, 
            get_squared_distance=False): 
    """
    Get all vectors from p1 to p2 that are less than the cutoff in length
    Also able to calculate the distance using the minimum image convention

    This function is useful when you want to constantly calculate mic between 
    two given positions. Please use ase.geometry.find_mic if you want to 
    calculate an array of vectors all at a time (useful for e.g. neighborlist).    
    
    :param p1:
    :param p2:
    :param cutoff:
    :param max_cell_multiples:
    :return:
    """
    # Precompute some useful values
    a, b, c = cell[0], cell[1], cell[2]
    vol = np.abs(a @ np.cross(b, c))
    a_cross_b = np.cross(a, b)
    a_cross_b_len = np.linalg.norm(a_cross_b)
    a_cross_b_hat = a_cross_b / a_cross_b_len
    b_cross_c = np.cross(b, c)
    b_cross_c_len = np.linalg.norm(b_cross_c)
    b_cross_c_hat = b_cross_c / b_cross_c_len
    a_cross_c = np.cross(a, c)
    a_cross_c_len = np.linalg.norm(a_cross_c)
    a_cross_c_hat = a_cross_c / a_cross_c_len

    # TODO: Wrap p1, and p2 into the current unit cell
    dr = p2 - p1
    min_dr_sq = dr @ dr
    min_length = math.sqrt(min_dr_sq)
    a_max = math.ceil(min_length / vol * b_cross_c_len)
    a_max = min(a_max, max_cell_multiples)
    b_max = math.ceil(min_length / vol * a_cross_c_len)
    b_max = min(b_max, max_cell_multiples)
    if not pbc[2]:
        c_max = 0
    else:
        c_max = math.ceil(min_length / vol * a_cross_b_len)
        c_max = min(c_max, max_cell_multiples)

    min_dr = dr
    for i in range(-a_max, a_max + 1):
        ra = i * a
        for j in range(-b_max, b_max + 1):
            rab = ra + j * b
            for k in range(-c_max, c_max + 1):
                if i == 0 and j == 0 and k == 0:
                    continue
                out_vec = rab + k * c + dr
                len_sq = out_vec @ out_vec 
                if len_sq < min_dr_sq:
                    min_dr = out_vec
                    min_dr_sq = len_sq
    if not get_squared_distance:
        return min_dr

    else:
        return np.sum(min_dr**2)


def draw_graph(G, savefig=None):               
    import matplotlib.pyplot as plt
    labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, labels=labels, font_size=8)
    if savefig:
        plt.savefig(savefig)
    plt.show() 
