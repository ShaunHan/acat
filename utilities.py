from ase.data import covalent_radii
from ase.geometry import find_mic
from collections import defaultdict
import numpy as np
import scipy

def neighbor_shell_list(atoms, dx=0.3, neighbor_number=1, 
                        different_species=False, mic=False):
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
        Whether apply minimum image convention or not
    """

    assert True in atoms.pbc    
    atoms = atoms.copy()
    cell = atoms.cell
    pbc = atoms.pbc
    conn = {k: [] for k in range(len(atoms))}
    for atomi in atoms:
        for atomj in atoms:
            if atomi.index != atomj.index:
                if not (different_species & (atomi.symbol == atomj.symbol)):
                    if mic:
                        d = get_mic_distance(atomi.position,
                                             atomj.position,
                                             cell, pbc)
                    else:
                        d = np.linalg.norm(atomi.position - atomj.position)

                    cri = covalent_radii[atomi.number]
                    crj = covalent_radii[atomj.number]
                    if neighbor_number == 1:
                        d_max1 = 0.
                    else:
                        d_max1 = ((neighbor_number - 1) * (crj + cri)) + dx
                    d_max2 = (neighbor_number * (crj + cri)) + dx
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


def get_mic_distance(p1, p2, cell, pbc=True):
    '''Calculate the distance using the minimum image convention'''

    return find_mic(np.asarray([p1]) - np.asarray([p2]), 
                    cell, pbc)[1][0]


def point_projection_on_line_old(point, position, vec):         
    '''Calculate the position of a point projection on a line'''
    ap = point - position
    t = np.dot(ap, vec) / np.dot(vec, vec)
    projection = position + vec * t

    return projection

def point_projection_on_line(point, position, vec, h):         
    '''Calculate the position of a point projection on a line'''
    ap = point - position
    b = position + vec * h
    ab = b - position
    t = np.dot(ap, ab) / np.dot(ab, ab)
    projection = position + ab * t

    return projection

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
        diags = np.sqrt((
            np.dot([[1, 1, 1],
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

    offsets = np.mgrid[
        -padding[0]:padding[0] + 1,
        -padding[1]:padding[1] + 1,
        -padding[2]:padding[2] + 1].T
    tvecs = np.dot(offsets, cell)
    coords = pos[None, None, None, :, :] + tvecs[:, :, :, None, :]

    ncell = np.prod(offsets.shape[:-1])
    index = np.arange(len(atoms))[None, :].repeat(ncell, axis=0).flatten()
    coords = coords.reshape(np.prod(coords.shape[:-1]), 3)
    offsets = offsets.reshape(ncell, 3)

    return index, coords, offsets
