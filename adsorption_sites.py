from ase.io import read, write
from ase.build import molecule
from ase.formula import Formula


# Adsorbate elements must be different from catalyst elements
adsorbate_elements = 'SCHON'

# Adsorbate height on different sites
site_heights = {'ontop': 1.7, 
                'bridge': 1.7, 
                'short-bridge': 1.7,
                'long-bridge': 1.7,
                'fcc': 1.6, 
                'hcp': 1.6,
                '3fold': 1.6, 
                '4fold': 1.6,
                '5fold': 1.6,
                '6fold': 0.,}

# Make your own adsorbate list. Make sure you always sort the 
# indices of the atoms in the same order as the symbol. 
# First element always starts from bonded index or the 
# bonded element with smaller atomic number if multi-dentate.
                             # Monodentate (vertical)
monodentate_adsorbate_list = ['H','C','O','CH','OH','CO','CH2','OH2','COH','CH3','OCH','OCH2','OCH3',]
                              # Multidentate (lateral)
multidentate_adsorbate_list = ['CHO','CHOH','CH2O','CH3O','CH2OH','CH3OH','CHOOH','COOH','CHOO','CO2',]

adsorbate_list = monodentate_adsorbate_list + multidentate_adsorbate_list
adsorbate_formulas = {k: ''.join(list(Formula(k))) for k in adsorbate_list}

# Make your own bidentate fragment dict
adsorbate_fragments = {'CO': ['C','O'],     # Possible
                       'OC': ['O','C'],     # to 
                       'COH': ['C','OH'],   # bidentate
                       'OCH': ['O','CH'],   # on
                       'OCH2': ['O','CH2'], # rugged 
                       'OCH3': ['O','CH3'], # surfaces
                       'CHO': ['CH','O'],
                       'CHOH': ['CH','OH'],
                       'CH2O': ['CH2','O'],
                       'CH3O': ['CH3','O'],
                       'CH2OH': ['CH2','OH'],
                       'CH3OH': ['CH3','OH'],
                       'CHOOH': ['CH','O','OH'],
                       'COOH': ['C','O','OH'],
                       'CHOO': ['CH','O','O'],
                       'CO2': ['C','O','O'],}

# Make your own adsorbate molecules
def adsorbate_molecule(adsorbate):
    # The ase.build.molecule module has many issues.       
    # Adjust positions, angles and indexing for your needs.
    if adsorbate  == 'CO':
        ads = molecule(adsorbate)[::-1]
    elif adsorbate == 'OH2':
        ads = molecule('H2O')
        ads.rotate(180, 'y')
    elif adsorbate == 'CH2':
        ads = molecule('NH2')
        ads[0].symbol = 'C'
        ads.rotate(180, 'y')
    elif adsorbate == 'COH':
        ads = molecule('H2COH')
        del ads[-2:]
        ads.rotate(90, 'y')
    elif adsorbate == 'CHO':
        ads = molecule('HCO')[[0,2,1]] 
    elif adsorbate == 'OCH2':
        ads = molecule('H2CO')
        ads.rotate(180, 'y')
    elif adsorbate == 'OCH3':
        ads = molecule('CH3O')[[1,0,2,3,4]]
        ads.rotate(90, '-x')
    elif adsorbate == 'CH2O':
        ads = molecule('H2CO')[[1,2,3,0]]
        ads.rotate(90, 'y')
    elif adsorbate == 'CH3O':
        ads = molecule('CH3O')[[0,2,3,4,1]]
        ads.rotate(30, 'y')
    elif adsorbate == 'CHOH':
        ads = molecule('H2COH')
        del ads[-1]
        ads = ads[[0,3,1,2]]
    elif adsorbate == 'CH2OH':
        ads = molecule('H2COH')[[0,3,4,1,2]]
    elif adsorbate == 'CH3OH':
        ads = molecule('CH3OH')[[0,2,4,5,1,3]]
        ads.rotate(-30, 'y')
    elif adsorbate == 'CHOOH':
        ads = molecule('HCOOH')[[1,4,2,0,3]]
    elif adsorbate == 'COOH':
        ads = molecule('HCOOH')
        del ads[-1]
        ads = ads[[1,2,0,3]]
        ads.rotate(90, '-x')
        ads.rotate(15, '-y')
    elif adsorbate == 'CHOO':
        ads = molecule('HCOOH')
        del ads[-2]
        ads = ads[[1,3,2,0]]
        ads.rotate(90, 'x')
        ads.rotate(15, 'y')
    elif adsorbate == 'CO2':
        ads = molecule(adsorbate)
        ads.rotate(-90, 'y')
    else:
        try:
            ads = molecule(adsorbate)
        except:
            print('Molecule {} does not exist in the databse'.format(adsorbate))
            return 
    return ads

# Dictionaries for constructing fingerprints of machine learning models
# Norskov
dband_centers = {'H': 0., 'C': 0., 'O': 0., 'Ni': -1.29, 'Pt': -2.25,}

# Wiki
electron_affinities = {'H': 0.754, 'C': 1.262, 'O': 1.461, 'Ni': 1.157, 'Pt': 2.125,}

# http://srdata.nist.gov/cccbdb/
ionization_energies = {'H': 13.60, 'C': 11.26, 'O': 13.62, 'Ni': 7.64, 'Pt': 8.96,}

# https://www.tandfonline.com/doi/full/10.1080/00268976.2018.1535143
dipole_polarizabilities = {'H': 4.5, 'C': 11.3, 'O': 5.3, 'Ni': 49.0, 'Pt': 48.0,}

# https://www.lenntech.com/periodic-chart-elements/electronegativity.htm
electronegativities = {'H': 2.2, 'C': 2.55, 'O': 3.44, 'Ni': 1.9, 'Pt': 2.2,}

