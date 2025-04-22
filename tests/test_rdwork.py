import pytest
import ase.io
import os
import numpy as np
import rdwork

from rdwork import MolLibr
from collections.abc import Callable
from aimnet2calc import AIMNet2Calculator

#CODATA 2018 energy conversion factor
hartree2ev = 27.211386245988
hartree2kcalpermol = 627.50947337481
ev2kcalpermol = 23.060547830619026

# Lahey, S.-L. J., Thien Phuc, T. N. & Rowley, C. N.
# Benchmarking Force Field and the ANI Neural Network Potentials for the
# Torsional Potential Energy Surface of Biaryl Drug Fragments.
# J. Chem. Inf. Model. 60, 6258–6268 (2020)

torsion_dataset_smiles = [
    "C1(C2=CC=CN2)=CC=CC=C1",
    "C1(C2=NC=CN2)=CC=CC=C1",
    "C1(N2C=CC=C2)=NC=CC=N1",
    "C1(C2=NC=NC=N2)=CC=CC=C1",
    "C1(N2C=CC=C2)=CC=CC=C1",
    "O=C(N1)C=CC=C1C2=COC=C2",
    "C1(C2=NC=CC=N2)=NC=CC=N1",
    "O=C(N1)C=CC=C1C2=NC=CN2",
    ]

torsion_dataset_names=["07", "09","20", "39", "10", "23", "12", "29"]

# def _struct_list():
#     filename = os.path.join(DIR, 'mols_size_var.xyz')
#     atoms = ase.io.read(filename, index=':')
#     ret = dict()
#     ret['coord'] = np.concatenate([a.positions for a in atoms])
#     ret['numbers'] = np.concatenate([a.numbers for a in atoms])
#     ret['mol_idx'] = np.concatenate([[i] * len(a) for i, a in enumerate(atoms)])
#     ret['charge'] = [0.0] * len(atoms)
#     return ret


# def _stuct_batch():
#     filename = os.path.join(DIR, 'mols_size_36.xyz')
#     atoms = ase.io.read(filename, index=':') # read all
#     ret = dict()
#     ret['coord'] = [a.positions for a in atoms]
#     ret['numbers'] = [a.numbers for a in atoms]
#     ret['charge'] = [0.0] * len(atoms)
#     return ret


def _test_energy(calc:Callable, data:dict) -> str:
    _out = calc(data)
    assert 'energy' in _out
    assert len(_out['energy']) == len(data['charge'])
    assert _out['energy'].requires_grad == False
    print(_out)
    return 'success'


def _test_forces(calc:Callable, data:dict) -> str:
    _out = calc(data, forces=True)
    assert 'energy' in _out
    assert 'forces' in _out
    assert len(_out['energy']) == len(data['charge'])
    assert _out['energy'].requires_grad == True
    assert len(_out['forces']) == len(data['coord']), _out['forces'].shape
    assert _out['forces'].requires_grad == False
    print(_out)
    return 'success'


def libr_to_batches(libr:MolLibr, batchsize:int=1000) -> list:
    """Split workload flexibily into a numer of batches.
    
    - Each batch has up to `batchsize` number of atoms. 
    - Conformers originated from a same molecule can be splitted into multiple batches.
    - Or one batch can contain conformers originated from multiple molecules.

    coord: coordinates of input molecules (N, m, 3) where N is the number of structures and 
    m is the number of atoms in each structure.
    numbers: atomic numbers in the molecule (include H). (N, m)
    charges: (N,)

    Args:
        settings (dict): dictionary of settings.

    Returns:
        list: list of batches.
    """
        
    pre_batches = []
    batch_confs = []
    batch_mols = []
    batch_n_atoms = 0
    for mol in libr:
        for conf in mol.confs:
            n_atoms = conf.props['atoms']
            if (batch_n_atoms + n_atoms) > batchsize:
                pre_batches.append((batch_mols, batch_confs, batch_n_atoms))
                # start over a new batch
                batch_mols =  [mol]
                batch_confs = [conf]
                batch_n_atoms = n_atoms
            else:
                batch_mols.append(mol)
                batch_confs.append(conf)
                batch_n_atoms += n_atoms
    if batch_n_atoms > 0: # last remaining batch
        pre_batches.append((batch_mols, batch_confs, batch_n_atoms))
    batches = []
    for i, (batch_mols, batch_confs, batch_n_atoms) in enumerate(pre_batches, start=1):
        charges = [mol.props['charge'] for mol in batch_mols]
        coord = [conf.rdmol.GetConformer().GetPositions().tolist() for conf in batch_confs]
        # to be consistent with legacy code
        coord = [[tuple(xyz) for xyz in inner] for inner in coord]
        # numbers should be got from conformers because of hydrogens
        numbers = [[a.GetAtomicNum() for a in conf.rdmol.GetAtoms()] for conf in batch_confs]
        print(f"batch {i:3d} <- {len(batch_confs):4d} conformers {batch_n_atoms:6d} atoms")
        batches.append((coord, numbers, charges, batch_confs, batch_mols))
    return batches


def padding_coords(lists, pad_value=0.0):
    lengths = [len(lst) for lst in lists]
    max_length = max(lengths)
    pad_length = [max_length - len(lst) for lst in lists]
    assert (len(pad_length) == len(lists))
    lists_padded = []
    for i in range(len(pad_length)):
        lst_i = lists[i]
        pad_i = [(pad_value, pad_value, pad_value) for _ in range(pad_length[i])]
        lst_i_padded = lst_i + pad_i
        lists_padded.append(lst_i_padded)
    return np.array(lists_padded)


def padding_numbers(lists, pad_value=0):
    lengths = [len(lst) for lst in lists]
    max_length = max(lengths)
    pad_length = [max_length - len(lst) for lst in lists]
    assert (len(pad_length) == len(lists))
    lists_padded = []
    for i in range(len(pad_length)):
        lst_i = lists[i]
        pad_i = [pad_value for _ in range(pad_length[i])]
        lst_i_padded = lst_i + pad_i
        lists_padded.append(lst_i_padded)
    return np.array(lists_padded)


def test_rdwork():
    libr = MolLibr(torsion_dataset_smiles, torsion_dataset_names)
    libr = libr.make_confs(n_rel=1.0, progress=True)
    libr = libr.drop_confs(similar=True).rename()
    batches = libr_to_batches(libr)
    print(libr.count(), "molecules")
    print("batches", batches)

    for model in ['aimnet2', 'aimnet2_b973c']:
        print('Testing model:', model)
        calc = AIMNet2Calculator(model)
        for (coord, numbers, charges, batch_confs, batch_mols) in batches:
            data = {
                'coord'  : padding_coords(coord),
                'numbers': padding_numbers(numbers),
                'charge' : np.array(charges),
                }
            print('energy: ', _test_energy(calc, data))
            print('forces: ', _test_forces(calc, data))


def test_ase_optimize():
    from aimnet2calc import AIMNet2ASE
    from ase import Atoms
    from ase.optimize import BFGS, FIRE
    import io
    from types import SimpleNamespace

    libr = rdwork.read_sdf("./cyclooctane.sdf", confs=True)
    mol = libr[0]
    conformer = mol.confs[0]
    ase_atoms = ase.Atoms(symbols = conformer.symbols(),
                          positions = conformer.positions())
    
    print(conformer.numbers())

    ase_atoms.calc = AIMNet2ASE('aimnet2')
    with io.StringIO() as output:
        FIRE(ase_atoms, logfile=output).run(fmax=0.05)
        lines = [l.strip().split()[1:] for l in output.getvalue().split('\n') if l.startswith('FIRE')]
        opt_data = [SimpleNamespace(step=int(step), energy=float(energy), fmax=float(fmax)) for (step, time, energy, fmax) in lines]
            
    print(ase_atoms.positions)
    print(opt_data)
    print(opt_data[-1].energy - opt_data[0].energy)
    print(opt_data[-1].fmax < 0.05)


def test_ase_torsion():
    import ase
    from aimnet2calc import AIMNet2ASE
    from ase.optimize import FIRE
    from rdkit.Chem import AllChem
    import io
    import types
    import itertools
    
    # Lahey, S.-L. J., Thien Phuc, T. N. & Rowley, C. N. 
    # Benchmarking Force Field and the ANI Neural Network Potentials for the 
    # Torsional Potential Energy Surface of Biaryl Drug Fragments. 
    # J. Chem. Inf. Model. 60, 6258–6268 (2020)

    torsion_dataset_smiles = [
        "C1(C2=CC=CN2)=CC=CC=C1",
        "C1(C2=NC=CN2)=CC=CC=C1",
        "C1(N2C=CC=C2)=NC=CC=N1",
        "C1(C2=NC=NC=N2)=CC=CC=C1",
        "C1(N2C=CC=C2)=CC=CC=C1",
        "O=C(N1)C=CC=C1C2=COC=C2",
        "C1(C2=NC=CC=N2)=NC=CC=N1",
        "O=C(N1)C=CC=C1C2=NC=CN2",
        ]

    torsion_dataset_names = ["07", "09", "20", "39", "10", "23", "12", "29"]

    libr = rdwork.MolLibr(torsion_dataset_smiles, torsion_dataset_names)
    libr = libr.make_confs(n=50, progress=False)

    use_converged_only = True
    interval = 15.0
    target_fmax = 0.05

    mol = libr[0].copy()
    torsion_angles = mol.torsion_atoms()
    print(f"number of torsion angles= {len(torsion_angles)}")

    conf = mol.confs[0].copy()
    mol.confs = [] # overwrite mol.confs with torsion conformers

    torsion_data = []
    for k, indices in enumerate(torsion_angles):
        torsion_data.append({'angle':[], 'init(eV)':[], 'last(eV)':[], 'Converged':[]})
        (a, b, c, d, rot_indices, fix_indices) = indices
        # -180., ..., (180) i.e the last 180 is not included in the list
        for angle in np.arange(-180.0, 180.0, interval): # numpy.ndarray
            x = conf.copy()
            x.props.update({'index': k, 'angle': angle})
            AllChem.SetDihedralDeg(x.rdmol.GetConformer(), a, b, c, d, angle)
            # all atoms bonded to atom d are moved
            mol.confs.append(x)
    
    mol.props['torsion_angles'] = torsion_angles
    mol.props['torsion_data'] = torsion_data

    for conf in mol.confs:
        ase_atoms = ase.Atoms(symbols = conf.symbols(), positions = conf.positions())
        ase_atoms.calc = AIMNet2ASE('aimnet2')
        with io.StringIO() as logfile:
            FIRE(ase_atoms, logfile=logfile).run(fmax=target_fmax)
            lines = [l.strip().split()[1:] for l in logfile.getvalue().split('\n') if l.startswith('FIRE')]
            data = [types.SimpleNamespace(energy=float(e), fmax=float(f)) for (_, _, e, f) in lines]
            conf = conf.sync(ase_atoms.get_positions())
            datadict = mol.props['torsion_data'][conf.props['index']]
            datadict['angle'].append(conf.props['angle'])
            datadict['init(eV)'].append(data[0].energy)
            datadict['last(eV)'].append(data[-1].energy)
            datadict['Converged'].append(data[-1].fmax < target_fmax)

    torsion = []
    for indices, datadict in zip(mol.props['torsion_angles'], mol.props['torsion_data']):
        if use_converged_only:
            datadict['angle'] = list(itertools.compress(datadict['angle'], datadict['Converged']))
            datadict['init(eV)'] = list(itertools.compress(datadict['init(eV)'], datadict['Converged']))
            datadict['last(eV)'] = list(itertools.compress(datadict['last(eV)'], datadict['Converged']))
    
        datadict['E_loss(eV)'] = np.array(datadict['init(eV)']) - np.median(datadict['last(eV)'])
    
        b = min(datadict['E_loss(eV)'])
    
        datadict['E_rel(kcal/mol)'] = [ev2kcalpermol*(x-b) for x in datadict['E_loss(eV)']]
    
        torsion.append({
            'indices': indices,
            'angle': datadict['angle'],
            'E_rel(kcal/mol)': datadict['E_rel(kcal/mol)'],
            })
    
    mol.props.update({
        'torsion': torsion,
        'torsion method': 'aimnet2',
        })
    
    del mol.props['torsion_angles'] # this info. is included in 'torsion'
    del mol.props['torsion_data'] # this info. is included in 'torsion'

    print(mol.props['torsion'])



def test_batch_energies():
    libr = rdwork.read_sdf("./cyclooctane.sdf", confs=True)
    libr += libr.copy()

    batches = libr_to_batches(libr)
    print(libr.count(), "molecules")
    print("batches", batches)

    e_ref = -314.689736079491 * hartree2ev

    for model in ['aimnet2', 'aimnet2_b973c']:
        print('Testing model:', model)
        calc = AIMNet2Calculator(model)
        for (coord, numbers, charges, batch_confs, batch_mols) in batches:
            data = {
                'coord'  : padding_coords(coord),
                'numbers': padding_numbers(numbers),
                'charge' : np.array(charges),
                }
            print('energy: ', _test_energy(calc, data))
            print('forces: ', _test_forces(calc, data))


if __name__ == '__main__':
    # test_batch_energies()
    # test_ase_optimize()
    test_ase_torsion()
