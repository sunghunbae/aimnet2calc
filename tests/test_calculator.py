import pytest
import ase.io
import os
import numpy as np

from collections.abc import Callable
from aimnet2calc import AIMNet2Calculator


DIR = os.path.dirname(__file__)


def _list_input():
    filename = os.path.join(DIR, 'mols_size_var.xyz')
    atoms = ase.io.read(filename, index=':')
    ret = dict()
    ret['coord'] = np.concatenate([a.positions for a in atoms])
    ret['numbers'] = np.concatenate([a.numbers for a in atoms])
    ret['mol_idx'] = np.concatenate([[i] * len(a) for i, a in enumerate(atoms)])
    ret['charge'] = [0.0] * len(atoms)
    return ret


def _batch_input():
    filename = os.path.join(DIR, 'mols_size_36.xyz')
    atoms = ase.io.read(filename, index=':')
    ret = dict()
    ret['coord'] = np.array([a.positions for a in atoms])
    ret['numbers'] = np.array([a.numbers for a in atoms])
    ret['charge'] = np.array([0.0] * len(atoms))
    return ret


def _test_energy(calc:Callable, data:dict) -> str:
    _out = calc(data)
    assert 'energy' in _out
    assert len(_out['energy']) == len(data['charge'])
    assert _out['energy'].requires_grad == False
    return 'success'


def _test_forces(calc:Callable, data:dict) -> str:
    _out = calc(data, forces=True)
    assert 'energy' in _out
    assert 'forces' in _out
    assert len(_out['energy']) == len(data['charge'])
    assert _out['energy'].requires_grad == True
    assert len(_out['forces']) == len(data['coord']), _out['forces'].shape
    assert _out['forces'].requires_grad == False
    return 'success'


def test_aimnet2_batch():
    calc = AIMNet2Calculator('aimnet2')
    data = _batch_input()
    print('energy: ', _test_energy(calc, data))
    print('forces: ', _test_forces(calc, data))


def test_aimnet2_list():
    calc = AIMNet2Calculator('aimnet2')
    data = _list_input()
    print('energy: ', _test_energy(calc, data))
    print('forces: ', _test_forces(calc, data))


def test_aimnet2_b973c_batch():
    calc = AIMNet2Calculator('aimnet2_b973c')
    data = _batch_input()
    print('energy: ', _test_energy(calc, data))
    print('forces: ', _test_forces(calc, data))


def test_aimnet2_b973c_list():
    calc = AIMNet2Calculator('aimnet2_b973c')
    data = _list_input()
    print('energy: ', _test_energy(calc, data))
    print('forces: ', _test_forces(calc, data))
