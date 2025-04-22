import pytest
import numpy as np

from ase import Atoms
from ase.optimize import BFGS, FIRE
from ase.calculators.emt import EMT

from aimnet2calc import AIMNet2ASE

def _generate_water():
    d = 0.9575
    t = np.pi / 180 * 104.51
    water = Atoms('H2O', 
                  positions=[(d, 0, 0), (d * np.cos(t), d * np.sin(t), 0), (0, 0, 0)])
    return water

def test_ase_bfgs():
    water = _generate_water()
    water.calc = EMT()
    dyn = BFGS(water)
    dyn.run(fmax=0.05)
    print(water.positions)


def test_aimnet2_bfgs():
    water = _generate_water()
    water.calc = AIMNet2ASE('aimnet2')
    dyn = BFGS(water)
    dyn.run(fmax=0.05)
    print(water.positions)


def test_aimnet2_fire():
    water = _generate_water()
    water.calc = AIMNet2ASE('aimnet2')
    dyn = FIRE(water)
    dyn.run(fmax=0.05)
    print(water.positions)

if __name__ == '__main__':
    test_ase_bfgs()
    test_aimnet2_bfgs()
    test_aimnet2_fire()