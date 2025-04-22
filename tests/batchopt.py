# https://github.com/isayevlab/Auto3D_pkg
# Modified from Auto3D_pkg/src/Auto3D/batch_opt/batchopt.py
# Original source: /labspace/models/aimnet/batch_opt_script/

try:
    import torch
    import torch.nn as nn
    torch.manual_seed(1993)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except:
    raise ImportError('torch is required')

try:
    import torchani
except:
    raise ImportError('torchani is required')

import numpy as np

from collections import defaultdict
from typing import List, Tuple


#CODATA 2018 energy conversion factor
hartree2ev = 27.211386245988
hartree2kcalpermol = 627.50947337481
ev2kcalpermol = 23.060547830619026


@torch.jit.script
class FIRE():
    """a general optimization program """
    # For a list of documentation for different optimization programs: 
    # https://wiki.fysik.dtu.dk/ase/ase/optimize.html
    def __init__(self, coord):
        ## default parameters
        self.dt_max = 0.1
        self.Nmin = 5
        self.maxstep = 0.1
        self.finc = 1.5
        self.fdec = 0.7
        self.astart = 0.1
        self.fa = 0.99
        self.v = torch.zeros_like(coord)
        self.Nsteps = torch.zeros(coord.shape[0], dtype=torch.long, device=coord.device)
        self.dt = torch.full(coord.shape[:1], 0.1, device=coord.device)
        self.a = torch.full(coord.shape[:1], 0.1, device=coord.device)


    def __call__(self, coord, forces):
        """Moving atoms based on forces
        
        Arguments:
            coord: coordinates of atoms. Size (Batch, N, 3), where Batch is
                   the number of structures, N is the number of atom in each structure.
            forces: forces on each atom. Size (Batch, N, 3).
            
        Return:
            new coordinates that are moved based on input forces. Size (Batch, N, 3)"""
        vf = (forces * self.v).flatten(-2, -1).sum(-1)
        w_vf = vf > 0.0
        if w_vf.all():
            a = self.a.unsqueeze(-1).unsqueeze(-1)
            v = self.v
            f = forces
            self.v = (1.0 - a) * v + a * v.flatten(-2, -1).norm(p=2, dim=-1).unsqueeze(
                -1).unsqueeze(-1) * f / f.flatten(-2, -1).norm(p=2, dim=-1).unsqueeze(-1).unsqueeze(
                -1)
            self.Nsteps += 1
        elif w_vf.any():
            a = self.a[w_vf].unsqueeze(-1).unsqueeze(-1)
            v = self.v[w_vf]
            f = forces[w_vf]
            self.v[w_vf] = (1.0 - a) * v + a * v.flatten(-2, -1).norm(p=2, dim=-1).unsqueeze(
                -1).unsqueeze(-1) * f / f.flatten(-2, -1).norm(p=2, dim=-1).unsqueeze(-1).unsqueeze(
                -1)

            w_N = self.Nsteps > self.Nmin
            w_vfN = w_vf & w_N
            self.dt[w_vfN] = (self.dt[w_vfN] * self.finc).clamp(max=self.dt_max)
            self.a[w_vfN] *= self.fa
            self.Nsteps[w_vfN] += 1

        w_vf = ~w_vf
        if w_vf.all():
            self.v[:] = 0.0
            self.a[:] = torch.tensor(self.astart, device=self.a.device)
            self.dt[:] *= self.fdec
            self.Nsteps[:] = 0
        elif w_vf.any():
            self.v[w_vf] = torch.tensor(0.0, device=self.v.device)
            self.a[w_vf] = torch.tensor(self.astart, device=self.a.device)
            self.dt[w_vf] *= self.fdec
            self.Nsteps[w_vf] = torch.tensor(0, device=self.v.device)

        dt = self.dt.unsqueeze(-1).unsqueeze(-1)
        self.v += dt * forces
        dr = dt * self.v
        normdr = dr.flatten(-2, -1).norm(p=2, dim=-1).unsqueeze(-1).unsqueeze(-1)
        dr *= (self.maxstep / normdr).clamp(max=1.0)
        return coord + dr


    def clean(self, mask):
        # types: (Tensor) -> bool
        self.v = self.v[mask]
        self.Nsteps = self.Nsteps[mask]
        self.dt = self.dt[mask]
        self.a = self.a[mask]
        return True


class EnForce_ANI(torch.nn.Module):
    """Takes in an torch model, then defines two forward functions for it.
    The input model should be able to calculate energy and disp_energy given
    coordiantes, species and charges of a molecule. 

    Arguments:
        ani: model
        batchsize: the maximum nmber atoms that can be handled in one batch.

    Returns:
        the energies and forces for the input molecules. One time calculation.
    """

    def __init__(self, ani, name, batchsize=16*1024):
        super().__init__()
        self.add_module('ani', ani)
        self.name = name
        self.batchsize = batchsize


    def forward(self, coord, numbers, charges):
        """Calculate the energies and forces for input molecules. Called by self.forward_batched
        
        Arguments:
            coord: coordinates for all input structures. size (B, N, 3), where
                  B is the number of structures in coord, N is the number of
                  atoms in each structure, 3 represents xyz dimensions.
            numbers: the periodic numbers for all atoms.
            charges: tensor size (B)
            
        Returns:
            energies
            forces
        """
        # charge = torch.zeros_like(numbers[:, 0])
        # torch.set_grad_enabled(True)
        # torch._C._set_grad_enabled(True)
        d = self.ani(dict(coord=coord, numbers=numbers, charge=charges))  # Output from the model
        # e = (d['energy'] + d['disp_energy']).to(torch.double)
        e = d['energy'].to(torch.double)
        # g = torch.autograd.grad([e.sum()], [coord])[0]  # size(100, 23, 3)
        # assert g is not None
        # f = -g
        f = d['forces']

        return e, f


    #    @torch.jit.script_method
    def forward_batched(self, coord, numbers, charges):
        """Calculate the energies and forces for input molecules.
        
        Arguments:
            coord: coordinates for all input structures. size (B, N, 3), where
                  B is the number of structures in coord, N is the number of
                  atoms in each structure, 3 represents xyz dimensions.
            numbers: the periodic numbers for all atoms. size (B, N)
            
        Returns:
            energies
            forces
        """
        B, N = coord.shape[:2]
        e = []
        f = []
        idx = torch.arange(B, device=coord.device)
        for batch in idx.split(self.batchsize // N):  # How was the batchsize decided?
            _e, _f = self(coord[batch], numbers[batch], charges[batch])
            e.append(_e)
            f.append(_f)
        return torch.cat(e, dim=0), torch.cat(f, dim=0)


    
class BatchOptimizer:
    def __init__(self, coord:list, numbers:list, charges:list, config:dict) -> None:
        self.coord = None
        self.numbers = None
        self.charges = charges
        self.model = config['model']
        self.model_path = config['model_path']
        self.device = config['device']
        self.config = config
        if  self.model == "AIMNET":
            # self.ani = torch.jit.load(model_path / "aimnet2_wb97m-d3_0.jpt", map_location=device)
            self.ani = torch.jit.load(self.model_path / "aimnet2_wb97m_ens_f.jpt", map_location=self.device)
        elif self.model == "ANI-2xt":
            self.ani = ANI2xt(self.device, self.model_path / "ani2xt_no_repulsion.pt")
        elif self.model == "ANI-2x":
            self.ani = torchani.models.ANI2x(periodic_table_index=True).to(self.device)
        else:
            raise ValueError("BatchOptimizer supports ANI-2x, ANI-2xt, and AIMNET.")
        # coord: coordinates of input molecules (N, m, 3). 
        #     N is the number of structures
        #     m is the number of atoms in each structure.
        # numbers: atomic numbers in the molecule (include H). (N, m)
        # charges: (N,)
        if self.model == "AIMNET":
            self.coord = BatchOptimizer.padding_coords(coord, 0)
            self.numbers = BatchOptimizer.padding_species(numbers, 0)
        else:
            self.coord = BatchOptimizer.padding_coords(coord, 0)
            self.numbers = BatchOptimizer.padding_species(numbers, -1)


    @staticmethod
    def n_steps(state, n, opt_tol, opt_patience):
        """Doing n steps optimization for each input. Only converged structures are 
        modified at each step. n_steps does not change input conformer order.
        
        Argument:
            state: an dictionary containing all information about this optimization step
            n: optimization step
            opt_patience: optimization stops for a conformer if the force does not decrease for a continuous opt_patience steps
        """
        # t0 = perf_counter()
        numbers = state['numbers']
        charges = state['charges']
        # num_total = numbers.size()[0]
        coord = state['coord']
        optimizer = FIRE(coord)
        # the following two terms are used to detect oscillating conformers
        smallest_fmax0 = torch.tensor(np.ones((len(coord), 1)) * 999, dtype=torch.float).to(coord.device)
        oscilating_count0 = torch.tensor(np.zeros((len(coord), 1)), dtype=torch.float).to(coord.device)
        state["oscilating_count"] = oscilating_count0
        assert (len(coord.shape) == 3)
        assert (len(numbers.shape) == 2)
        assert (len(charges.shape) == 1)
        assert (len(smallest_fmax0.shape) == 2)
        assert (len(oscilating_count0.shape) == 2)
        for istep in range(1, (n + 1), 1):
            not_converged = ~ state['converged_mask']  # Essential tracker handle, size fixed
            # stop optimization if all structures converged.
            if not not_converged.any():
                break
            coord = state['coord'][not_converged]  # Subset coordinates, size=not_converged.
            numbers = state['numbers'][not_converged]
            charges = state['charges'][not_converged]
            smallest_fmax = smallest_fmax0[not_converged]
            oscilating_count = state["oscilating_count"][not_converged]
            
            coord.requires_grad_(True)
            e, f = state['nn'].forward_batched(coord, numbers, charges)
            # Key step to calculate all energies and forces.
            if istep == 1:
                # record energy before optimization
                # .detach() method in PyTorch is used to separate a tensor from
                # the computational graph by returning a new tensor that doesn't
                # require a gradient.
                state['init_energy'][not_converged] = e.detach()
            coord.requires_grad_(False)
            coord = optimizer(coord, f) # optimized coord

            # Tensor, Norm is the length of each vector.
            # Here it returns the maximum force length for each conformer. Size (100)
            fmax = f.norm(dim=-1).max(dim=-1)[0]
            assert (len(fmax.shape) == 1)
            not_converged_post1 = fmax > opt_tol
            # update smallest_fmax for each molecule
            fmax_reduced = fmax.reshape(-1, 1) < smallest_fmax
            fmax_reduced = fmax_reduced.reshape(-1, )
            smallest_fmax[fmax_reduced] = fmax.reshape(-1, 1)[fmax_reduced]
            # reduce count to 0 for reducing; raise count for non-reducing
            oscilating_count[fmax_reduced] = 0
            fmax_not_reduced = ~fmax_reduced
            oscilating_count += fmax_not_reduced.reshape(-1, 1)
            not_oscilating = oscilating_count < opt_patience
            not_oscilating = not_oscilating.reshape(-1, )
            not_converged_post = not_converged_post1 & not_oscilating
            # Subset v, a in FIRE for next optimization
            optimizer.clean(not_converged_post)
            
            # Update converged_mask, so that converged structures will not be updated in future steps.
            state['converged_mask'][not_converged] = ~ not_converged_post
            # Update fmax for conformers that are optimized in this iteration
            state['fmax'][not_converged] = fmax
            # Update energy for conformers that are optimized in this iteration
            state['energy'][not_converged] = e.detach()
            # Update coordinates for conformers that are optimized in this iteration
            state['coord'][not_converged] = coord
            # update smalles_fmax for each conformer
            smallest_fmax0[not_converged] = smallest_fmax
            # update counts for continuous no reduction in fmax
            state["oscilating_count"][not_converged] = oscilating_count
            if (n // 10) > 0 and (istep % (n // 10)) == 0:
                BatchOptimizer.print_stats(state, opt_patience)
        
        if istep == n:
            main_logger.info(f"  reached maximum optimization step {istep}")
        else:
            main_logger.info(f"  finished at step {istep}")
        
        BatchOptimizer.print_stats(state, opt_patience)


    @staticmethod
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
        return lists_padded


    @staticmethod
    def padding_species(lists, pad_value=-1):
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
        return lists_padded
    

    @staticmethod
    def print_stats(state, opt_patience):
        """Print the optimization status"""
        numbers = state['numbers']
        num_total = numbers.size()[0]
        num_converged_dropped = torch.sum(state['converged_mask']).to('cpu')
        oscillating_count = state['oscilating_count'].to('cpu').reshape(-1, ) >= opt_patience
        num_dropped = torch.sum(oscillating_count)
        num_converged = num_converged_dropped - num_dropped
        num_active = num_total - num_converged_dropped
        main_logger.info(f"  total {num_total:3d} "
                         f"converged {num_converged:3d} "
                         f"dropped(oscillating) {num_dropped:3d} "
                         f"active {num_active:3d}"
                         )
    

    def ensemble_opt(self, net:object) -> dict:
        """optimizing a group of molecules
        Arguments:
            net: an EnForce_ANI object
        """
        coord = torch.tensor(self.coord, dtype=torch.float, device=self.device)
        numbers = torch.tensor(self.numbers, dtype=torch.long, device=self.device)
        charges = torch.tensor(self.charges, dtype=torch.long, device=self.device)
        converged_mask = torch.zeros(coord.shape[0], dtype=torch.bool, device=self.device)
        fmax = torch.full(coord.shape[:1], 999.0, device=coord.device)  
        # size=N, a tensored filled with 999.0, 
        # representing the current maximum forces at each conformer.
        energy = torch.full(coord.shape[:1], 999.0, dtype=torch.double, device=coord.device)
        init_energy = torch.full(coord.shape[:1], 999.0, dtype=torch.double, device=coord.device)
        ids = torch.arange(coord.shape[0], device=coord.device)  
        # Returns a 1D tensor
        # optimizer = FIRE(coord)
        state = dict(ids=ids,
                     coord=coord, 
                     numbers=numbers,
                     converged_mask=converged_mask,
                     # optimizer=optimizer,
                     nn=net,
                     fmax=fmax,
                     energy=energy,
                     init_energy=init_energy,
                     timing=defaultdict(float),
                     charges=charges,
                     he=list(),
                     close=list())
        
        BatchOptimizer.n_steps(
            state, 
            self.config['steps'], 
            self.config['tol'], 
            self.config['patience'],
            )
        
        return dict(
            coord=state['coord'].tolist(),
            ids=state['ids'].tolist(),
            init_energy=state['init_energy'].tolist(),
            energy=state['energy'].tolist(),
            fmax=state['fmax'].tolist(),
            he=state['he'],
            close=state['close'],
            timing=dict(state['timing']),
            numbers=state['numbers'].tolist(),
            )


    def run(self) -> dict:
        for p in self.ani.parameters():
            p.requires_grad_(False)
        ani = EnForce_ANI(self.ani, self.model, self.config["batchsize"])
        with torch.jit.optimized_execution(False):
            return self.ensemble_opt(ani)