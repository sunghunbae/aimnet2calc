import torch
import torch.nn as nn
from torch.optim import Adam, LBFGS
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import logging

@dataclass
class BatchConfig:
    """Configuration for batch optimization"""
    batch_size: int = 32
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    learning_rate: float = 0.001
    optimizer_type: str = "adam"  # "adam" or "lbfgs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
class AIMNet2BatchOptimizer:
    """
    Batch optimizer for AIMNet2 neural network potential model
    Handles multiple molecular systems simultaneously for efficient optimization
    """
    
    def __init__(self, model, config: BatchConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize optimizer
        self.optimizer = self._setup_optimizer()
        
        # Track optimization history
        self.history = {
            'energies': [],
            'forces': [],
            'gradients': [],
            'convergence': []
        }
    
    def _setup_optimizer(self):
        """Setup the optimizer based on configuration"""
        if self.config.optimizer_type.lower() == "adam":
            return Adam(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer_type.lower() == "lbfgs":
            return LBFGS(self.model.parameters(), 
                        lr=self.config.learning_rate,
                        max_iter=20,
                        line_search_fn='strong_wolfe')
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")
    
    def prepare_batch(self, coord: List[torch.Tensor], 
                     numbers: List[torch.Tensor],
                     charge: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Prepare batch data for AIMNet2 model
        
        Args:
            coord: List of coordinate tensors (N_atoms, 3) for each system
            numbers: List of atomic number tensors (N_atoms,) for each system
            charge: Optional list of charge tensors for each system
            
        Returns:
            Batched data dictionary
        """
        batch_size = len(coord)
        
        # Find maximum number of atoms for padding
        max_atoms = max(coord.shape[0] for coord in coord)
        
        # Initialize batched tensors
        batch_coords = torch.zeros(batch_size, max_atoms, 3, device=self.device)
        batch_atomic_nums = torch.zeros(batch_size, max_atoms, dtype=torch.long, device=self.device)
        batch_masks = torch.zeros(batch_size, max_atoms, dtype=torch.bool, device=self.device)
        
        if charge is not None:
            batch_charge = torch.zeros(batch_size, device=self.device)
        
        # Fill batched tensors
        for i, (coord, atomic_num) in enumerate(zip(coord, numbers)):
            n_atoms = coord.shape[0]
            batch_coords[i, :n_atoms] = coord.to(self.device)
            batch_atomic_nums[i, :n_atoms] = atomic_num.to(self.device)
            batch_masks[i, :n_atoms] = True
            
            if charge is not None:
                batch_charge[i] = charge[i].to(self.device)
        
        batch_data = {
            'coord': batch_coords,
            'numbers': batch_atomic_nums,
            'mask': batch_masks,
        }
        
        if charge is not None:
            batch_data['charge'] = batch_charge

        # nbmat
        batch_data = self.model.make_nbmat(batch_data)

        return batch_data
    
    def compute_batch_energy_forces(self, batch_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute energies and forces for a batch of systems
        
        Args:
            batch_data: Batched molecular data
            
        Returns:
            Tuple of (energies, forces) tensors
        """
        coord = batch_data['coord']
        coord.requires_grad_(True)
        
        # Forward pass through AIMNet2
        with torch.enable_grad():
            energies = self.model(batch_data)
            
            # Compute forces as negative gradient of energy w.r.t. coord
            forces = -torch.autograd.grad(
                energies.sum(),
                coord,
                create_graph=True,
                retain_graph=True
            )[0]
        
        return energies, forces
    
    def optimize_batch(self, 
                      coord: List[torch.Tensor],
                      numbers: List[torch.Tensor],
                      charge: Optional[List[torch.Tensor]] = None,
                      target_energies: Optional[torch.Tensor] = None,
                      target_forces: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Optimize a batch of molecular systems
        
        Args:
            coord: Initial coord for each system
            numbers: Atomic numbers for each system
            charge: Optional charge for each system
            target_energies: Optional target energies for supervised learning
            target_forces: Optional target forces for supervised learning
            
        Returns:
            Dictionary with optimized results
        """
        # Prepare batch data
        batch_data = self.prepare_batch(coord, numbers, charge)
        
        # Convert coord to optimizable parameters
        opt_coords = batch_data['coord'].clone().detach().requires_grad_(True)
        batch_data['coord'] = opt_coords
        
        # Setup optimizer for coord
        coord_optimizer = Adam([opt_coords], lr=self.config.learning_rate)
        
        self.logger.info(f"Starting batch optimization for {len(coord)} systems")
        
        for iteration in range(self.config.max_iterations):
            coord_optimizer.zero_grad()
            
            # Compute energies and forces
            energies, forces = self.compute_batch_energy_forces(batch_data)
            
            # Compute loss
            loss = self._compute_loss(energies, forces, target_energies, target_forces)
            
            # Backward pass
            loss.backward()
            
            # Update coord
            coord_optimizer.step()
            
            # Check convergence
            grad_norm = torch.norm(opt_coords.grad).item()
            converged = grad_norm < self.config.convergence_threshold
            
            # Log progress
            if iteration % 100 == 0 or converged:
                self.logger.info(f"Iteration {iteration}: Loss = {loss.item():.6f}, "
                               f"Grad norm = {grad_norm:.6f}")
            
            # Store history
            self.history['energies'].append(energies.detach().cpu())
            self.history['forces'].append(forces.detach().cpu())
            self.history['gradients'].append(grad_norm)
            self.history['convergence'].append(converged)
            
            if converged:
                self.logger.info(f"Converged after {iteration} iterations")
                break
        
        return {
            'optimized_coord': opt_coords.detach().cpu(),
            'final_energies': energies.detach().cpu(),
            'final_forces': forces.detach().cpu(),
            'iterations': iteration + 1,
            'converged': converged
        }
    
    def _compute_loss(self, 
                     energies: torch.Tensor,
                     forces: torch.Tensor,
                     target_energies: Optional[torch.Tensor] = None,
                     target_forces: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        Compute optimization loss
        
        Args:
            energies: Predicted energies
            forces: Predicted forces
            target_energies: Target energies (for supervised learning)
            target_forces: Target forces (for supervised learning)
            
        Returns:
            Loss tensor
        """
        loss = torch.tensor(0.0, device=self.device)
        
        if target_energies is not None:
            # Energy loss
            energy_loss = torch.mean((energies - target_energies.to(self.device))**2)
            loss += energy_loss
        
        if target_forces is not None:
            # Force loss
            force_loss = torch.tensor(0.0, device=self.device)
            for i, target_force in enumerate(target_forces):
                pred_force = forces[i, :target_force.shape[0]]
                force_loss += torch.mean((pred_force - target_force.to(self.device))**2)
            loss += force_loss / len(target_forces)
        
        # If no targets provided, use force magnitude as regularization
        if target_energies is None and target_forces is None:
            # Minimize force magnitude (geometry optimization)
            force_loss = torch.mean(torch.sum(forces**2, dim=-1))
            loss += force_loss
        
        return loss
    
    def optimize_geometries(self, 
                          coord: List[torch.Tensor],
                          numbers: List[torch.Tensor],
                          charge: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Optimize molecular geometries to minimize forces
        
        Args:
            coord: Initial coord
            numbers: Atomic numbers
            charge: Optional charge
            
        Returns:
            Optimized geometries and energies
        """
        return self.optimize_batch(coord, numbers, charge)
    
    def train_on_batch(self,
                      coord: List[torch.Tensor],
                      numbers: List[torch.Tensor],
                      target_energies: torch.Tensor,
                      target_forces: List[torch.Tensor],
                      charge: Optional[List[torch.Tensor]] = None) -> float:
        """
        Train the AIMNet2 model on a batch of data
        
        Args:
            coord: Input coord
            numbers: Atomic numbers
            target_energies: Target energies
            target_forces: Target forces
            charge: Optional charge
            
        Returns:
            Training loss
        """
        self.optimizer.zero_grad()
        
        # Prepare batch
        batch_data = self.prepare_batch(coord, numbers, charge)
        
        # Forward pass
        energies, forces = self.compute_batch_energy_forces(batch_data)
        
        # Compute loss
        loss = self._compute_loss(energies, forces, target_energies, target_forces)
        
        # Backward pass
        loss.backward()
        
        # Update model parameters
        self.optimizer.step()
        
        return loss.item()
    
    def get_optimization_history(self) -> Dict[str, List]:
        """Get optimization history for analysis"""
        return self.history
    
    def save_checkpoint(self, filepath: str):
        """Save model and optimizer state"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model and optimizer state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.logger.info(f"Checkpoint loaded from {filepath}")


from rdworks import MolLibr


def libr_to_batches(libr:MolLibr, batchsize:int=1000) -> list:
    """Split workload flexibily into a numer of batches.
    
    - Each batch has up to `batchsize` number of atoms. 
    - Conformers originated from a same molecule can be splitted into multiple batches.
    - Or one batch can contain conformers originated from multiple molecules.

    coord: coord of input molecules (N, m, 3) where N is the number of structures and 
    m is the number of atoms in each structure.
    numbers: atomic numbers in the molecule (include H). (N, m)
    charge: (N,)

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
        charge = [mol.props['charge'] for mol in batch_mols]
        coord = [conf.rdmol.GetConformer().GetPositions().tolist() for conf in batch_confs]
        # to be consistent with legacy code
        coord = [[tuple(xyz) for xyz in inner] for inner in coord]
        # numbers should be got from conformers because of hydrogens
        numbers = [[a.GetAtomicNum() for a in conf.rdmol.GetAtoms()] for conf in batch_confs]
        print(f"batch {i:3d} <- {len(batch_confs):4d} conformers {batch_n_atoms:6d} atoms")
        batches.append((coord, numbers, charge, batch_confs, batch_mols))
    return batches



# Example usage
if __name__ == "__main__":
    # Example of how to use the batch optimizer
    
    # Initialize configuration
    config = BatchConfig(
        batch_size=16,
        max_iterations=500,
        convergence_threshold=1e-5,
        learning_rate=0.001,
        optimizer_type="adam"
    )
    
    from aimnet2calc import AIMNet2Calculator
    import rdworks
    import os

    workdir = os.path.dirname(__file__)
    
    # Assuming you have an AIMNet2 model instance
    model = AIMNet2Calculator('aimnet2').model  # Your AIMNet2 model
    
    # Initialize optimizer
    optimizer = AIMNet2BatchOptimizer(model, config)
    
    # Example coordinate and atomic number data
    # coord = [torch.randn(10, 3) for _ in range(16)]  # 16 systems, 10 atoms each
    # numbers = [torch.randint(1, 10, (10,)) for _ in range(16)]
    
    libr = rdworks.read_sdf(os.path.join(workdir, "cyclooctane.sdf"), confs=True)
    coord = [conf.rdmol.GetConformer().GetPositions().tolist() for mol in libr for conf in mol.confs]
    coord = [[tuple(xyz) for xyz in inner] for inner in coord]
    coord = torch.tensor(coord)
    # numbers should be got from conformers because of hydrogens
    numbers = [[a.GetAtomicNum() for a in conf.rdmol.GetAtoms()] for mol in libr for conf in mol.confs]
    numbers = torch.tensor(numbers)
    charge = [mol.props['charge'] for mol in libr]
    charge = torch.tensor(charge)

    # for mol in libr:
    #     for conf in mol.confs:

    # libr += libr.copy()

    # batches = libr_to_batches(libr)
    
    print(libr.count(), "molecules")
    # print("batches", batches)
    print(coord)
    print(numbers)
    print(charge)

    # e_ref = -314.689736079491 * hartree2ev

    # for (coord, numbers, charge, batch_confs, batch_mols) in batches:
            # data = {
            #     'coord'  : padding_coords(coord),
            #     'numbers': padding_numbers(numbers),
            #     'charge' : np.array(charge),
            #     }
            # print('energy: ', _test_energy(calc, data))
            # print('forces: ', _test_forces(calc, data))

    # Optimize geometries
    results = optimizer.optimize_geometries(coord, numbers, charge)
    
    # Or train on batch
    # target_energies = torch.randn(16)
    # target_forces = [torch.randn(10, 3) for _ in range(16)]
    # loss = optimizer.train_on_batch(coord, numbers, target_energies, target_forces)
    
    print("AIMNet2 Batch Optimizer implementation complete!")