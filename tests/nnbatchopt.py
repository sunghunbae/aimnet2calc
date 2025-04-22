import os
import time
import math
import logging
import importlib.resources

from pathlib import PosixPath
from typing import List, Union, Tuple, Optional, Self
from datetime import timedelta

try:
    import torch
    torch.manual_seed(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except:
    raise ImportError('torch is required')

import numpy as np
import psutil

from rdwork.auto3d.batchopt import BatchOptimizer, hartree2ev
from rdwork.xtb.wrapper import XTBOptimizer


main_logger = logging.getLogger()


class NNBatchOpt:
    def __init__(self) -> None:
        """Adds method functions to `MolLibr` class
        """
        pass


    def split_confs_into_batches(self, settings:dict) -> list:
        """Split workload flexibily into a numer of batches.
        
        - Each batch has up to `batchsize_atoms` number of atoms. 
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
        for mol in self.libr:
            for conf in mol.confs:
                n_atoms = conf.props['atoms']
                if (batch_n_atoms + n_atoms) > settings['batchsize_atoms']:
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
            if settings['model'] == "ANI-2xt":
                ani2xt_index = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 17: 6} # H,C,N,O,S,F,Cl
                numbers = [[ani2xt_index[a.GetAtomicNum()] for a in conf.rdmol.GetAtoms()] for conf in batch_confs]
            else:
                numbers = [[a.GetAtomicNum() for a in conf.rdmol.GetAtoms()] for conf in batch_confs]
            main_logger.info(f"batch {i:3d} <- {len(batch_confs):4d} conformers {batch_n_atoms:6d} atoms")
            batches.append((coord, numbers, charges, batch_confs, batch_mols))
        return batches


    def nn_singlepoint(self, model:str="ANI-2x", model_path:PosixPath=importlib.resources.files('rdwork.auto3d.models'),
                       log:Optional[Union[PosixPath, str]]=None, batchsize_atoms:int=16*1024, steps:int=1,
                       tol:float=0.003, patience:int=1000, gpu:bool=True, gpu_idx:int=0) -> Self:
        """Perform single point energy evaluation of 3D coordinates using a neural network potential.

            Args:
                model (str): neural network model for energy and force evaluation (`ANI-2x` | `ANI-2xt` | `AIMNET`).
                model_path (path): dirname or path to `ANI-2xt` and `AIMNET` models. Defaults to `rdwork.auto3d.models` folder.
                log (path): log filename or path.
                batchsize_atoms (int): number of atoms in one batch, defaults to 1024 * 16.
                    see `batchNNopt.EnForce_ANI` class. In RTX2080 (11GB), batchsize_atoms of 1024*16 makes 
                    up to 70-85% usage of GPU and taking up ~3GB GPU memory.
                steps (int): maximum optimization steps for each structure, defaults to 5000.
                tol (float): converged if maximum force is below this threshold. Defaults to 0.003 (eV/Angstrom).
                patience (int): if the force does not decrease for a continuous patience steps, 
                    the conformer will drop out of the optimization loop, defaults to 1000.
                gpu (bool): use GPU if True.
                gpu_idx (int): GPU device to use.

            Returns:
                Self: self.
        """
        if isinstance(log, os.PathLike) or isinstance(log, str):
            # If this class is initiated with a different log file,
            # the previous file handler should be closed before
            # opening a new one.
            # logger.handlers shoule be [ StreamHandler, FileHandler ]
            # note that .addHandler() would not add the identical handler
            if len(main_logger.handlers) < 2:
                logger_fh = logging.FileHandler(log, mode='a', encoding='utf-8')
                logger_fh.setFormatter(
                    logging.Formatter(fmt='%(asctime)s %(levelname)s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S'))
                main_logger.addHandler(logger_fh)
        if gpu and torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_idx}")
            memory = int(math.ceil(torch.cuda.get_device_properties(gpu_idx).total_memory/(1024**3)))
        else:
            device = torch.device("cpu")          
            memory = int(psutil.virtual_memory().total/(1024**3))

        settings = {
            "model": model,
            "model_path": model_path,
            "device": device,
            "steps": steps, 
            "tol": tol, 
            "patience": patience, 
            "batchsize_atoms": batchsize_atoms,
            "memory": memory,
            "log": log,
        }
        start = time.time()
        main_logger.info(f">>> Single-Point Energy Using ANI Neural Network Model <<<")
        for k, v in settings.items():
            main_logger.info(f"{k:<20} {v}")
        main_logger.info(f"{'number of molecules':<20} {self.count()}")
        main_logger.info(f"{'number of conformers':<20} {sum([_.count() for _ in self.libr])}")
        
        batches = self.split_confs_into_batches(settings)
        eta_data = []
        for i, (coord, numbers, charges, confs, _) in enumerate(batches, start=1):
            eta_data.append(time.time())
            if len(eta_data) >= 2:
                eta_sec = (len(batches)-i+1) * np.mean([ eta_data[x]-eta_data[x-1] for x in range(1, len(eta_data)) ])
                main_logger.info(f"batch {i} of {len(batches)} [ETA {timedelta(seconds=eta_sec)}]")
            else:
                main_logger.info(f"batch {i} of {len(batches)}")
            optdict = BatchOptimizer(coord, numbers, charges, settings).run()
            for (conf, init_energy) in zip(confs, optdict['init_energy']):
                conf.props.update({'E_tot(eV)': init_energy})     
        elapsed = (time.time() - start) # (sec)
        main_logger.info(f'elapsed {str(timedelta(seconds=elapsed))}')

        return self


    def nn_opt(self, 
               model:str="ANI-2x", 
               model_path:PosixPath=importlib.resources.files('rdwork.auto3d.models'),
               log:Optional[Union[PosixPath, str]]=None, 
               batchsize_atoms:int=16*1024, 
               steps:int=5000,
               tol:float=0.003, 
               patience:int=1000, 
               gpu:bool=True, 
               gpu_idx:int=0) -> Self:
        """Optimize 3D coordinates using a neural network potential.

            Args:
                model (str): neural network model for energy and force evaluation (`ANI-2x` | `ANI-2xt` | `AIMNET`).
                model_path (path): dirname or path to ``ANI-2xt`` and ``AIMNET`` models. Defaults to `rdwork.auto3d.models` folder.
                log (path): log filename or path.
                batchsize_atoms (int): number of atoms in one batch, defaults to 1024 * 16. See `batchNNopt.EnForce_ANI` class. 
                    In RTX2080 (11GB), batchsize_atoms of 1024*16 makes up to 70-85% usage of GPU and taking up ~3GB GPU memory.
                steps (int): maximum optimization steps for each structure, defaults to 5000.
                tol (float): converged if maximum force is below this threshold. Defaults to 0.003 (eV/Angstrom).
                patience (int): if the force does not decrease for a continuous patience steps, 
                    the conformer will drop out of the optimization loop, defaults to 1000.
                gpu (bool): use GPU if True.
                gpu_idx (int): GPU device to use.

            Returns:
                Self: self.
        """
        if isinstance(log, os.PathLike) or isinstance(log, str):
            # If this class is initiated with a different log file,
            # the previous file handler should be closed before
            # opening a new one.
            # logger.handlers shoule be [ StreamHandler, FileHandler ]
            # note that .addHandler() would not add the identical handler
            if len(main_logger.handlers) < 2:
                logger_fh = logging.FileHandler(log, mode='a', encoding='utf-8')
                logger_fh.setFormatter(
                    logging.Formatter(fmt='%(asctime)s %(levelname)s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S'))
                main_logger.addHandler(logger_fh)
        if gpu and torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_idx}")
            memory = int(math.ceil(torch.cuda.get_device_properties(gpu_idx).total_memory/(1024**3)))
        else:
            device = torch.device("cpu")          
            memory = int(psutil.virtual_memory().total/(1024**3))

        settings = {
            "model": model,
            "model_path": model_path,
            "device": device,
            "steps": steps, 
            "tol": tol, 
            "patience": patience, 
            "batchsize_atoms": batchsize_atoms,
            "memory": memory,
            "log": log,
        }
        start = time.time()
        main_logger.info(f">>> Optimizing Conformers Using ANI Neural Network Model <<<")
        for k, v in settings.items():
            main_logger.info(f"{k:<20} {v}")
        main_logger.info(f"{'number of molecules':<20} {self.count()}")
        main_logger.info(f"{'number of conformers':<20} {sum([_.count() for _ in self.libr])}")
        
        batches = self.split_confs_into_batches(settings)
        
        eta_data = []
        for i, (coord, numbers, charges, confs, _) in enumerate(batches, start=1):
            eta_data.append(time.time())
            if len(eta_data) >= 2:
                eta_sec = (len(batches)-i+1) * np.mean([ eta_data[x]-eta_data[x-1] for x in range(1, len(eta_data)) ])
                main_logger.info(f"batch {i} of {len(batches)} [ETA {timedelta(seconds=eta_sec)}]")
            else:
                main_logger.info(f"batch {i} of {len(batches)}")

            optdict = BatchOptimizer(coord, numbers, charges, settings).run()

            for (conf, coord, energy, fmax) in zip(confs, optdict['coord'], optdict['energy'], optdict['fmax']):
                conf.sync(coord)
                conf.props.update({
                    'E_tot(eV)': energy,
                    'Converged': fmax < tol,
                    })
                
        elapsed = (time.time() - start) # (sec)
        main_logger.info(f'elapsed {str(timedelta(seconds=elapsed))}')
        return self


    def xtb_singlepoint(self, method:str="GFN2-xTB", log:Optional[Union[PosixPath, str]]=None) -> Self:
        """Perform single point energy evaluation of 3D coordinates using GFNx-xTB.

            Args:
                method (str): method.
                log (path): log filename or path.
                
            Returns:
                Self: self.
        """
        if isinstance(log, os.PathLike) or isinstance(log, str):
            # If this class is initiated with a different log file,
            # the previous file handler should be closed before
            # opening a new one.
            # logger.handlers shoule be [ StreamHandler, FileHandler ]
            # note that .addHandler() would not add the identical handler
            if len(main_logger.handlers) < 2:
                logger_fh = logging.FileHandler(log, mode='a', encoding='utf-8')
                logger_fh.setFormatter(
                    logging.Formatter(fmt='%(asctime)s %(levelname)s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S'))
                main_logger.addHandler(logger_fh)
        
        settings = {
            "method": method,
            "log": log,
        }
        start = time.time()
        main_logger.info(f">>> Single-Point Energy Using GFN2-xTB <<<")
        for k, v in settings.items():
            main_logger.info(f"{k:<20} {v}")
        main_logger.info(f"{'number of molecules':<20} {self.count()}")
        main_logger.info(f"{'number of conformers':<20} {sum([_.count() for _ in self.libr])}")
        
        for mol in self.libr:
            for conf in mol.confs:
                xtb = XTBOptimizer(conf.rdmol)
                Eh = xtb.singlepoint() # Eh
                conf.props.update({'E_tot(eV)': Eh * hartree2ev})
                
        elapsed = (time.time() - start) # (sec)
        main_logger.info(f'elapsed {str(timedelta(seconds=elapsed))}')
        return self
    

    def xtb_opt(self, method:str="GFN2-xTB", log:Optional[Union[PosixPath, str]]=None) -> Self:
        """Optimize 3D coordinates using GFNx-xTB.

            Args:
                method (str): method.
                log (path): log filename or path.
                
            Returns:
                Self: self.
        """
        if isinstance(log, os.PathLike) or isinstance(log, str):
            # If this class is initiated with a different log file,
            # the previous file handler should be closed before
            # opening a new one.
            # logger.handlers shoule be [ StreamHandler, FileHandler ]
            # note that .addHandler() would not add the identical handler
            if len(main_logger.handlers) < 2:
                logger_fh = logging.FileHandler(log, mode='a', encoding='utf-8')
                logger_fh.setFormatter(
                    logging.Formatter(fmt='%(asctime)s %(levelname)s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S'))
                main_logger.addHandler(logger_fh)
        
        settings = {
            "method": method,
            "log": log,
        }
        start = time.time()
        main_logger.info(f">>> Optimizing Conformers Using GFN2-xTB <<<")
        for k, v in settings.items():
            main_logger.info(f"{k:<20} {v}")
        main_logger.info(f"{'number of molecules':<20} {self.count()}")
        main_logger.info(f"{'number of conformers':<20} {sum([_.count() for _ in self.libr])}")
        
        for mol in self.libr:
            for conf in mol.confs:
                xtb = XTBOptimizer(conf.rdmol)
                Eh, rdmol_opt = xtb.optimize()
                conf.rdmol = rdmol_opt
                conf.props.update({
                    'E_tot(eV)': Eh * hartree2ev,
                    'Converged': True,
                    })
                
        elapsed = (time.time() - start) # (sec)
        main_logger.info(f'elapsed {str(timedelta(seconds=elapsed))}')
        return self
