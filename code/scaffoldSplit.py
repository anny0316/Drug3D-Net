import tempfile
import numpy as np
import pandas as pd
import itertools
import os

def generate_scaffold(smiles, include_chirality=False):
  """Compute the Bemis-Murcko scaffold for a SMILES string."""
  from rdkit import Chem
  mol = Chem.MolFromSmiles(smiles)
  engine = ScaffoldGenerator(include_chirality=include_chirality)
  scaffold = engine.get_scaffold(mol)
  return scaffold

class ScaffoldGenerator(object):
  """
  Generate molecular scaffolds.
  Parameters
  ----------
  include_chirality : : bool, optional (default False)
      Include chirality in scaffolds.
  """

  def __init__(self, include_chirality=False):
    self.include_chirality = include_chirality

  def get_scaffold(self, mol):
    """
    Get Murcko scaffolds for molecules.

    Murcko scaffolds are described in DOI: 10.1021/jm9602928. They are
    essentially that part of the molecule consisting of rings and the linker atoms between them.

    Parameters
    ----------
    mols : array_like
        Molecules.
    """
    from rdkit.Chem.Scaffolds import MurckoScaffold
    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=self.include_chirality)


class Splitter(object):
  """Abstract base class for chemically aware splits.."""

  def __init__(self, verbose=True):
    """Creates splitter object."""
    self.verbose = verbose

  def train_valid_test_split(self,
                             grid3d, gridy, smiles,
                             train_dir=None,
                             valid_dir=None,
                             test_dir=None,
                             frac_train=.8,
                             frac_valid=.1,
                             frac_test=.1,
                             seed=None,
                             log_every_n=1000,
                             verbose=True,
                             **kwargs):
    """
        Splits self into train/validation/test sets.

        Returns Dataset objects.
        """
    print("Computing train/valid/test indices", self.verbose)
    train_inds, valid_inds, test_inds = self.split(grid3d, gridy, smiles,
        seed=seed,
        frac_train=frac_train,
        frac_test=frac_test,
        frac_valid=frac_valid,
        log_every_n=log_every_n,
        **kwargs)

    return train_inds, valid_inds, test_inds


  def split(self, grid3d, gridy, smiles,
            seed=None,
            frac_train=None,
            frac_valid=None,
            frac_test=None,
            log_every_n=None,
            verbose=False,
            **kwargs):
    """
    Stub to be filled in by child classes.
    """
    raise NotImplementedError


class ScaffoldSplitter(Splitter):
  """
    Class for doing data splits based on the scaffold of small molecules.
    """

  def split(self, grid3d, gridy, smileslist,
            seed=None,
            frac_train=.8,
            frac_valid=.1,
            frac_test=.1,
            log_every_n=1000):
    """
        Splits internal compounds into train/validation/test by scaffold.
        """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    scaffolds = {}
    print("About to generate scaffolds", self.verbose)
    data_len = len(smileslist)
    for ind, smiles in enumerate(smileslist):
      if ind % log_every_n == 0:
          print("Generating scaffold %d/%d" % (ind, data_len), self.verbose)
      scaffold = generate_scaffold(smiles)
      if scaffold not in scaffolds:
        scaffolds[scaffold] = [ind]
      else:
        scaffolds[scaffold].append(ind)
    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    # scaffold_sets = [
    #     scaffold_set for (scaffold, scaffold_set) in sorted(
    #         scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    # ]
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in scaffolds.items()
    ]
    train_cutoff = frac_train * len(smileslist)
    valid_cutoff = (frac_train + frac_valid) * len(smileslist)
    train_inds, valid_inds, test_inds = [], [], []
    print("About to sort in scaffold sets", self.verbose)
    for scaffold_set in scaffold_sets:
      if len(train_inds) + len(scaffold_set) > train_cutoff:
        if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
          test_inds += scaffold_set
        else:
          valid_inds += scaffold_set
      else:
        train_inds += scaffold_set
    return train_inds, valid_inds, test_inds
