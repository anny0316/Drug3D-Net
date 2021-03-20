from keras.utils import to_categorical, Sequence
from rdkit import Chem
from rdkit.Chem import rdmolops, AllChem
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
from model.utils_ import read_csv, read_csv2, read_griddata, normalized_laplacian, normalize_adj, scaled_laplacian, adjacency, gen_conformer
from scipy.spatial import cKDTree
from grid import scaffoldSplit as scaffoldsplit

def one_hot(x, allowable_set):
    # If x is not in allowed set, use last index
    if x not in allowable_set:
        x = allowable_set[-1]

    return list(map(lambda s: x == s, allowable_set))


class grid_Dataset(object):
    def __init__(self, dataset, batch=128):
        self.dataset = dataset
        self.task = "binary"
        self.target_name = "active"
        self.max_atoms = 3

        self.batch = batch
        self.outputs = 1

        self.smiles = []
        self.mols = []
        self.coords = []
        self.target = []
        self.rlist = []
        self.gridx = []
        self.x, self.y, self.grid3d = {}, {}, {}
        self.gridshape = ()

        self.use_atom_symbol = True
        self.use_degree = True
        self.use_hybridization = True
        self.use_implicit_valence = True
        self.use_partial_charge = False
        self.use_formal_charge = True
        self.use_ring_size = True
        self.use_hydrogen_bonding = True
        self.use_acid_base = True
        self.use_aromaticity = True
        self.use_chirality = True
        self.use_num_hydrogen = True

        # Load data
        self.load_grid_dataset()

        # Normalize
        if self.task == "regression":
            self.mean = np.mean(self.y["train"])
            self.std = np.std(self.y["train"])

            self.y["train"] = (self.y["train"] - self.mean) / self.std
            self.y["valid"] = (self.y["valid"] - self.mean) / self.std
            self.y["test"] = (self.y["test"] - self.mean) / self.std

        else:
            self.mean = 0
            self.std = 1

    def load_grid_dataset(self):
        # Dataset parameters
        if self.dataset == "bace_reg" or self.dataset == "delaney" or self.dataset == "freesolv":
            self.task = "regression"
            # self.target_name = "target"
        elif self.dataset == "hiv":
            self.task = "binary"
        else:
            pass


        if self.dataset == "delaney":
            grid_x, grid_y, grid_smiles, sample_shape = read_griddata("gridMols/grid3Dmols_delaney11")
        elif self.dataset == "freesolv":
            grid_x, grid_y, grid_smiles, sample_shape = read_griddata("gridMols/grid3Dmols_freesolv_rotate_5")
        elif self.dataset == "hiv":
            grid_x, grid_y, grid_smiles, sample_shape = read_griddata("gridMols/grid3Dmols_hiv_rotate")
            t = np.array(grid_y)
            t1=0
            t2=0
            for h in range(len(t)):
                if t[h]==0:
                    t1=t1+1
                elif t[h]==1:
                    t2 = t2 + 1

        elif self.dataset == "tox21_NR-AR":
            grid_x, grid_y, grid_smiles, sample_shape = read_griddata("gridMols/grid3Dmols_tox21_NR-AR_rotate")
        elif self.dataset == "tox21_NR-AR-LBD":
            grid_x, grid_y, grid_smiles, sample_shape = read_griddata("gridMols/grid3Dmols_tox21_NR-AR-LBD_rotate")
        elif self.dataset == "tox21_NR-AhR":
            grid_x, grid_y, grid_smiles, sample_shape = read_griddata("gridMols/grid3Dmols_tox21_NR-AhR_rotate")
        elif self.dataset == "tox21_NR-Aromatase":
            grid_x, grid_y, grid_smiles, sample_shape = read_griddata("gridMols/grid3Dmols_tox21_NR-Aromatase_rotate")
        elif self.dataset == "tox21_NR-ER":
            grid_x, grid_y, grid_smiles, sample_shape = read_griddata("gridMols/grid3Dmols_tox21_NR-ER_rotate")
        elif self.dataset == "tox21_NR-ER-LBD":
            grid_x, grid_y, grid_smiles, sample_shape = read_griddata("gridMols/grid3Dmols_tox21_NR-ER-LBD_rotate")
        elif self.dataset == "tox21_NR-PPAR-gamma":
            grid_x, grid_y, grid_smiles, sample_shape = read_griddata("gridMols/grid3Dmols_tox21_NR-PPAR-gamma_rotate")
        elif self.dataset == "tox21_SR-ARE":
            grid_x, grid_y, grid_smiles, sample_shape = read_griddata("gridMols/grid3Dmols_tox21_SR-ARE_rotate")
        elif self.dataset == "tox21_SR-ATAD5":
            grid_x, grid_y, grid_smiles, sample_shape = read_griddata("gridMols/grid3Dmols_tox21_SR-ATAD5_rotate")
        elif self.dataset == "tox21_SR-HSE":
            grid_x, grid_y, grid_smiles, sample_shape = read_griddata("gridMols/grid3Dmols_tox21_SR-HSE_rotate")
        elif self.dataset == "tox21_SR-MMP":
            grid_x, grid_y, grid_smiles, sample_shape = read_griddata("gridMols/grid3Dmols_tox21_SR-MMP_rotate")
        elif self.dataset == "tox21_SR-p53":
            grid_x, grid_y, grid_smiles, sample_shape = read_griddata("gridMols/grid3Dmols_tox21_SR-p53_rotate")


        self.smiles, self.gridx, self.gridy, self.gridshape = np.array(grid_smiles), np.array(grid_x), np.array( grid_y), sample_shape

        if self.dataset == "hiv":
            train_inds, valid_inds, test_inds = scaffoldsplit.ScaffoldSplitter().train_valid_test_split(self.gridx, self.gridy, self.smiles)
            train_smiles = self.smiles[train_inds]
            train_gridy = self.gridy[train_inds]
            train_grid3d = self.gridx[train_inds]
            np.random.seed(66)
            index_train = np.random.permutation(len(train_smiles))

            valid_smiles = self.smiles[valid_inds]
            valid_gridy = self.gridy[valid_inds]
            valid_grid3d = self.gridx[valid_inds]
            index_valid = np.random.permutation(len(valid_smiles))

            test_smiles = self.smiles[test_inds]
            test_gridy = self.gridy[test_inds]
            test_grid3d = self.gridx[test_inds]
            index_test = np.random.permutation(len(test_smiles))

            self.x = {"train": train_smiles[index_train],
                      "valid": valid_smiles[index_valid],
                      "test": test_smiles[index_test]}
            self.y = {"train": train_gridy[index_train],
                      "valid": valid_gridy[index_valid],
                      "test": test_gridy[index_test]}
            self.grid3d = {"train": train_grid3d[index_train],
                           "valid": valid_grid3d[index_valid],
                           "test": test_grid3d[index_test]}
        else:
            # Shuffle data
            idx = np.random.permutation(len(self.smiles))
            self.smiles, self.gridx, self.gridy = self.smiles[idx], self.gridx[idx], self.gridy[idx]

            # Split data
            spl1 = int(len(self.smiles) * 0.2)
            spl2 = int(len(self.smiles) * 0.1)

            self.x = {"train": self.smiles[spl1:],
                      "valid": self.smiles[spl2:spl1],
                      "test": self.smiles[:spl2]}
            self.y = {"train": self.gridy[spl1:],
                      "valid": self.gridy[spl2:spl1],
                      "test": self.gridy[:spl2]}
            self.grid3d = {"train": self.gridx[spl1:],
                           "valid":self.gridx[spl2:spl1],
                           "test":self.gridx[:spl2]}
        print("aa")

    def save_dataset(self, path, pred=None, target="test", filename=None):
        mols = []
        # for idx, (smile, y) in enumerate(zip(self.t_smiles[target], self.y[target])):
        #     smile.SetProp("true", str(y * self.std + self.mean))
        #     # smile.SetProp("smiles", self.smiles[idx])
        #     # smile.SetProp("name", self.x[target][idx])
        #     if pred is not None:
        #         smile.SetProp("pred", str(pred[idx][0] * self.std + self.mean))
        #     mols.append(smile)
        #
        # if filename is not None:
        #     w = Chem.SDWriter(path + filename + ".sdf")
        # else:
        #     w = Chem.SDWriter(path + target + ".sdf")
        # for mol in mols:
        #     if mol is not None:
        #         w.write(mol)

    def replace_dataset(self, path, subset="test", target_name="target"):
        x, c, y = [], [], []
        mols = Chem.SDMolSupplier(path)

        for mol in mols:
            if mol is not None:
                # Multitask
                if type(target_name) is list:
                    y.append([float(mol.GetProp(t)) if t in mol.GetPropNames() else -1 for t in target_name])
                    self.outputs = len(self.target_name)

                # Singletask
                elif target_name in mol.GetPropNames():
                    _y = float(mol.GetProp(target_name))
                    if _y == -1:
                        continue
                    else:
                        y.append(_y)

                else:
                    continue

                x.append(mol)
                c.append(mol.GetConformer().GetPositions())

        # Normalize
        x = np.array(x)
        c = np.array(c)
        y = (np.array(y) - self.mean) / self.std

        self.x[subset] = x
        self.c[subset] = c
        self.y[subset] = y.astype(int) if self.task != "regression" else y

    def set_features(self, use_atom_symbol=True, use_degree=True, use_hybridization=True, use_implicit_valence=True,
                     use_partial_charge=False, use_formal_charge=True, use_ring_size=True, use_hydrogen_bonding=True,
                     use_acid_base=True, use_aromaticity=True, use_chirality=True, use_num_hydrogen=True):

        self.use_atom_symbol = use_atom_symbol
        self.use_degree = use_degree
        self.use_hybridization = use_hybridization
        self.use_implicit_valence = use_implicit_valence
        self.use_partial_charge = use_partial_charge
        self.use_formal_charge = use_formal_charge
        self.use_ring_size = use_ring_size
        self.use_hydrogen_bonding = use_hydrogen_bonding
        self.use_acid_base = use_acid_base
        self.use_aromaticity = use_aromaticity
        self.use_chirality = use_chirality
        self.use_num_hydrogen = use_num_hydrogen

    def generator(self, target, task=None):
        return grid_MPGenerator(self.x[target], self.y[target], self.grid3d[target], self.gridshape, self.batch,
                           task=task if task is not None else self.task,
                           use_atom_symbol=self.use_atom_symbol,
                           use_degree=self.use_degree,
                           use_hybridization=self.use_hybridization,
                           use_implicit_valence=self.use_implicit_valence,
                           use_partial_charge=self.use_partial_charge,
                           use_formal_charge=self.use_formal_charge,
                           use_ring_size=self.use_ring_size,
                           use_hydrogen_bonding=self.use_hydrogen_bonding,
                           use_acid_base=self.use_acid_base,
                           use_aromaticity=self.use_aromaticity,
                           use_chirality=self.use_chirality,
                           use_num_hydrogen=self.use_num_hydrogen)

class grid_MPGenerator(Sequence):
    def __init__(self, x_set, y_set, grid3d, gridshape, batch, task="binary",
                 use_degree=True, use_hybridization=True, use_implicit_valence=True, use_partial_charge=False,
                 use_formal_charge=True, use_ring_size=True, use_hydrogen_bonding=True, use_acid_base=True,
                 use_aromaticity=True, use_chirality=True, use_num_hydrogen=True, use_atom_symbol=True):
        self.x, self.y = x_set, y_set
        self.grid3d, self.gridshape = grid3d, gridshape
        self.batch = batch
        self.task = task

        self.use_atom_symbol = use_atom_symbol
        self.use_degree = use_degree
        self.use_hybridization = use_hybridization
        self.use_implicit_valence = use_implicit_valence
        self.use_partial_charge = use_partial_charge
        self.use_formal_charge = use_formal_charge
        self.use_ring_size = use_ring_size
        self.use_hydrogen_bonding = use_hydrogen_bonding
        self.use_acid_base = use_acid_base
        self.use_aromaticity = use_aromaticity
        self.use_chirality = use_chirality
        self.use_num_hydrogen = use_num_hydrogen

        self.hydrogen_donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
        self.hydrogen_acceptor = Chem.MolFromSmarts(
            "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]")
        self.acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
        self.basic = Chem.MolFromSmarts(
            "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]")

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch:(idx + 1) * self.batch]
        batch_y = self.y[idx * self.batch:(idx + 1) * self.batch]
        batch_grid = self.grid3d[idx * self.batch:(idx + 1) * self.batch]
        shapelist = list(self.gridshape)

        grid_tensor = np.zeros((len(batch_x), shapelist[0], shapelist[1], shapelist[2], shapelist[3]), dtype = np.bool)

        for mol_idx, mol in enumerate(batch_x):
            # 1. grid3D
            for matrix_ind in batch_grid[mol_idx]:
                grid_tensor[(mol_idx,) + tuple(matrix_ind)] =True

        return [grid_tensor], np.array(batch_y, dtype=float)
