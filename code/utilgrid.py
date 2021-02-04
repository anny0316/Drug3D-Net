import numpy as np
import scipy
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolStandardize
import joblib
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
# from IPython.display import SVG, display
import seaborn as sns; sns.set(color_codes=True)
from rdkit.Chem import AllChem

def read_tox21(raw_filename):
    smiles_tasks_df = pd.read_csv(raw_filename)
    smilesList = smiles_tasks_df.smiles.values
    print("number of all smiles:", len(smilesList))
    atom_num_distribution = []
    remained_smiles = []
    canonical_smiles_list = []
    for smiles in smilesList:
        try:
            mol = Chem.MolFromSmiles(smiles)
            atom_num_distribution.append(len(mol.GetAtoms()))
            remained_smiles.append(smiles)
            canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
        except:
            print("not successfully processed smiles: ", smiles)
            pass

    print("number of successfully processed smiles: ", len(remained_smiles))
    smiles_tasks_df = smiles_tasks_df[smiles_tasks_df["smiles"].isin(remained_smiles)]
    smiles_tasks_df['cano_smiles'] =canonical_smiles_list
    aa=np.array(atom_num_distribution)   #(1, 132)
    plt.figure(figsize=(5, 3))
    sns.set(font_scale=0.8)
    ax = sns.distplot(atom_num_distribution, bins=28, kde=False, color='red')
    ax.set_title("Distribution of atomic number in Tox21 dataset", fontdict={'color':'black', 'family':'Times New Roman'})
    ax.set_xlabel("Number of atoms", fontdict={'color':'black', 'family':'Times New Roman'})
    ax.set_ylabel("Molecule Count", fontdict={'color':'black', 'family':'Times New Roman'})

    # plt.xlabel('Atom Number',fontdict={'color': 'black',
    #                          'family': 'STFangsong',
    #                          'weight': 'normal',
    #                          'size': 15})
    # plt.ylabel('Molecule Count')
    # plt.title("Atom number distribution in tox21 dataset")
    plt.tight_layout()
    plt.savefig("atom_num_dist_tox21.png",dpi=600)

    plt.show()
    plt.close()

# read_tox21("../data/tox21.csv")

def uniformRandomRotation():
    """
    Return a uniformly distributed rotation 3 x 3 matrix

    The initial description of the calculation can be found in the section 5 of "How to generate random matrices from
    the classical compact groups" of Mezzadri (PDF: https://arxiv.org/pdf/math-ph/0609050.pdf; arXiv:math-ph/0609050;
    and NOTICES of the AMS, Vol. 54 (2007), 592-604). Sample code is provided in that section as the ``haar_measure``
    function.

    Apparently this code can randomly provide flipped molecules (chirality-wise), so a fix found in
    https://github.com/tmadl/sklearn-random-rotation-ensembles/blob/5346f29855eb87241e616f6599f360eba12437dc/randomrotation.py
    was applied.

    Returns
    -------
    M : np.ndarray
        A uniformly distributed rotation 3 x 3 matrix
    """
    q, r = np.linalg.qr(np.random.normal(size=(3, 3)))
    M = np.dot(q, np.diag(np.sign(np.diag(r))))
    if np.linalg.det(M) < 0:  # Fixing the flipping
        M[:, 0] = -M[:, 0]  # det(M)=1
    return M

def rotate(coords, rotMat, center=(0,0,0)):
    """
    Rotate a selection of atoms by a given rotation around a center
    """
    newcoords = coords - center
    return np.dot(newcoords, np.transpose(rotMat)) + center

def translate(center, displacement=2.):
    center = center + (np.random.rand(3) - 0.5) * 2 * displacement
    return center

def read_hiv(raw_filename):
    smiles_tasks_df = pd.read_csv(raw_filename)
    smilesList = smiles_tasks_df.smiles.values
    print("number of all smiles:", len(smilesList))
    atom_num_distribution = []
    remained_smiles = []
    canonical_smiles_list = []
    for smiles in smilesList:
        try:
            mol = Chem.MolFromSmiles(smiles)
            atom_num_distribution.append(len(mol.GetAtoms()))
            remained_smiles.append(smiles)
            canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
        except:
            print("not successfully processed smiles: ", smiles)
            pass

    print("number of successfully processed smiles: ", len(remained_smiles))
    smiles_tasks_df = smiles_tasks_df[smiles_tasks_df["smiles"].isin(remained_smiles)]
    smiles_tasks_df['cano_smiles'] =canonical_smiles_list
    aa=np.array(atom_num_distribution)   #(2, 222)
    plt.figure(figsize=(5, 3))
    sns.set(font_scale=0.8)
    ax = sns.distplot(atom_num_distribution, bins=28, kde=False, color='red')
    ax.set_title("Distribution of atomic number in HIV dataset", fontdict={'color':'black', 'family':'Times New Roman'})
    ax.set_xlabel("Number of atoms", fontdict={'color':'black', 'family':'Times New Roman'})
    ax.set_ylabel("Molecule Count", fontdict={'color':'black', 'family':'Times New Roman'})

    # plt.xlabel('Atom Number',fontdict={'color': 'black',
    #                          'family': 'STFangsong',
    #                          'weight': 'normal',
    #                          'size': 15})
    # plt.ylabel('Molecule Count')
    # plt.title("Atom number distribution in tox21 dataset")
    plt.tight_layout()
    plt.savefig("atom_num_dist_HIV.png",dpi=600)

    plt.show()
    plt.close()

# read_hiv("../data/preprocess_HIV.csv")

def _load_tox21_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
       'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values
