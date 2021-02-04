import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from enzynet.PDB import PDB_backbone
from enzynet.volume import adjust_size, coords_to_volume, coords_center_to_zero, weights_to_volume
import argparse
import pandas as pd
import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
import rdkit.Geometry

class Grid3D:
    def __init__(self, xmin, ymin, zmin, xmax, ymax, zmax, resolution):
        assert(xmin < xmax and ymin < ymax and zmin < zmax and resolution > 0)
        self.xmin = np.float(xmin)
        self.ymin = np.float(ymin)
        self.zmin = np.float(zmin)
        self.xmax = np.float(xmax)
        self.ymax = np.float(ymax)
        self.zmax = np.float(zmax)
        self.resolution = np.float(resolution)
        self.nx = int((abs(self.xmax - self.xmin) / self.resolution)) + 1
        self.ny = int((abs(self.ymax - self.ymin) / self.resolution)) + 1
        self.nz = int((abs(self.zmax - self.zmin) / self.resolution)) + 1

    def contains_index(self, indices):
        if indices[0] < 0 or indices[0] >= self.nx:
            return False
        if indices[1] < 0 or indices[1] >= self.ny:
            return False
        if indices[2] < 0 or indices[2] >= self.nz:
            return False
        return True

    def get_bounding_box(self, point, radius):
        x_upper = np.floor(((point[0] - self.xmin) + radius) / self.resolution)
        x_lower = np.ceil(((point[0] - self.xmin) - radius) / self.resolution)
        y_upper = np.floor(((point[1] - self.ymin) + radius) / self.resolution)
        y_lower = np.ceil(((point[1] - self.ymin) - radius) / self.resolution)
        z_upper = np.floor(((point[2] - self.zmin) + radius) / self.resolution)
        z_lower = np.ceil(((point[2] - self.zmin) - radius) / self.resolution)
        return x_upper, x_lower, y_upper, y_lower, z_upper, z_lower

    def get_grid_points_in_radius(self, point, radius):
        x_upper, x_lower, y_upper, y_lower, z_upper, z_lower = self.get_bounding_box(point, radius)
        p = np.array(point)
        return [(x, y, z)
                for x in np.arange(x_lower, x_upper + 1)
                for y in np.arange(y_lower, y_upper + 1)
                for z in np.arange(z_lower, z_upper + 1)
                if np.linalg.norm(p - [self.xmin + (x * self.resolution),
                                       self.ymin + (y * self.resolution),
                                       self.zmin + (z * self.resolution)]) <= radius
                ]

def adjust_size(coords, v_size=48, max_radius=20):
    return np.multiply((v_size/2-1)/max_radius, coords)

# --input ../../data/esol_v2.csv
def visualize_pdb(p=5, v_size=48, num=1, weights=None, max_radius=20, noise_treatment=True):
    parser = argparse.ArgumentParser(description='visual')
    parser.add_argument('--input', type=str, required=True, help='input files to process')
    args = parser.parse_args()
    filename = args.input
    considered_elements = ['Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S']  # correspond to grid channels
    elem_indice_map = {e: i for i, e in enumerate(considered_elements)}
    grid = Grid3D(xmin=-11.5, ymin=-11.5, zmin=-11.5, xmax=12, ymax=12, zmax=12, resolution=0.5)

    esol = pd.read_csv(filename, float_precision='round_trip')
    esol_smiles = esol['smiles'].tolist()
    y = esol['measured log solubility in mols per litre'].tolist()

    m = rdkit.Chem.MolFromSmiles(esol_smiles[0])   # smiles='OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O'
    m = rdkit.Chem.AddHs(m)
    re = rdkit.Chem.AllChem.EmbedMolecule(m, rdkit.Chem.AllChem.ETKDG())

    # move mols centroid to the origin
    center = rdkit.Chem.rdMolTransforms.ComputeCentroid(m.GetConformers()[0])
    rdkit.Chem.rdMolTransforms.CanonicalizeConformer(m.GetConformers()[0], center=center, ignoreHs=True)

    conf = m.GetConformers()[0]  # with a single conformer only
    mol_descriptors = []
    for atom_id in range(conf.GetNumAtoms()):
        atom = m.GetAtomWithIdx(atom_id)

        a_sym = atom.GetSymbol()
        if a_sym not in elem_indice_map.keys():
            continue


        atom_pt = conf.GetAtomPosition(atom_id)
        atom_pt = adjust_size(atom_pt, v_size, max_radius)
        radius = rdkit.Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())

        px = grid.get_grid_points_in_radius(atom_pt, radius)
        for point in px:
            if grid.contains_index(point):
                mol_descriptors.append(point + (elem_indice_map[a_sym],))

    mol_descriptors = list(set(mol_descriptors))  # remove duplicates
    mol_descriptors = np.array(mol_descriptors, dtype=np.int8)

    color = 9
    volume = np.zeros((v_size, v_size, v_size, color))
    coords = mol_descriptors[:,0:4]
    for matrix_ind in coords:
        volume[tuple(matrix_ind)] = 1

    plot_volume(volume, v_size, num, weights=weights)   # Plot

def plot_volume(volume, v_size, num, weights=None):
    'Plots volume in 3D, interpreting the coordinates as voxels'
    # Initialization
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig = plt.figure(figsize=(4,4))
    ax = fig.gca(projection='3d')
    # ax.set_aspect('equal')

    # Parameters
    len_vol = volume.shape[0]

    # Set position of the view
    ax.view_init(elev=20, azim=135)

    # Hide tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    # ax.set_xlabel('X', fontsize=15)
    # ax.set_ylabel('Y', fontsize=15)
    # ax.set_zlabel('Z', fontsize=15)

    # Plot
    if weights == None:
        plot_matrix(ax, volume)
    else:
        plot_matrix_of_weights(ax, volume)

    # Tick at every unit
    ax.set_xticks(np.arange(len_vol))
    ax.set_yticks(np.arange(len_vol))
    ax.set_zticks(np.arange(len_vol))


    # Min and max that can be seen
    ax.set_xlim(0, len_vol-1)
    ax.set_ylim(0, len_vol-1)
    ax.set_zlim(0, len_vol-1)

    # Clear grid
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Change thickness of grid
    ax.xaxis._axinfo["grid"]['linewidth'] = 0.1
    ax.yaxis._axinfo["grid"]['linewidth'] = 0.1
    ax.zaxis._axinfo["grid"]['linewidth'] = 0.1

    # Change thickness of ticks
    ax.xaxis._axinfo["tick"]['linewidth'] = 0.1
    ax.yaxis._axinfo["tick"]['linewidth'] = 0.1
    ax.zaxis._axinfo["tick"]['linewidth'] = 0.1

    # Change tick placement
    ax.xaxis._axinfo['tick']['inward_factor'] = 0
    ax.xaxis._axinfo['tick']['outward_factor'] = 0.2
    ax.yaxis._axinfo['tick']['inward_factor'] = 0
    ax.yaxis._axinfo['tick']['outward_factor'] = 0.2
    ax.zaxis._axinfo['tick']['inward_factor'] = 0
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.2
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.2

    # Save
    # plt.savefig('./delaney' + '_' + str(v_size) + '_' + str(weights) + '_' + str(num) + '.pdf')
    plt.savefig('aaaa.pdf')

def cuboid_data(pos, size=(1,1,1)):
    'Gets coordinates of cuboid'
    # Gets the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(pos, size)]

    # Get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]] for i in range(4)]
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],
         [o[1], o[1], o[1], o[1], o[1]],
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]
    z = [[o[2], o[2], o[2], o[2], o[2]],
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]

    return x, y, z

def plot_cube_at(pos=(0,0,0), ax=None, color='b'):
    'Plots a cube element at position pos'
    if ax != None:
        X, Y, Z = cuboid_data(pos)
        ax.plot_surface(np.array(X), np.array(Y), np.array(Z), color=color, rstride=1, cstride=1, alpha=1)

def plot_cube_weights_at(pos=(0,0,0), ax=None, color='g'):
    'Plots a cube element at position pos'
    if ax != None:
        X, Y, Z = cuboid_data(pos)
        ax.plot_surface(X, Y, Z, color=color, rstride=1, cstride=1, alpha=1)


def random_color(number=9):
    color = []
    intnum = [str(x) for x in np.arange(10)]
    # Out[138]: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    alphabet = [chr(x) for x in (np.arange(6) + ord('A'))]
    # Out[139]: ['A', 'B', 'C', 'D', 'E', 'F']
    colorArr = np.hstack((intnum, alphabet))
    # Out[142]: array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C','D', 'E', 'F'], dtype='<U1')
    for j in range(number):
        color_single = '#'
        for i in range(6):
            index = np.random.randint(len(colorArr))
            color_single += colorArr[index]
        # Out[148]: '#EDAB33'
        color.append(color_single)
    return color


def plot_matrix(ax, matrix):
    colors =random_color(9)
    'Plots cubes from a volumic matrix'
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(matrix.shape[2]):
                for p in range(matrix.shape[3]):
                    if matrix[i,j,k,p] == 1:
                        if p==0:
                            print("p=0 " +colors[p])
                        elif p==1:
                            print("p=1 " + colors[p])
                        elif p==2:
                            print("p=2 " + colors[p])
                        elif p==3:
                            print("p=3 " + colors[p])
                        elif p==4:
                            print("p=4 " + colors[p])
                        elif p==5:
                            print("p=5 " + colors[p])
                        elif p==6:
                            print("p=6 " + colors[p])
                        elif p==7:
                            print("p=7 " + colors[p])
                        elif p==8:
                            print("p=8 " + colors[p])
                        plot_cube_at(pos=(i-0.5,j-0.5,k-0.5), ax=ax, color = colors[p])

def plot_matrix_of_weights(ax, matrix_of_weights):
    'Plots cubes from a volumic matrix'
    # Initialization
    min_value = np.amin(matrix_of_weights)
    max_value = np.amax(matrix_of_weights)
    n_colors = 101

    # Check if matrix of weights or not
    if min_value == max_value == 1:
        return plot_matrix(ax, matrix_of_weights)

    # Generate colors
    cm = plt.get_cmap('seismic')
    cgen = [cm(1.*i/n_colors) for i in range(n_colors)]

    # Plot cubes
    for i in range(matrix_of_weights.shape[0]):
        for j in range(matrix_of_weights.shape[1]):
            for k in range(matrix_of_weights.shape[2]):
                if matrix_of_weights[i,j,k] != 0:
                    # Translate to [0,100]
                    normalized_weight = (matrix_of_weights[i,j,k] - min_value)/ \
                                        (max_value - min_value)
                    normalized_weight = int(100*normalized_weight)

                    # Plot cube with color
                    plot_cube_weights_at(pos=(i-0.5,j-0.5,k-0.5), ax=ax,
                                         color=cgen[normalized_weight])

if __name__ == '__main__':
    visualize_pdb(p=0, v_size=64, weights=None)
