import sys
sys.path.append(r'/home/lichunyan/project/interpretable_ml_drug_discovery/3DGCN/')
from grid.gridtrainer import GridTrainer

if __name__ == "__main__":
    trainer = GridTrainer("freesolv")

    hyperparameters = {"epoch": 200, "batch": 5, "fold": 10, "units_conv_coord": 64,  "units_conv_atom": 64, "units_dense": 128, "chebyshev_order": 2, "pooling": "sum",
                       "num_layers": 2, "loss": "mse", "monitor": "val_rmse", "label": "", "isattention": True}

    features = {"use_atom_symbol": True, "use_degree": True, "use_hybridization": True, "use_implicit_valence": True,
                "use_partial_charge": True, "use_ring_size": True, "use_hydrogen_bonding": True,
                "use_acid_base": True, "use_aromaticity": True, "use_chirality": True, "use_num_hydrogen": True}

    # Baseline
    trainer.fit_grid("model_3D_grid", **hyperparameters, **features)