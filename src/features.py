import numpy as np
import pandas as pd
import cajal

def compute_gw_embedding(gw_dist_dict, cells):
    """
    Convert CAJAL GW distances → embedding (UMAP-ready matrix)
    """
    gw_dist = cajal.utilities.dist_mat_of_dict(gw_dist_dict, cells)

    import umap
    reducer = umap.UMAP(metric="precomputed", random_state=42)
    embedding = reducer.fit_transform(gw_dist)

    return embedding, gw_dist
