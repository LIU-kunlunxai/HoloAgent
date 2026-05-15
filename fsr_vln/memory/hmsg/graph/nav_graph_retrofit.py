"""
Retrofit an existing nav_graph JSON with meta fields (cell_size, pcd_min).

Usage:
    python nav_graph_retrofit.py /path/to/graph/nav_graph/global_nav_graph_graph.json
"""

import json
import sys

import numpy as np
import open3d as o3d


def retrofit(json_path: str, cell_size: float = 0.03) -> None:
    with open(json_path) as f:
        data = json.load(f)

    if "meta" in data:
        print("Already has meta, skipping.")
        return

    # Compute pcd_min from the node positions
    positions = np.array([n["pos"] for n in data["nodes"]])
    pcd_min = positions.min(axis=0)

    data["meta"] = {
        "cell_size": cell_size,
        "pcd_min": pcd_min.tolist(),
    }

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Added meta — cell_size={cell_size}, pcd_min={pcd_min.tolist()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    retrofit(sys.argv[1])
