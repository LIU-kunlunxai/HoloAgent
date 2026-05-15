"""
Navigation graph query utility.

Loads a saved navigation graph and supports nearest-node lookup, shortest-path
planning, reachability checks, and path visualization on the top-down map.

Usage:
    nav = NavGraphQuery.load("path/to/graph/nav_graph/")
    path = nav.shortest_path(start_xyz, goal_xyz)
    nav.visualize(path, save="result.png")
"""

from dataclasses import dataclass
import json
import os
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.spatial import KDTree


@dataclass
class PathResult:
    waypoints: np.ndarray
    length: float
    node_ids: List[Tuple[int, int, str]]


class NavGraphQuery:
    """Query interface for the Voronoi-based navigation graph."""

    def __init__(
        self,
        graph: nx.Graph,
        cell_size: float,
        pcd_min: np.ndarray,
        top_down_map: Optional[np.ndarray] = None,
    ):
        self.graph = graph
        self.cell_size = cell_size
        self.pcd_min = pcd_min
        self.top_down_map = top_down_map

        self._nodes = list(graph.nodes)
        self._positions = np.array([graph.nodes[n]["pos"] for n in self._nodes])
        self._tree = KDTree(self._positions)

    # ---- factory ----------------------------------------------------------

    @classmethod
    def load(cls, nav_dir: str, cell_size: float = None, pcd_min: np.ndarray = None) -> "NavGraphQuery":
        """Load the nav graph and meta info from *nav_dir*.

        *nav_dir* should contain *global_nav_graph_graph.json* (and
        optionally *top_down_rgb.png*).

        If the JSON was saved by older code without meta fields, pass
        *cell_size* and *pcd_min* explicitly.
        """
        graph_path = os.path.join(nav_dir, "global_nav_graph_graph.json")
        with open(graph_path) as f:
            data = json.load(f)

        graph = nx.node_link_graph(data, edges="links")

        meta = data.get("meta", {})
        cell_size = cell_size or meta.get("cell_size")
        if pcd_min is not None:
            pcd_min = np.asarray(pcd_min)
        elif meta.get("pcd_min") is not None:
            pcd_min = np.array(meta["pcd_min"])
        else:
            pcd_min = np.array([])

        if cell_size is None or len(pcd_min) == 0:
            raise ValueError(
                "JSON missing 'meta' fields (cell_size / pcd_min). "
                "Either re-run with the updated code, or pass cell_size= / pcd_min= manually."
            )

        top_down = None
        top_down_path = os.path.join(nav_dir, "top_down_rgb.png")
        if os.path.exists(top_down_path):
            try:
                import cv2
                top_down = cv2.imread(top_down_path)
                top_down = cv2.cvtColor(top_down, cv2.COLOR_BGR2RGB)
            except ImportError:
                pass  # cv2 not available; visualize() will raise later

        return cls(graph, cell_size, pcd_min, top_down)

    # ---- coordinate helpers -----------------------------------------------

    def world_to_pixel(self, pos: np.ndarray) -> np.ndarray:
        """Convert a 3D world coordinate *(x, z)* to pixel *(col, row)*."""
        x, z = pos[0], pos[2]
        col = int((x - self.pcd_min[0]) / self.cell_size)
        row = int((z - self.pcd_min[2]) / self.cell_size)
        return np.array([col, row])

    # ---- queries ----------------------------------------------------------

    def find_nearest_node(self, xyz: np.ndarray) -> Tuple[int, int, str]:
        """Return the graph node id closest to *xyz* in world coordinates."""
        _, idx = self._tree.query(xyz.reshape(1, 3))
        return self._nodes[idx[0]]

    def shortest_path(
        self, start_xyz: np.ndarray, goal_xyz: np.ndarray
    ) -> PathResult:
        """Plan the shortest path between two 3D world coordinates."""
        src = self.find_nearest_node(start_xyz)
        dst = self.find_nearest_node(goal_xyz)
        node_ids = nx.shortest_path(self.graph, src, dst, weight="dist")
        waypoints = np.array([self.graph.nodes[n]["pos"] for n in node_ids])
        length = nx.shortest_path_length(self.graph, src, dst, weight="dist")
        return PathResult(waypoints=waypoints, length=length, node_ids=node_ids)

    def is_reachable(self, start_xyz: np.ndarray, goal_xyz: np.ndarray) -> bool:
        """Check whether *goal_xyz* is reachable from *start_xyz*."""
        src = self.find_nearest_node(start_xyz)
        dst = self.find_nearest_node(goal_xyz)
        return nx.has_path(self.graph, src, dst)

    # ---- visualization ----------------------------------------------------

    def visualize(
        self,
        path: PathResult,
        save: str = "nav_path_result.png",
        show_all_edges: bool = True,
    ) -> None:
        """Draw *path* on the top-down map and save to *save*."""
        if self.top_down_map is None:
            raise RuntimeError("No top_down_rgb.png found in nav_dir")

        import matplotlib.pyplot as plt

        h, w = self.top_down_map.shape[:2]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(self.top_down_map, origin="lower", extent=(0, w, 0, h))

        if show_all_edges:
            for u, v in self.graph.edges:
                p1 = self.world_to_pixel(
                    np.array(self.graph.nodes[u]["pos"]))
                p2 = self.world_to_pixel(
                    np.array(self.graph.nodes[v]["pos"]))
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        color="gray", linewidth=0.3, alpha=0.4)

        for i in range(len(path.waypoints) - 1):
            p1 = self.world_to_pixel(path.waypoints[i])
            p2 = self.world_to_pixel(path.waypoints[i + 1])
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color="red", linewidth=2, alpha=0.9)

        start_px = self.world_to_pixel(path.waypoints[0])
        goal_px = self.world_to_pixel(path.waypoints[-1])
        ax.scatter(*start_px, color="lime", s=80, edgecolors="black", label="Start")
        ax.scatter(*goal_px, color="cyan", s=80, edgecolors="black", label="Goal")

        ax.set_title(
            f"Shortest path  |  {len(path.waypoints)} nodes  |  {path.length:.2f} m")
        ax.legend()
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(save, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved visualization to {save}")
