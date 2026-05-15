"""
测试导航图路径规划效果。

用法:
    python scripts/test_nav_graph.py <nav_graph目录>

示例:
    python scripts/test_nav_graph.py /home/liu/workspace/recorded_scene/icra_ic3f/fsr/graph/nav_graph

    # 指定起终点
    python scripts/test_nav_graph.py <nav_graph目录> --start 1.5,0,4.5 --goal 3.2,0,6.1

    # 交互模式(在图上点选起终点)
    python scripts/test_nav_graph.py <nav_graph目录> --interactive

依赖: networkx, numpy, scipy, opencv-python, matplotlib
"""

import argparse
import os
import sys

# 确保项目在 Python 路径中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_project_root, "fsr_vln"))

import json
import numpy as np
from memory.hmsg.graph.nav_graph_query import NavGraphQuery


def print_info(nav: NavGraphQuery):
    """打印导航图基本信息."""
    g = nav.graph
    positions = np.array([g.nodes[n]["pos"] for n in g.nodes])
    print(f"节点数: {len(g.nodes)}")
    print(f"边数:   {len(g.edges)}")
    print(f"X 范围: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}]")
    print(f"Y (高度): [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}]")
    print(f"Z 范围: [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}]")
    print(f"cell_size: {nav.cell_size}")
    print(f"pcd_min:   {nav.pcd_min.round(2)}")


def run_test(nav: NavGraphQuery, start: np.ndarray, goal: np.ndarray, save: str):
    """执行路径规划并可视化."""
    print(f"\n起点 (world): {start}")
    print(f"终点 (world): {goal}")

    src_node = nav.find_nearest_node(start)
    dst_node = nav.find_nearest_node(goal)
    print(f"起点最近节点: {src_node}  pos={nav.graph.nodes[src_node]['pos'].round(2)}")
    print(f"终点最近节点: {dst_node}  pos={nav.graph.nodes[dst_node]['pos'].round(2)}")

    if not nav.is_reachable(start, goal):
        print("ERROR: 两点不可达!")
        return

    path = nav.shortest_path(start, goal)
    print(f"路径节点数: {len(path.waypoints)}")
    print(f"总距离:     {path.length:.2f} m")

    nav.visualize(path, save=save)
    print(f"可视化保存到: {save}")


def interactive(nav: NavGraphQuery):
    """交互模式: 命令行输入起终点坐标."""
    print("\n--- 交互模式 ---")
    print("输入起终点坐标 (格式: x,y,z), 输入 q 退出")
    while True:
        try:
            s = input("\n起点 (x,y,z): ").strip()
            if s.lower() == "q":
                break
            start = np.array([float(v) for v in s.split(",")])
            s = input("终点 (x,y,z): ").strip()
            if s.lower() == "q":
                break
            goal = np.array([float(v) for v in s.split(",")])
        except (ValueError, EOFError):
            print("格式错误, 重试")
            continue
        run_test(nav, start, goal, "nav_result.png")


def main():
    parser = argparse.ArgumentParser(description="测试导航图路径规划")
    parser.add_argument("nav_dir", help="导航图目录 (含 global_nav_graph_graph.json)")
    parser.add_argument("--start", help="起点坐标 x,y,z")
    parser.add_argument("--goal", help="终点坐标 x,y,z")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    parser.add_argument("--save", default="nav_result.png", help="输出图片路径")
    parser.add_argument("--cell_size", type=float, default=None,
                        help="cell_size (仅旧数据需要)")
    args = parser.parse_args()

    # ---- 处理旧数据缺少 meta 的情况 ----
    json_path = os.path.join(args.nav_dir, "global_nav_graph_graph.json")
    if not os.path.exists(json_path):
        print(f"ERROR: 找不到 {json_path}")
        sys.exit(1)

    with open(json_path) as f:
        data = json.load(f)
    has_meta = "meta" in data

    if not has_meta:
        print("JSON 缺少 meta 字段, 正在自动补全...")
        cell_size = args.cell_size or 0.03
        positions = np.array([n["pos"] for n in data["nodes"]])
        pcd_min = positions.min(axis=0)
        data["meta"] = {"cell_size": cell_size, "pcd_min": pcd_min.tolist()}
        backup = json_path + ".bak"
        os.rename(json_path, backup)
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"已更新 {json_path} (原文件备份为 .bak)")

    # ---- 加载 ----
    nav = NavGraphQuery.load(args.nav_dir)
    print_info(nav)

    if args.interactive:
        interactive(nav)
    elif args.start and args.goal:
        start = np.array([float(v) for v in args.start.split(",")])
        goal = np.array([float(v) for v in args.goal.split(",")])
        run_test(nav, start, goal, args.save)
    else:
        # 默认: 选最远的两个节点做 demo
        positions = np.array([nav.graph.nodes[n]["pos"] for n in nav.graph.nodes])
        d = np.linalg.norm(positions - positions[0], axis=1)
        far_idx = np.argmax(d)
        start = positions[0]
        goal = positions[far_idx]
        print("\n未指定起终点, 自动选最远的两个节点做 demo:")
        run_test(nav, start, goal, args.save)


if __name__ == "__main__":
    main()
