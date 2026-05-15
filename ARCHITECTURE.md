# HoloAgent 架构文档

## 1. 项目概述

HoloAgent 是一个面向通用机器人的统一 Agent 框架，核心模块 **FSR-VLN**（Fast-to-Slow Reasoning for Vision-Language Navigation）通过**分层多模态场景图 (HMSG)** 与 **快慢推理机制**，实现自然语言驱动的空间目标搜索与导航。

整体流程：

```
传感器数据 → 3D重建 → 语义分割+特征提取 → 分层场景图(楼层/房间/物体) → 导航图(Voronoi) → VLN查询
```

---

## 2. 目录结构

```
HoloAgent/
├── fsr_vln/                          # 核心：视觉语言导航 & 场景图引擎
│   ├── memory/hmsg/                  #   分层多模态场景图 (HMSG)
│   │   ├── graph/                    #     图构建核心（Graph, Floor, Room, Object, View, NavigationGraph）
│   │   ├── dataloader/               #     多数据源加载器（Replica, ScanNet, HM3D, Horizon, iPhone）
│   │   ├── utils/                    #     工具函数（CLIP, SAM, graph/spatial/eval utils, LLM parsing）
│   │   ├── eval/                     #     评估（HM3DSemanticEvaluator）
│   │   ├── labels/                   #     语义标签映射（Matterport, ScanNet200, ImageNet21k）
│   │   └── data/                     #     数据辅助
│   ├── perception/models/            #   感知：SAM 分割 + CLIP 特征提取
│   ├── application/                  #   应用入口
│   │   ├── semantic_scene_reconstrucion_offline/  # 离线建图：重建 → 分割 → 构建 HMSG
│   │   └── visualize_query_graph/                 # VLN 查询可视化（各场景的 query 脚本）
│   ├── config/                       #   Hydra 配置文件
│   └── rgbd_datasets/                #   原始 RGB-D 数据存放目录
│
├── nav_agent/                        # 导航执行层：机器人控制 & 定位
│   ├── humble_localization_nav2/     #   ROS 2 定位/导航/避障 (Docker 内运行)
│   │   ├── lio_mapping_loc/          #     FastLIVO2 激光里程计 + 地图重定位
│   │   ├── g1_navigation2/           #     Nav2 导航参数接口
│   │   ├── g1_nav_bringup/           #     一键启动所有 launch
│   │   ├── pubpose/                  #     接收目标位姿 → 转发 Nav2 做全局导航与避障
│   │   ├── navigation2-humble/       #     ROS 2 Navigation2 框架
│   │   └── rpg_vikit-ros2/           #     FastLIVO2 第三方依赖
│   ├── sem_nav_ctr/                  #   语义导航控制 (宿主机运行)
│   │   ├── goal_publisher/           #     语义目标定位 → 调用 FSR-VLN 查询目标坐标
│   │   ├── chat_loc_python/          #     语音交互客户端
│   │   └── g1_move/                  #     机器人运控接口
│   └── scripts/                      #   一键启动脚本 (run_nav / run_sem_nav / run_sensors)
│
├── tools/pocket2_toolbox/            # 数据预处理工具箱
│   ├── pocket2_locnav_pipeline/      #   点云 → PLY 转换、关键帧提取、网格地图生成
│   └── pocket2_rgbd_gen_pipeline/    #   点云投影生成 RGB-D 图像、相机位姿插值
│
├── env/                              # 仿真环境
│   ├── sim/habitat_sim/              #   Habitat Sim 场景仿真 (HM3DSem 数据生成)
│   └── robot/                        #   机器人传感器定义
│
├── docs/                             # 文档资源 (Logo、框架图等)
└── outputs/                          # 运行输出（Hydra 日志等）
```

---

## 3. FSR-VLN 核心模块详解

### 3.1 总体架构

FSR-VLN = **HMSG（分层多模态场景图）** + **FSR（快慢推理）**

```
用户指令 "Find me a bottle in the Exhibition Hall"
        │
        ▼
  FSR 推理 (graph.py: query_hierarchy_protected_icra)
        │
        ├── Fast Matching:   CLIP 特征粗匹配 Top-K 候选
        ├── Object Check:    检查物体是否在相机视野中
        ├── VLM Rethinking:  VLM 精细确认（GPT）
        └── Re-Matching:     特征重新匹配
        │
        ▼
  结果: (Floor, Room, Object) → 物体的 3D 坐标 → 发送给 nav_agent
```

### 3.2 Graph（场景图核心）

文件: `fsr_vln/memory/hmsg/graph/graph.py`

**Graph 类** 是整个场景理解的顶层入口，包含所有建图和查询方法。

#### 图节点类型

| 类 | 文件 | 含义 |
|---|---|---|
| `Floor` | `graph/floor.py` | 楼层节点：包含点云、高度直方图、所属 Room 列表 |
| `Room` | `graph/room.py` | 房间节点：包含点云、CLIP 特征、所属 Object 列表、名称 |
| `Object` | `graph/object.py` | 物体节点：包含点云、CLIP 特征、语义标签、3D 包围盒 |
| `View` | `graph/view.py` | 视角节点：相机位姿、对应帧号、全景/深度数据 |

#### 核心方法

**建图阶段：**

| 方法 | 功能 |
|---|---|
| `build_hier_multimodal_scene_graph()` | **主入口**：完整构建分层场景图 |
| `create_feature_map()` | 对每帧图像运行 SAM 分割 + CLIP 提取特征 |
| `segment_floors()` | 通过高度直方图聚类分割楼层 |
| `segment_rooms()` | 通过距离变换 + 分水岭算法分割房间 |
| `segment_objects()` | 对每个房间融合所有帧的 SAM mask，合并去重物体 |
| `create_nav_graph()` | 构建 Voronoi 导航图（见 3.3 节） |
| `save_hmsg_graph()` | 保存完整场景图为 JSON + PLY |

**查询阶段（FSR 快慢推理）：**

| 方法 | 功能 |
|---|---|
| `query_hierarchy_protected_icra()` | ICRA 版分层查询（最常用） |
| `query_hierarchy_protected()` | 通用版分层查询 |
| `query_hierarchy()` | 基础版分层查询 |
| `query_floor()` | CLIP 匹配确定目标楼层 |
| `query_room()` | CLIP 匹配确定目标房间 |
| `query_object()` | CLIP 匹配确定目标物体 |
| `query_room_obj_slow_reasoning()` | VLM 慢推理精确认证 |

### 3.3 导航图 (NavigationGraph)

文件: `fsr_vln/memory/hmsg/graph/navigation_graph.py`

**NavigationGraph 类** 负责从楼层点云构建可导航空间的 Voronoi 骨架图。

#### 构建流程

```
楼层点云
  │
  ├── create_occupancy_grid()     → 2D 占据栅格地图
  ├── get_main_free_map()         → 可通行区域（减去障碍物）
  ├── get_voronoi_graph()         → 稠密 Voronoi 图 (scipy.spatial.Voronoi)
  ├── sparsify_graph()            → Dijkstra 稀疏化（保留路口+死胡同）
  ├── trim_graph()                → 裁剪无效 dead-end
  ├── get_stairs_graph()          → 检测楼梯子图
  ├── connect_stairs_and_floor_graphs() → 连接楼梯与楼层图
  └── connect_voronoi_graphs()    → 连接相邻楼层
  │
  ▼
global_nav_graph_graph.json  (多层联合导航图)
```

#### 输出文件

保存路径: `<save_path>/graph/nav_graph/`

| 文件 | 含义 |
|---|---|
| `global_nav_graph_graph.json` | **最终多层联合导航图** |
| `sparse_voronoi_graph.json` | 单楼层（含楼梯）导航图 |
| `top_down_rgb.png` | 俯视 RGB 地图 |
| `top_down_rgb_poses.png` | 俯视图 + 相机位姿标记 |
| `floor_free.png` | 可通行区域（白=可行走，黑=障碍物） |
| `floor_obstacles.png` | 障碍物分布 |
| `sparse_vor_rgb.png` | 稀疏 Voronoi 图叠加俯视图 |
| `vor_rgb_combined.png` | 楼层 Voronoi + 楼梯子图 + 俯视图 |
| `vor_rgb_combined_highlighted.png` | 同上，额外高亮楼梯连接点（绿点） |

#### 导航图数据结构

```json
{
  "directed": false,
  "multigraph": false,
  "nodes": [
    {
      "id": [row, col, "floor_id"],
      "pos": [x, y, z],
      "floor_id": "0"
    }
  ],
  "links": [
    {
      "source": [row1, col1, "floor_id1"],
      "target": [row2, col2, "floor_id2"],
      "dist": 0.45
    }
  ]
}
```

- **节点**: 稀疏化后的 Voronoi 关键点（路口 + 死胡同端点），含 3D 世界坐标
- **边**: 可导航路径，`dist` 为欧氏距离权重
- **格式**: NetworkX `node_link_data`，可直接用 `nx.node_link_graph(json, edges="links")` 加载

#### 加载与路径规划

```python
import networkx as nx, json, numpy as np
from scipy.spatial import KDTree

# 1. 加载图
with open("graph/nav_graph/global_nav_graph_graph.json") as f:
    G = nx.node_link_graph(json.load(f), edges="links")

# 2. 构建 KDTree 用于坐标→节点映射
nodes = list(G.nodes(data=True))
positions = np.array([n[1]["pos"] for n in nodes])
tree = KDTree(positions)

# 3. 查询起点/终点最近节点
src = nodes[tree.query(start_xyz)[1][0]][0]
dst = nodes[tree.query(goal_xyz)[1][0]][0]

# 4. 最短路径
path_nodes = nx.shortest_path(G, src, dst, weight="dist")
waypoints_3d = [G.nodes[n]["pos"] for n in path_nodes]

# 5. 路径长度
length = nx.shortest_path_length(G, src, dst, weight="dist")

# 6. 可达性判断
nx.has_path(G, src, dst)
```

> **注意:** 当前代码中 `nx.all_pairs_dijkstra_path` 已用于 `sparsify_graph()` (第651行) 做图的稀疏化，但尚未封装独立的导航查询接口。上述代码是加载已保存的图做路径规划的用法。

### 3.4 感知 (Perception)

文件: `fsr_vln/perception/models/sam_clip_feats_extractor.py`

- **SAM** (Segment Anything Model): 对每帧图像做 instance segmentation
- **CLIP** (OpenCLIP): 对每个分割出的 mask 计算 CLIP 视觉特征
- 输出: 每帧的 mask 列表 + 对应的 CLIP 特征向量

### 3.5 工具函数 (utils)

| 文件 | 功能 |
|---|---|
| `clip_utils.py` | CLIP 特征提取封装（图像/文本） |
| `sam_utils.py` | SAM mask 过滤、裁剪 |
| `graph_utils.py` | 空间计算（点云投影、KDTree、IoU、DBSCAN、距离变换、mask 合并） |
| `llm_utils.py` | LLM 查询解析（GPT 3.5 parse floor/room/object） |
| `eval_utils.py` | 3D IoU 计算、评估指标 |
| `label_feats.py` | 语义标签 → CLIP 文本特征映射 |
| `constants.py` | Matterport 标签、CLIP 维度常量 |
| `long_query_eval_utils.py` | 长查询评估 |

### 3.6 数据加载器 (Dataloader)

支持多种数据源，统一接口：

| 文件 | 数据源 |
|---|---|
| `replica.py` | Meta Replica 数据集 |
| `scannet.py` | ScanNet 数据集 |
| `hm3dsem.py` | Habitat-Matterport 3D Semantics |
| `horizon.py` | 地平线自采数据 (Horizon Robotics) |
| `iphone.py` | iPhone 拍摄数据 |
| `generic.py` | 通用加载器（Pocket2 数据处理） |

---

## 4. Application 使用流程

### 4.1 离线建图

**入口脚本**: `application/semantic_scene_reconstrucion_offline/semantic_scene_reconstruction.py`

```bash
cd fsr_vln
python application/semantic_scene_reconstrucion_offline/semantic_scene_reconstruction.py \
    --config-name=semantic_scene_reconstruction_ic3f
```

**流程:**

```
1. 加载 RGB-D 帧 + 相机位姿（Dataloader）
2. SAM 分割 → CLIP 特征提取 (create_feature_map)
3. 点云重建 → full_pcd.ply
4. 楼层分割 (segment_floors)
5. 房间分割 (segment_rooms / distance_transform + watershed)
6. 物体分割 + 合并去重 (segment_objects / hierarchical_merge)
7. 房间命名 (可选: generate_room_names)
8. 构建导航图 (create_nav_graph)
9. 保存场景图 (save_hmsg_graph)
```

**输出** (以 ICRA IC3F 场景为例):

```
recorded_scene/icra_ic3f/fsr/
├── full_pcd.ply                   # 完整场景点云
├── full_feats.pt                  # 逐点 CLIP 特征 (N, D)
├── masked_pcd.ply                 # 所有物体着色拼合的点云
├── mask_feats.pt                  # 每个物体的 CLIP 特征
├── objects/                       # 980 个独立物体点云 (pcd_0.ply ... pcd_979.ply)
├── graph_<timestamp>/             # 分层场景图
│   ├── floors/                    #   楼层点云 + JSON
│   ├── rooms/                     #   房间点云 + JSON
│   ├── objects/                   #   图物体节点
│   └── views/                     #   图视角节点
├── graph/nav_graph/               # 导航图
│   ├── global_nav_graph_graph.json
│   ├── sparse_voronoi_graph.json
│   ├── top_down_rgb.png
│   ├── sparse_vor_rgb.png
│   ├── vor_rgb_combined.png
│   └── ...
├── tmp/                           # 中间计算产物
│   ├── floor_histogram.png
│   ├── room_views.npz
│   └── 0/                         # floor 0 中间结果 (距离变换、墙体骨架等)
└── vln_result_presentation/       # VLN 查询结果（查询时填充）
```

### 4.2 VLN 查询可视化

**入口脚本**: `application/visualize_query_graph/visualize_query_graph_icra_ic3f.py`

```bash
cd fsr_vln
python application/visualize_query_graph/visualize_query_graph_icra_ic3f.py
```

**功能:**
1. 加载已建好的 HMSG 场景图
2. 对每条 query 指令运行 FSR 推理 (`query_hierarchy_protected_icra`)
3. 将查询结果（目标房间+物体的 3D 点云）可视化保存
4. 统计平均耗时（FastMatching, VLM Rethinking, LLM Parse 等）
5. 输出 `all_results.json` 到 `vln_result_presentation/`

---

## 5. 定位系统（双层架构）

HoloAgent 的定位系统分为两层，解决不同维度的问题，共享同一个地图坐标系：

| | 机器人定位 (Robot Localization) | 语义定位 (Semantic Localization) |
|---|---|---|
| **问题** | 我在哪个坐标？(x, y, z, yaw) | 我在哪个楼层/房间？目标物体在哪？ |
| **传感器** | LiDAR + IMU + 相机 | RGB 相机 + CLIP |
| **算法** | FastLIVO2 + ScanContext + NDT/ICP | CLIP 文本-图像余弦相似度匹配 |
| **频率** | 连续 (~10-50Hz) | 收到指令时按需触发 |
| **输出** | TF 变换 (map→odom→base_link) | floor_id, room_id, object 3D 坐标 |
| **代码** | `nav_agent/lio_mapping_loc/` | `fsr_vln/memory/hmsg/graph/graph.py` |

### 5.1 机器人定位 — "我在哪"

代码位置: `nav_agent/humble_localization_nav2/lio_mapping_loc/`

#### FastLIVO2（实时里程计）

`src/vio.cpp` 实现 LiDAR-惯性-视觉紧耦合 SLAM 前端，持续输出 6-DoF 位姿。

```
激光点云 + IMU + 相机图像
        │
        ▼
  VIOManager (vio.cpp)
        │
        ├── retrieveFromVisualSparseMap()   # 稀疏视觉特征地图检索 (line 2645)
        ├── ESIKF 状态估计                   # 误差状态迭代卡尔曼滤波
        └── 输出: /Odometry → 发布 map→odom TF
```

#### ScanContext（全局重定位）

`include/sc-relo/Scancontext.cpp` 将 3D 激光扫描压缩成 2D 矩阵（行=环，列=方位角），通过列移匹配找最相似的历史帧：

```cpp
detectLoopClosureID(num_exclude_recent)        // 回环检测 (line 650)
relocalize(target_sc)                            // 全局重定位 (line 707)
detectLoopClosureIDBetweenSession(...)          // 跨 session 回环 (line 337)
```

#### Online Relocalization（精确重定位）

`src/online_relocalization.cpp` + `include/online-relo/pose_estimator.cpp`：

```
ScanContext 粗定位（找回环候选帧）
    │
    ├── 构建 local map (候选帧邻域拼接)
    ├── NDT 或 ICP 精配准: current_cloud ←→ local_map
    └── 输出精确重定位位姿
```

核心函数 `pose_estimator::relocalization()` (pose_estimator.cpp:549)：
```cpp
// 根据配置选择配准方式
if (reg_mode_ == 0) ndt_success = ndt_pcl(input_cloud, localMap, initial_guess, transform);
if (reg_mode_ == 1) ndt_success = icp_pcl(input_cloud, localMap, initial_guess, transform);
```

#### 重定位的意义

机器人重启或被搬动后，当前坐标系和已建地图的坐标系不重合，语义查询返回的坐标就失去了物理意义。重定位系统通过 ScanContext（粗）+ NDT/ICP（精）将当前位姿对齐回地图坐标系，使语义查询结果恢复可用性。

#### 启动方式

```bash
# Docker 内：里程计模式 (持续跟踪)
ros2 launch fast_livo online_robot_odom.launch.py

# Docker 内：重定位模式 (找回在地图中的位置)
ros2 launch fast_livo online_reloc.launch.py
```

---

### 5.2 语义定位 — "目标在哪"

代码位置: `fsr_vln/memory/hmsg/graph/graph.py`

#### 建图阶段：给点云注入语义

`create_feature_map()` (graph.py:262)：

```
每一帧 RGB-D
    │
    ├── SAM 分割 → 每帧产生 ~50-100 个 2D mask
    │       "这片像素是一把椅子"、"那片像素是一张桌子"
    │
    ├── CLIP 编码 → 每个 mask → 768维视觉特征向量
    │       椅子 → [0.12, -0.34, 0.56, ...]
    │
    ├── 3D 投影 → 2D mask + 深度 → 3D mask 点云
    │
    ├── 点级特征融合 → full_feats.pt: 每个3D点的CLIP特征取多帧平均
    │
    ├── 物体合并去重 → hierarchical_merge: 跨帧同一物体合并
    │
    └── 物体特征融合 → mask_feats.pt: 每个最终物体的CLIP特征
```

#### 查询阶段：FSR 快慢推理

`query_hierarchy_protected_icra()` (graph.py:3491)：

```
用户输入: "Find me a stainless steel cup in the Exhibition Hall"
    │
    ├── Step 0: LLM 解析 (parse_hier_query)
    │     → floor_query="1", room_query="Exhibition Hall", object_query="stainless steel cup"
    │
    ├── Step 1: query_floor()
    │     "floor 1" CLIP文本特征 ←cos→ ["floor 0", "floor 1", ...] 文本特征 → floor_id
    │
    ├── Step 2: query_hmsg_room() / query_room()
    │     "Exhibition Hall" CLIP文本特征 ←cos→ 每个房间的代表性视觉特征 → top-k 房间
    │
    ├── Step 3: query_hmsg_object() [FAST]
    │     "stainless steel cup" CLIP文本特征 ←cos→ 候选房间内所有物体的视觉特征 → Top-5
    │
    ├── Step 4: detect_object_in_image()
    │     检查 Top-1 物体的 best_view 图像中目标是否可见
    │       可见 → 直接返回 ✓
    │       不可见 → 进入慢推理
    │
    └── Step 5: VLM 慢推理 [SLOW]
        │
        ├── CLIP 在所有视角中找 top-24 最相关图片
        ├── 发给 GPT 挑选最匹配目标描述的图片
        ├── GPT 确认后找到该视角下的最佳物体
        └── 物体点云投影回图片验证可见性
```

#### CLIP 匹配的核心机制

```python
# 余弦相似度 —— 整个语义系统的基石
text_feats  = CLIP.encode_text("stainless steel cup")     # (1, 768)
object_embs = np.array([obj.embedding for obj in objects])  # (N, 768)
sim_mat = np.dot(text_feats, object_embs.T)                 # cos(query, each_object)
top_k_idx = np.argsort(sim_mat[0])[::-1][:5]
```

CLIP 在大规模图文对上训练，所以"不锈钢杯"文本和一张真的不锈钢杯图片在特征空间中余弦相似度很高。这是零样本匹配的基础。

#### 快慢推理 (FSR) 对比

| | Fast Matching | Slow Reasoning |
|---|---|---|
| **算法** | CLIP 余弦相似度排序 | GPT VLM 看图选择 |
| **速度** | ~0.1s | ~2-5s |
| **准确度** | 可能匹配到相似类别 | 精细区分细节 |
| **触发条件** | 每次都跑 | Fast 结果不可靠时才触发 |

---

### 5.3 两层定位的协作关系

```
┌──────────────────────────────────────────────────────────┐
│              FastLIVO2 机器人定位 (连续, ~10-50Hz)          │
│                                                          │
│  提供: 当前位姿 (x,y,z,yaw), map→base_link TF              │
│  作用: Nav2 路径规划、局部避障                              │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │          FSR-VLN 语义定位 (按需触发)                │   │
│  │                                                   │   │
│  │  提供: 目标楼层/房间/物体的语义信息 + 3D坐标         │   │
│  │  作用: 把自然语言变成导航目标点                      │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  两者共用同一个地图参考系 (建图时由 FastLIVO2 建立)         │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
          Nav2: 从 (当前位置) → 规划路径 → (目标坐标)
```

**关键点：** 机器人定位回答"我在哪"，语义定位回答"目标在哪"，两者在同一个地图坐标系下，Nav2 负责连接这两个坐标。

---

## 6. Nav Agent（导航执行层）

### 6.1 系统架构

```
                    宿主机                            │              Docker
                                                      │
  chat_loc_python ───► goal_publisher ───┐              │
       (语音输入)        (语义目标→坐标)    │              │
                                          │              │
                                          ▼              │
                                    FSR-VLN HMSG         │
                                    (目标物体3D坐标)      │
                                          │              │
                                          ▼              │
                                    pubpose (目标位姿) ──► Nav2 (全局规划+局部避障)
                                                      │            │
                                                      │            ▼
                                                      │     FastLIVO2 (里程计)
```

### 6.2 各模块职责

| 模块 | 功能 |
|---|---|
| `lio_mapping_loc` | FastLIVO2 实时激光里程计 + 地图重定位，提供 `map→odom` TF |
| `g1_navigation2` | Nav2 参数配置（planner/controller/costmap 参数） |
| `pubpose` | 接收 `goal_publisher` 的目标位姿，转为 Nav2 目标，触发导航 |
| `goal_publisher` | **语义目标→3D坐标的核心桥梁**：调用 FSR-VLN 的 HMSG 查询 |
| `chat_loc_python` | 语音交互客户端（客户端采集，服务端处理） |
| `g1_move` | 机器人底层运控接口（行走/转向等基本动作） |

### 6.3 启动方式

```bash
# Docker 内：启动导航 + 定位 + 避障
bash nav_agent/scripts/run_nav.sh

# 宿主机：启动语义导航模块（语音 + 目标定位 + 运控）
bash nav_agent/scripts/run_sem_nav.sh

# 宿主机：启动传感器
bash nav_agent/scripts/run_sensors.sh
```

---

## 7. Tools（数据预处理）

### 7.1 pocket2_locnav_pipeline

| 脚本 | 功能 |
|---|---|
| `pcd2ply.py` | 原始 PCD 点云 → PLY 格式转换 + 降采样 |
| `pcd2pcd.py` | PCD 去噪、滤波 (statistical outlier removal) |
| `pcd2keyframe.py` | 从轨迹中提取关键帧 |
| `grid_map_gen.py` | 从点云生成 2D 占据栅格地图（障碍物/可通行/未知） |

### 7.2 pocket2_rgbd_gen_pipeline

| 脚本 | 功能 |
|---|---|
| `pcd_image_projection_parallel.py` | 点云投影生成 RGB-D 图像（并行版） |
| `pcd_image_projection_sequence.py` | 点云投影生成 RGB-D 图像（序列版，带 GUI） |
| `camera_pose_interpolation.py` | 相机轨迹平滑插值 |
| `undisort_test_pipeline.py` | 去畸变测试 |

---

## 8. 数据流总览

```
┌─────────────────────────────────────────────────────────────────┐
│                        离线建图 (Offline)                        │
│                                                                 │
│  RGB-D帧 + 位姿 (由 FastLIVO2 提供)                                │
│      │                                                          │
│      ├──► SAM 分割 ──► CLIP 特征 (perception)                    │
│      ├──► 点云重建 ──► full_pcd.ply + full_feats.pt              │
│      ├──► 高度聚类 ──► 楼层分割 (Floor)                          │
│      ├──► 距离变换+分水岭 ──► 房间分割 (Room)                      │
│      ├──► 层次化合并 ──► 物体分割 (Object)                        │
│      └──► Voronoi骨架 ──► 导航图 (NavigationGraph)               │
│                    │                                             │
│                    ▼                                             │
│           分层场景图 (HMSG) + 导航图                               │
│           (所有坐标在地图参考系中)                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    在线执行 (Online)                             │
│                                                                 │
│  ┌────────────────────── 两层定位 ──────────────────────┐       │
│  │                                                       │       │
│  │  机器人定位 (FastLIVO2):                                │       │
│  │    LiDAR+IMU+camera → 实时位姿 (x,y,z,yaw)             │       │
│  │    连续输出 map→base_link TF                           │       │
│  │                                                       │       │
│  │  语义定位 (FSR-VLN):                                   │       │
│  │    用户指令 "Find me a bottle in the Exhibition Hall"  │       │
│  │      │                                                │       │
│  │      ├──► LLM 解析 (floor/room/object)                 │       │
│  │      ├──► query_floor → CLIP 匹配 → 楼层               │       │
│  │      ├──► query_room  → CLIP 匹配 → 房间               │       │
│  │      ├──► query_object → CLIP 匹配 → Top-K 候选        │       │
│  │      ├──► VLM 重验证 → 确认目标                        │       │
│  │      └──► 目标物体 3D 坐标 (在地图参考系中)             │       │
│  │                                                       │       │
│  └────────────────────── 共享坐标系 ─────────────────────┘       │
│                    │                                             │
│   当前位置 + 目标坐标 → 都在同一个地图参考系                        │
│                    │                                             │
│                    ▼                                             │
│            Nav2 全局规划 → 局部避障 → 机器人到达目标                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. 关键依赖

| 组件 | 用途 |
|---|---|
| Open3D | 点云处理、I/O、可视化 |
| SAM (Segment Anything) | 图像 instance 分割 |
| OpenCLIP | 视觉-语言特征提取 |
| NetworkX | 图数据结构 (场景图、导航图) |
| scipy.spatial.Voronoi | 导航图的 Voronoi 骨架 |
| scipy.spatial.cKDTree | 最近邻搜索 |
| scipy.spatial.Delaunay | 房间分割的 Delaunay 三角化 |
| Hydra / OmegaConf | 配置管理 |
| GPT / Azure OpenAI | LLM 指令解析、VLM 重验证 |
| FastLIVO2 | 机器人实时 LiDAR-惯性-视觉里程计 |
| ScanContext | 激光点云全局重定位 (回环检测) |
| NDT / ICP (PCL) | 点云精配准重定位 |
| ROS 2 Navigation2 | 全局路径规划 + 局部避障 |
| AMCL (Nav2) | 2D 栅格地图粒子滤波定位 |
| FAISS | 高效向量搜索 |
| PyVista | 场景图 3D 可视化 |

---

## 10. G1 机器人部署指南

在 G1 机器人上完整部署 HoloAgent 需要两个阶段：**建图**（先跑一次）和**运行**（每次开机执行）。

### 10.1 整体部署架构

```
┌─────────────────────────────────────────────────────────┐
│                      宿主机 (X86)                        │
│                                                         │
│  run_sensors.sh         run_sem_nav.sh                   │
│  ├── Livox MID360 LiDAR ├── chat_loc_python (语音输入)     │
│  ├── Realsense RGB-D    ├── goal_publisher (语义→坐标)     │
│  └── IMU                ├── g1_getvel_node (管道写入)     │
│                         └── g1_pubvel_node (运控执行)     │
│                                                         │
│  ┌──────────────────────────────────────────────┐      │
│  │              FSR-VLN (Python)                 │      │
│  │  离线建图脚本 + 在线查询引擎                    │      │
│  │  SAM + CLIP + GPT + NetworkX                  │      │
│  └──────────────────────────────────────────────┘      │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                      Docker 容器                         │
│                                                         │
│  run_nav.sh                                             │
│  ├── FastLIVO2 (online_livo + online_reloc)              │
│  ├── Nav2 (global planner + local controller)            │
│  └── pubpose (接收目标位姿 → 触发 Nav2)                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 10.2 阶段一：建图

建图只需执行一次（或场景变化后重新执行），为目标环境建立场景图和导航地图。

#### Step 1: 传感器数据采集

```bash
# 宿主机：启动所有传感器
bash nav_agent/scripts/run_sensors.sh
```

启动后确保以下数据在录制：
- **LiDAR**: Livox MID360 (topic: `/livox/lidar`)
- **RGB-D**: Realsense (topic: `/camera/color/image_raw`, `/camera/depth/image_rect_raw`)
- **IMU**: (topic: `/imu/data`)

在目标场景中操控机器人走一遍（覆盖所有区域），录制 ROS 2 bag：

```bash
ros2 bag record -o scene_name \
  /livox/lidar \
  /camera/color/image_raw \
  /camera/depth/image_rect_raw \
  /camera/color/camera_info \
  /camera/depth/camera_info \
  /imu/data
```

#### Step 2: FastLIVO2 离线建图

```bash
# Docker 内：
ros2 launch fast_livo offline_mapping.launch.py
```

输入: bag 中的 LiDAR + IMU + 相机数据
输出: 稠密 3D 点云 (PCD) + 每帧相机位姿 (pose.txt)

#### Step 3: 数据预处理 (tools/pocket2_toolbox)

```bash
# 1. PCD → PLY 转换 + 降采样去噪
python tools/pocket2_toolbox/pocket2_locnav_pipeline/pcd2ply.py

# 2. 从点云生成 2D 占据栅格地图 (供 Nav2 使用)
python tools/pocket2_toolbox/pocket2_locnav_pipeline/grid_map_gen.py

# 3. 点云投影生成 RGB-D 图像 + 位姿 (如果 Step2 没直接产出)
python tools/pocket2_toolbox/pocket2_rgbd_gen_pipeline/pcd_image_projection_parallel.py
```

#### Step 4: FSR-VLN 语义建图

```bash
cd fsr_vln

# 复制一份配置文件，修改为自己的场景路径
cp config/semantic_scene_reconstruction_ic3f.yaml config/semantic_scene_reconstruction_my_scene.yaml

# 修改 my_scene.yaml:
#   main.scene_id: my_scene
#   main.dataset_path: /path/to/rgbd_datasets/
#   main.save_path: /path/to/output/

# 下载模型权重
#   checkpoints/open_clip_pytorch_model.bin (CLIP ViT-L/14)
#   checkpoints/sam_vit_l_0b3195.pth (SAM ViT-L)

# 运行离线建图
python application/semantic_scene_reconstrucion_offline/semantic_scene_reconstruction.py \
    --config-name=semantic_scene_reconstruction_my_scene
```

完成后检查输出目录：

```
output/my_scene/
├── full_pcd.ply           # 完整场景点云
├── objects/               # 分割出的物体
├── graph_<timestamp>/     # 分层场景图 (HMSG)
├── graph/nav_graph/       # 导航图
│   ├── global_nav_graph_graph.json
│   └── ...
└── tmp/                   # 中间结果
```

#### Step 5: 房间命名

建图完成后，为房间指定名称（用于语义查询）：

```bash
# 编辑配置，设置 spatial_reasoning_method 为 human_assign 或 auto_region
# human_assign: 手动给每个房间命名
# auto_region: 用 CLIP 自动推断房间名

python application/visualize_query_graph/visualize_query_graph_icra_ic3f.py
```

> 参考配置 `visualize_query_graph_icra_ic3f.yaml` 中的 `designated_room_names_ic3f` 列表

---

### 10.3 阶段二：在线运行

每次开机/重定位后执行。

#### Step 1: 启动传感器

```bash
# 宿主机，tmux 会话 "robot_sensors"
bash nav_agent/scripts/run_sensors.sh
```

启动内容：
- Livox MID360 LiDAR 驱动
- Realsense RGB-D 相机 (按需)
- IMU 发布节点

#### Step 2: Docker 内启动定位+导航

```bash
# Docker 内，tmux 会话 "robot_nav_ros2"
bash nav_agent/scripts/run_nav.sh
```

启动内容（4 个 pane 并行）：
- **Pane 0**: FastLIVO2 `online_reloc` — 重定位模式，通过 ScanContext + NDT/ICP 对齐到已有地图
- **Pane 1**: FastLIVO2 `online_livo` — 里程计模式，实时跟踪位姿
- **Pane 2**: `g1_navigation2` — Nav2 导航栈（planner + controller + costmap + AMCL）
- **Pane 3**: `pubpose` — 接收目标位姿，转发给 Nav2

> 确保栅格地图 `grid_map.yaml` 和 `grid_map.pgm` 已放在 `/workspace/map/grid_map/` 下

#### Step 3: 宿主机启动语义导航

```bash
# 宿主机，tmux 会话 "robot_nav"
bash nav_agent/scripts/run_sem_nav.sh
```

启动内容（4 个 pane 并行）：
- **Pane 0**: `chat_loc_python` — 语音交互，监听用户语音指令
- **Pane 1**: `goal_publisher` — 接收语音指令 → 调用 FSR-VLN → 发布目标位姿
- **Pane 2**: `g1_getvel_node` — 从 Nav2 读取速度指令，写入命名管道
- **Pane 3**: `g1_pubvel_node` — 从命名管道读取速度，执行电机控制

#### Step 4: 说出指令

对机器人说出类似指令（或文本输入到 `/chat_loc_pub` topic）：

```
"Find me a stainless steel cup in the Exhibition Hall"
"Take me to the Elevator Lobby"
```

系统自动完成：语音→语义解析→FSR-VLN查询→目标位姿→Nav2路径规划→机器人行走。

---

### 10.4 运行时的消息流

```
语音输入 "Find me a cup in the Exhibition Hall"
    │
    ▼
chat_loc_python ─── String消息('/chat_loc_pub') ───► goal_publisher
                                                          │
                                                    FSR-VLN HMSG 查询
                                                    query_hierarchy_protected_icra()
                                                          │
                                                    目标物体 3D 坐标
                                                          │
                                                    PoseStamped('/object_pose')
                                                          │
                                                          ▼
                                                       pubpose
                                                          │
                                                    转为 Nav2 目标
                                                          │
                                                          ▼
                                         Nav2 (global planner + local controller)
                                                          │
                                                    cmd_vel (速度指令)
                                                          │
                                                          ▼
                                    g1_getvel_node → 命名管道 → g1_pubvel_node
                                                          │
                                                          ▼
                                                     G1 电机执行
```

### 10.5 关键配置检查清单

| 配置项 | 位置 | 说明 |
|---|---|---|
| 栅格地图 | `/workspace/map/grid_map/` | Nav2 需要的 grid_map.yaml + grid_map.pgm |
| FastLIVO2 地图 | `lio_mapping_loc/config/*.yaml` | priorDir 指向预建地图的 PCD 目录 |
| HMSG 场景图路径 | `visualize_query_graph_icra_*.yaml` → `graph_path` | 离线建图产出的 graph_<timestamp> 路径 |
| CLIP 模型 | `checkpoints/open_clip_pytorch_model.bin` | ViT-L/14 权重 |
| SAM 模型 | `checkpoints/sam_vit_l_0b3195.pth` | SAM ViT-L 权重 |
| Azure OpenAI | graph.py 内部或环境变量 | GPT API 密钥 (VLM 慢推理用) |
| 房间名称 | `visualize_query_graph_icra_*.yaml` | human_assign 模式下手动指定 |
| Nav2 参数 | `g1_navigation2/param/g1.yaml` | 最大速度、控制器类型、避障参数 |
| TF 坐标变换 | FastLIVO2 发布 | map→odom→base_link→camera/lidar |

### 10.6 常见问题

**Q: 重定位失败怎么办？**
- 确保机器人开机位置在已建图区域附近（ScanContext 需要看到相似的激光特征）
- 增大 `searchDis` 和 `searchNum` 参数扩大搜索范围
- 手动给一个大致初始位姿

**Q: 语义查询找不到物体？**
- 检查 `graph_path` 配置是否指向正确的 scene graph 目录
- 运行 `generate_room_names()` 确保房间名正确
- 确认目标物体在建图时被 SAM 分割到了（检查 `objects/` 目录）
- 尝试 `use_gpt=True` 启动 VLM 慢推理提升准确率

**Q: Nav2 规划的路径执行不了？**
- 检查 costmap 是否加载正确（`grid_map.yaml` 路径和分辨率）
- 确认 `map→odom→base_link` TF 树完整
- 检查 `g1.yaml` 中 max_vel_x 等参数是否在机器人物理限制内
