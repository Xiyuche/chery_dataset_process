import json
import argparse
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"  # 设置默认渲染器为浏览器

# ========== 读取内外参 ==========
def load_intrinsics_txt(path):
    vals = np.loadtxt(path)
    fx, fy, cx, cy = vals[:4]
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ])
    return K

def load_extrinsics_txt(path):
    return np.loadtxt(path)

# ========== 生成 3D BBox ==========
def get_3d_bbox(l, w, h):
    x = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2 ]
    y = [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2 ]
    z = [ 0,    0,    0,    0,   -h,   -h,   -h,   -h ]
    return np.vstack([x, y, z, np.ones(8)])

# ========== 绘制场景 ==========
def _to_matrix_4x4(M):
    """Ensure input is a 4x4 homogeneous transform matrix."""
    A = np.array(M)
    if A.shape == (4, 4):
        return A
    if A.size == 16:
        return A.reshape(4, 4)
    if A.shape == (3, 4):
        bottom = np.array([[0, 0, 0, 1]], dtype=A.dtype)
        return np.vstack([A, bottom])
    raise ValueError(f"Unsupported transform shape: {A.shape}")

def plot_scene_plotly(T_obj_to_world_list, T_cam_to_world, box_size_list, cam_axis_len=1.0):
    """
    绘制场景，支持多个 3D BBox，并显示相机坐标轴。

    T_obj_to_world_list: list[np.ndarray] or (4,4) np.ndarray
    box_size_list: list[[l,w,h]] or single [l,w,h]
    T_cam_to_world: (4,4) np.ndarray
    cam_axis_len: length of camera axis to draw in meters
    """
    # 规范化输入为列表
    if isinstance(T_obj_to_world_list, (np.ndarray, list)) and np.array(T_obj_to_world_list).ndim == 2:
        T_list = [_to_matrix_4x4(T_obj_to_world_list)]
    else:
        T_list = [_to_matrix_4x4(T) for T in T_obj_to_world_list]

    if isinstance(box_size_list[0], (int, float)):
        sizes = [box_size_list]
    else:
        sizes = [np.array(s).tolist() for s in box_size_list]

    # 相机位置与坐标轴
    T_cam_to_world = _to_matrix_4x4(T_cam_to_world)
    cam_center = T_cam_to_world[:3, 3]

    fig = go.Figure()

    # 画每个 BBox
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for T_obj_to_world, box_size in zip(T_list, sizes):
        l, w, h = box_size
        bbox_obj = get_3d_bbox(l, w, h)
        bbox_world = T_obj_to_world @ bbox_obj

        for i, j in edges:
            fig.add_trace(
                go.Scatter3d(
                    x=[bbox_world[0, i], bbox_world[0, j]],
                    y=[bbox_world[1, i], bbox_world[1, j]],
                    z=[bbox_world[2, i], bbox_world[2, j]],
                    mode="lines",
                    line=dict(color="cyan", width=5),
                    showlegend=False,
                )
            )

    # 相机点
    fig.add_trace(
        go.Scatter3d(
            x=[cam_center[0]],
            y=[cam_center[1]],
            z=[cam_center[2]],
            mode="markers",
            marker=dict(size=6, color="red"),
            name="Camera Center",
        )
    )

    # 相机坐标轴（X=红, Y=绿, Z=蓝）
    axes_local = np.array(
        [
            [0, 0, 0, 1],  # origin
            [cam_axis_len, 0, 0, 1],  # X end
            [0, cam_axis_len, 0, 1],  # Y end
            [0, 0, cam_axis_len, 1],  # Z end
        ]
    ).T  # shape (4,4)
    axes_world = T_cam_to_world @ axes_local

    # X axis
    fig.add_trace(
        go.Scatter3d(
            x=[axes_world[0, 0], axes_world[0, 1]],
            y=[axes_world[1, 0], axes_world[1, 1]],
            z=[axes_world[2, 0], axes_world[2, 1]],
            mode="lines",
            line=dict(color="#ff0000", width=6),
            name="Cam X",
        )
    )
    # Y axis
    fig.add_trace(
        go.Scatter3d(
            x=[axes_world[0, 0], axes_world[0, 2]],
            y=[axes_world[1, 0], axes_world[1, 2]],
            z=[axes_world[2, 0], axes_world[2, 2]],
            mode="lines",
            line=dict(color="#00ff00", width=6),
            name="Cam Y",
        )
    )
    # Z axis
    fig.add_trace(
        go.Scatter3d(
            x=[axes_world[0, 0], axes_world[0, 3]],
            y=[axes_world[1, 0], axes_world[1, 3]],
            z=[axes_world[2, 0], axes_world[2, 3]],
            mode="lines",
            line=dict(color="#0000ff", width=6),
            name="Cam Z",
        )
    )

    fig.update_layout(scene=dict(aspectmode="data"))
    fig.write_html("bbox_scene.html", auto_open=True)

# ========== 主入口 ==========
if __name__ == "__main__":
    # 简单 CLI
    parser = argparse.ArgumentParser(description="Visualize 3D bounding boxes and camera axes")
    parser.add_argument("--intrinsics", default="outputs/outputs/test_gs_chery/intrinsics/0.txt", help="Path to intrinsics txt")
    parser.add_argument("--extrinsics", default="outputs/outputs/test_gs_chery/extrinsics/000_0.txt", help="Path to extrinsics txt")
    parser.add_argument("--instances", default="outputs/outputs/test_gs_chery/instances/instances_info.json", help="Path to instances JSON")
    parser.add_argument("--instance-id", default="1", help="Instance ID to visualize, e.g., '1'")
    parser.add_argument("--indices", default="0", help="Comma-separated indices to draw, e.g., '0,5,10'")
    parser.add_argument("--axis-len", type=float, default=1.0, help="Camera axis length")
    args = parser.parse_args()

    # 路径
    intrinsics_file = args.intrinsics
    extrinsics_file = args.extrinsics
    json_file = args.instances

    # 加载参数
    K = load_intrinsics_txt(intrinsics_file)
    T_cam_to_world = load_extrinsics_txt(extrinsics_file)

    # 读取 JSON
    with open(json_file, "r") as f:
        instances = json.load(f)

    # 选择 instance
    if args.instance_id not in instances:
        raise KeyError(f"Instance id {args.instance_id} not found. Available: {list(instances.keys())[:10]} ...")
    obj_info = instances[args.instance_id]["frame_annotations"]

    # 通过指定一组 number（索引）来加载多个 BBox
    # 例如：--indices "0,5,10"
    try:
        idx_list = [int(s) for s in args.indices.split(',') if s.strip() != ""]
    except ValueError:
        raise ValueError("--indices must be a comma-separated list of integers, e.g., '0,5,10'")

    T_list = []
    size_list = []
    for idx in idx_list:
        T_list.append(_to_matrix_4x4(obj_info["obj_to_world"][idx]))
        size_list.append(obj_info["box_size"][idx])

    plot_scene_plotly(T_list, T_cam_to_world, size_list, cam_axis_len=args.axis_len)