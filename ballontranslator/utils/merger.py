# -*- coding: utf-8 -*-
"""
区域合并工具 - 从 X-AnyLabeling 移植
用于合并相邻的文本框区域
"""

import os
import json
import copy
import math


def cross_product(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points):
    """
    使用 Monotone Chain 算法计算凸包
    """
    if len(points) <= 2:
        return points

    points.sort()

    lower = []
    for p in points:
        while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def dot_product(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]


def get_bounding_box(shape):
    """
    根据 shape 计算外接矩形 [x_min, y_min, x_max, y_max]
    BallonsTranslator 格式：使用 xyxy 字段
    """
    # BallonsTranslator 直接有 xyxy 字段，这是最准确的
    if 'xyxy' in shape and shape['xyxy']:
        xyxy = shape['xyxy']
        if len(xyxy) == 4:
            return [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]
    
    # 备用：从 lines 计算
    if 'lines' in shape and len(shape['lines']) > 0:
        # lines 是双层嵌套: [[[x1,y1], [x2,y2], ...]]
        points = shape['lines'][0] if isinstance(shape['lines'][0], list) else shape['lines']
        if points:
            x_coords = [float(p[0]) for p in points]
            y_coords = [float(p[1]) for p in points]
            return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
    
    return [0, 0, 0, 0]


def get_mabr_from_points(points):
    """
    计算最小面积外接矩形 (MABR)
    返回: (center_x, center_y, width, height, angle_radians)
    """
    if len(points) <= 1:
        if not points:
            return 0, 0, 0, 0, 0
        p = points[0]
        return p[0], p[1], 0, 0, 0

    hull_points = convex_hull(points)
    if len(hull_points) <= 1:
        p = hull_points[0]
        return p[0], p[1], 0, 0, 0

    min_area = float('inf')
    mabr_params = None

    for i in range(len(hull_points)):
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % len(hull_points)]

        edge_vec = (p2[0] - p1[0], p2[1] - p1[1])
        edge_len = math.sqrt(edge_vec[0]**2 + edge_vec[1]**2)
        if edge_len == 0:
            continue

        axis = (edge_vec[0] / edge_len, edge_vec[1] / edge_len)
        perp_axis = (-axis[1], axis[0])

        min_proj_axis = float('inf')
        max_proj_axis = float('-inf')
        min_proj_perp = float('inf')
        max_proj_perp = float('-inf')

        for p in hull_points:
            proj_axis = dot_product(p, axis)
            proj_perp = dot_product(p, perp_axis)

            min_proj_axis = min(min_proj_axis, proj_axis)
            max_proj_axis = max(max_proj_axis, proj_axis)
            min_proj_perp = min(min_proj_perp, proj_perp)
            max_proj_perp = max(max_proj_perp, proj_perp)

        width = max_proj_axis - min_proj_axis
        height = max_proj_perp - min_proj_perp
        area = width * height

        if area < min_area:
            min_area = area
            
            angle_radians = math.atan2(axis[1], axis[0])
            if angle_radians < 0:
                angle_radians += 2 * math.pi

            x_c_rot = min_proj_axis + width / 2
            y_c_rot = min_proj_perp + height / 2
            
            center_orig_x = x_c_rot * math.cos(angle_radians) - y_c_rot * math.sin(angle_radians)
            center_orig_y = x_c_rot * math.sin(angle_radians) + y_c_rot * math.cos(angle_radians)

            mabr_params = (center_orig_x, center_orig_y, width, height, angle_radians)

    if mabr_params is None:
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        x_min, y_min, x_max, y_max = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min
        angle_radians = 0.0
        return center_x, center_y, width, height, angle_radians

    return mabr_params


def get_merge_group_for_label(label, config):
    if not config.get("USE_SPECIFIC_MERGE_GROUPS", False):
        return -1
    for idx, group in enumerate(config.get("SPECIFIC_MERGE_GROUPS", [])):
        if label in group:
            return idx
    return -1


def can_labels_merge(label1, label2, config):
    if label1 in config.get("LABELS_TO_EXCLUDE_FROM_MERGE", set()) or label2 in config.get("LABELS_TO_EXCLUDE_FROM_MERGE", set()):
        return False

    if config.get("USE_SPECIFIC_MERGE_GROUPS", False):
        g1 = get_merge_group_for_label(label1, config)
        g2 = get_merge_group_for_label(label2, config)
        return g1 != -1 and g1 == g2

    if config.get("REQUIRE_SAME_LABEL", False):
        return label1 == label2
    return True


def merge_labels(label1, label2, strategy):
    if strategy == "FIRST":
        return label1
    elif strategy == "COMBINE":
        return label1 if label1 == label2 else f"{label1}+{label2}"
    elif strategy == "PREFER_NON_DEFAULT":
        default_labels = {"label", ""}
        if label1 in default_labels and label2 not in default_labels:
            return label2
        if label2 in default_labels and label1 not in default_labels:
            return label1
        return label1
    elif strategy == "PREFER_SHORTER":
        if label1 == label2:
            return label1
        return label1 if len(label1) <= len(label2) else label2
    else:
        return label1


def vertical_can_merge(box1, box2, params, advanced_options):
    eps = params.get("overlap_epsilon", 0.0)
    overlap_x = max(0.0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    overlap_x_adj = max(0.0, overlap_x + eps)
    width1 = max(0.0, box1[2] - box1[0])
    width2 = max(0.0, box2[2] - box2[0])
    min_width = max(1e-6, min(width1, width2))
    overlap_ratio_w = (overlap_x_adj / min_width) * 100.0
    vertical_gap = max(box1[1], box2[1]) - min(box1[3], box2[3])

    if overlap_ratio_w < params["min_width_overlap_ratio"]:
        return False

    if advanced_options.get("allow_negative_gap", True):
        return vertical_gap <= params["max_vertical_gap"]
    else:
        return 0 <= vertical_gap <= params["max_vertical_gap"]


def horizontal_can_merge(box1, box2, params, advanced_options):
    eps = params.get("overlap_epsilon", 0.0)
    overlap_y = max(0.0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    overlap_y_adj = max(0.0, overlap_y + eps)
    height1 = max(0.0, box1[3] - box1[1])
    height2 = max(0.0, box2[3] - box2[1])
    min_height = max(1e-6, min(height1, height2))
    overlap_ratio_h = (overlap_y_adj / min_height) * 100.0
    horizontal_gap = max(box1[0], box2[0]) - min(box1[2], box2[2])

    if advanced_options.get("debug_mode", False):
        print(f"      垂直重叠: {overlap_ratio_h:.1f}% (需要 >= {params['min_height_overlap_ratio']}%)")
        print(f"      水平间隙: {horizontal_gap:.1f}px (需要 <= {params['max_horizontal_gap']}px)")

    if overlap_ratio_h < params["min_height_overlap_ratio"]:
        return False

    if advanced_options.get("allow_negative_gap", True):
        return horizontal_gap <= params["max_horizontal_gap"]
    else:
        return 0 <= horizontal_gap <= params["max_horizontal_gap"]


def can_merge_shapes(shape1, shape2, mode, config):
    label1 = shape1.get('label', '')
    label2 = shape2.get('label', '')
    debug = config.get("ADVANCED_MERGE_OPTIONS", {}).get("debug_mode", False)
    
    if not can_labels_merge(label1, label2, config):
        if debug:
            print(f"  ✗ 标签不匹配: '{label1}' vs '{label2}'")
        return False

    box1, box2 = get_bounding_box(shape1), get_bounding_box(shape2)
    
    if debug:
        print(f"\n  检查: '{label1}' [{box1[0]:.0f},{box1[1]:.0f},{box1[2]:.0f},{box1[3]:.0f}] <-> '{label2}' [{box2[0]:.0f},{box2[1]:.0f},{box2[2]:.0f},{box2[3]:.0f}]")

    if mode == "VERTICAL":
        result = vertical_can_merge(box1, box2, config.get("VERTICAL_MERGE_PARAMS", {}), config.get("ADVANCED_MERGE_OPTIONS", {}))
        if debug:
            print(f"    → 垂直合并: {'✓ 可以' if result else '✗ 不可以'}")
        return result
    elif mode == "HORIZONTAL":
        result = horizontal_can_merge(box1, box2, config.get("HORIZONTAL_MERGE_PARAMS", {}), config.get("ADVANCED_MERGE_OPTIONS", {}))
        if debug:
            print(f"    → 水平合并: {'✓ 可以' if result else '✗ 不可以'}")
        return result
    return False


def perform_merge(shapes, mode, config):
    debug = config.get("ADVANCED_MERGE_OPTIONS", {}).get("debug_mode", False)
    
    shapes = [shape for shape in shapes if shape is not None]

    # BallonsTranslator 没有 shape_type 字段，不需要过滤
    if not shapes:
        return [], 0

    if debug:
        print(f"\n开始合并: 模式={mode}, 共 {len(shapes)} 个框")
        print(f"标签统计: {dict((label, sum(1 for s in shapes if s.get('label')==label)) for label in set(s.get('label') for s in shapes))}")

    parent = list(range(len(shapes)))
    
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i
            return True
        return False

    merge_pair_count = 0
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            if find(i) != find(j):
                if can_merge_shapes(shapes[i], shapes[j], mode, config):
                    union(i, j)
                    merge_pair_count += 1
                    if debug:
                        print(f"  ✓ 合并: 框{i} + 框{j}")
    
    if debug:
        print(f"共找到 {merge_pair_count} 对可合并的框")

    groups = {}
    for i in range(len(shapes)):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    final_shapes = []
    total_merge_count = 0
    output_shape_type = config.get("OUTPUT_SHAPE_TYPE", "rectangle")
    
    for root_index in groups:
        indices = groups[root_index]
        if len(indices) == 1:
            final_shapes.append(shapes[indices[0]])
        else:
            group_shapes = [shapes[i] for i in indices]
            
            final_label = ''
            if group_shapes:
                temp_label = group_shapes[0].get('label', '')
                for i in range(1, len(group_shapes)):
                    temp_label = merge_labels(temp_label, group_shapes[i].get('label', ''), config.get("LABEL_MERGE_STRATEGY", "FIRST"))
                final_label = temp_label

            per_label_directions = config.get("PER_LABEL_DIRECTIONS", {})
            default_direction = config.get("READING_DIRECTION", "LTR")
            reading_direction = per_label_directions.get(final_label, default_direction)

            if reading_direction == "RTL":
                group_shapes.sort(key=lambda s: get_bounding_box(s)[0], reverse=True)
            elif reading_direction == "TTB":
                group_shapes.sort(key=lambda s: get_bounding_box(s)[1])
            else:
                group_shapes.sort(key=lambda s: get_bounding_box(s)[0])

            # 合并文本内容 - BallonsTranslator 使用 text 字段（数组）
            final_description = ""
            for s in group_shapes:
                text = s.get('text', [])
                if isinstance(text, list):
                    text = ''.join(text)
                if text:
                    final_description += str(text).strip()
            
            # 获取所有点 - 从 lines 或 xyxy
            all_points = []
            for shape in group_shapes:
                if 'lines' in shape and len(shape['lines']) > 0:
                    points = shape['lines'][0] if isinstance(shape['lines'][0], list) else shape['lines']
                    all_points.extend(points)
                elif 'xyxy' in shape:
                    xyxy = shape['xyxy']
                    all_points.extend([[xyxy[0], xyxy[1]], [xyxy[2], xyxy[1]], 
                                      [xyxy[2], xyxy[3]], [xyxy[0], xyxy[3]]])
            
            if not all_points:
                continue

            merged_shape = copy.deepcopy(group_shapes[0])
            merged_shape['label'] = final_label
            
            # BallonsTranslator 使用 text 字段（数组格式）
            merged_shape['text'] = [final_description] if final_description else []

            # 计算合并后的边界框
            x_coords = [p[0] for p in all_points]
            y_coords = [p[1] for p in all_points]
            merged_box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

            if output_shape_type == "rotation":
                first_direction = group_shapes[0].get('direction')
                all_same_direction = all(s.get('direction') == first_direction for s in group_shapes)

                center_x, center_y, width, height, mab_angle_radians = get_mabr_from_points(all_points)
                
                final_angle = mab_angle_radians
                if all_same_direction:
                    final_angle = first_direction if first_direction is not None else 0

                cos_angle = math.cos(final_angle)
                sin_angle = math.sin(final_angle)
                half_w = width / 2
                half_h = height / 2
                corners = [[-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h]]
                rotated_points = []
                for dx, dy in corners:
                    rotated_x = dx * cos_angle - dy * sin_angle + center_x
                    rotated_y = dx * sin_angle + dy * cos_angle + center_y
                    rotated_points.append([rotated_x, rotated_y])

                # 更新坐标 - BallonsTranslator 格式
                merged_shape['lines'] = [rotated_points]
                merged_shape['xyxy'] = merged_box
                merged_shape['direction'] = final_angle
            else:
                # 矩形格式
                rect_points = [[merged_box[0], merged_box[1]], [merged_box[2], merged_box[1]], 
                              [merged_box[2], merged_box[3]], [merged_box[0], merged_box[3]]]
                
                # 更新坐标 - BallonsTranslator 格式
                merged_shape['xyxy'] = merged_box
                merged_shape['lines'] = [rect_points]
                merged_shape['_bounding_rect'] = [merged_box[0], merged_box[1], 
                                                   merged_box[2] - merged_box[0], 
                                                   merged_box[3] - merged_box[1]]
                
                if 'direction' in merged_shape:
                    merged_shape.pop('direction', None)
            
            final_shapes.append(merged_shape)
            total_merge_count += len(group_shapes) - 1

    return final_shapes, total_merge_count


def process_file(file_path, config):
    """
    处理单个 JSON 文件的区域合并
    仅支持 BallonsTranslator 格式
    """
    debug = config.get("ADVANCED_MERGE_OPTIONS", {}).get("debug_mode", False)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return False, f"读取文件失败: {e}"

    if 'pages' not in data:
        return False, "不是 BallonsTranslator 格式的 JSON 文件"
    
    # 从配置中获取当前图片名（由主窗口传递）
    img_name = config.get("CURRENT_IMAGE_NAME", None)
    
    if debug:
        print(f"\n{'='*60}")
        print(f"处理文件: {file_path}")
        print(f"当前图片: {img_name}")
        print(f"pages 中的图片数量: {len(data['pages'])}")
    
    # 如果没有指定图片名，尝试从文件路径推断
    if not img_name:
        import os.path as osp
        json_basename = osp.basename(file_path)
        img_name = osp.splitext(json_basename)[0] + '.jpg'
        
        if img_name not in data['pages']:
            for ext in ['.png', '.jpeg', '.webp', '.bmp']:
                test_name = osp.splitext(json_basename)[0] + ext
                if test_name in data['pages']:
                    img_name = test_name
                    break
    
    if img_name not in data['pages']:
        available = list(data['pages'].keys())[:5]
        return False, f"在 pages 中找不到图片: {img_name}\n\n可用的图片:\n" + "\n".join(available)
    
    initial_shapes = copy.deepcopy(data['pages'][img_name])
    if not initial_shapes:
        return False, "文件中无标注框"
    
    initial_count = len(initial_shapes)
    
    if debug:
        print(f"找到 {initial_count} 个文本框")
        print(f"前3个框的信息:")
        for i, shape in enumerate(initial_shapes[:3]):
            print(f"  框{i+1}: label={shape.get('label')}, xyxy={shape.get('xyxy')}")
        print(f"{'='*60}\n")
    
    mode = config.get("MERGE_MODE", "NONE")
    if mode == "NONE":
        return False, "合并模式为NONE"

    total_merged = 0
    if mode == "VERTICAL":
        final_shapes, count = perform_merge(initial_shapes, "VERTICAL", config)
        total_merged += count
    elif mode == "HORIZONTAL":
        final_shapes, count = perform_merge(initial_shapes, "HORIZONTAL", config)
        total_merged += count
    elif mode == "VERTICAL_THEN_HORIZONTAL":
        temp, count1 = perform_merge(initial_shapes, "VERTICAL", config)
        final_shapes, count2 = perform_merge(temp, "HORIZONTAL", config)
        total_merged += (count1 + count2)
    elif mode == "HORIZONTAL_THEN_VERTICAL":
        temp, count1 = perform_merge(initial_shapes, "HORIZONTAL", config)
        final_shapes, count2 = perform_merge(temp, "VERTICAL", config)
        total_merged += (count1 + count2)
    else:
        final_shapes = initial_shapes

    if total_merged == 0:
        # 提供更详细的失败原因
        mode_names = {
            "VERTICAL": "垂直合并",
            "HORIZONTAL": "水平合并",
            "VERTICAL_THEN_HORIZONTAL": "先垂直后水平",
            "HORIZONTAL_THEN_VERTICAL": "先水平后垂直"
        }
        mode_cn = mode_names.get(mode, mode)
        
        debug_info = f"未发生任何合并。\n"
        debug_info += f"共有 {len(initial_shapes)} 个文本框。\n"
        if len(initial_shapes) > 0:
            labels = set(s.get('label', '') for s in initial_shapes)
            debug_info += f"标签类型: {', '.join(labels)}\n"
            debug_info += f"合并模式: {mode_cn}"
        return False, debug_info

    # 写回数据
    data['pages'][img_name] = final_shapes
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        final_count = len(final_shapes)
        return True, f"处理完成: 框数 {initial_count} -> {final_count} (减少了 {initial_count - final_count} 个)"
    except Exception as e:
        return False, f"写入文件失败: {e}"
