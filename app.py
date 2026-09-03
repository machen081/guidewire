import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

st.set_page_config(page_title="微导管多层刚度沿长度分布计算器", layout="wide")

# ==================== 材料库 ====================
material_library = {
    "自定义": None,
    "PTFE": 500,
    "FEP": 400,
    "Pebax 3533": 10,
    "Pebax 5533": 30,
    "Pebax 7233": 50,
    "尼龙 12": 1500,
    "尼龙 6": 2500,
    "聚酰亚胺": 2500,
    "不锈钢 304": 200000,
    "镍钛合金": 60000,
    "钴铬合金": 220000,
}

# ==================== 计算函数 ====================
def compute_stiffness_at_x(layers):
    """根据给定位置的层参数计算 EA, EI, Kp"""
    EA = 0.0
    EI = 0.0
    for layer in layers:
        r_in = layer['r_in']
        r_out = layer['r_out']
        E_z = layer['E_z']
        if r_out <= r_in:
            raise ValueError(f"层内外半径错误：内半径 {r_in} 不小于外半径 {r_out}")
        EA += np.pi * E_z * (r_out**2 - r_in**2)
        EI += (np.pi / 4) * E_z * (r_out**4 - r_in**4)
    if not layers:
        raise ValueError("至少需要一层")
    r0 = layers[0]['r_in']
    rn = layers[-1]['r_out']
    R = (r0 + rn) / 2
    Kp = EI / (R**3 * (np.pi/2 - 4/np.pi))
    return EA, EI, Kp

# ==================== 默认分段数据 ====================
def create_default_segment_layers():
    """默认三层：PTFE、编织层、Pebax"""
    return [
        {"layer_type": "普通材料", "r_in": 0.40, "r_out": 0.45,
         "material": "PTFE", "E_z": 500},
        {"layer_type": "编织层", "r_in": 0.45, "r_out": 0.50,
         "d_w": 0.02, "alpha": 45.0, "PPI": 80,
         "E_f": 200000, "E_m": 30,
         "E_z": None},  # 占位，计算时更新
        {"layer_type": "普通材料", "r_in": 0.50, "r_out": 0.60,
         "material": "Pebax 7233", "E_z": 50},
    ]

def update_braid_Ez(layer):
    """根据编织参数计算等效轴向模量"""
    d_w = layer['d_w']
    alpha = layer['alpha']
    PPI = layer['PPI']
    E_f = layer['E_f']
    E_m = layer['E_m']
    r_in = layer['r_in']
    r_out = layer['r_out']
    alpha_rad = np.radians(alpha)
    denom = 25.4 * 2 * (r_out**2 - r_in**2) * np.cos(alpha_rad)
    if denom > 0 and r_out > r_in:
        V_f = min(1.0, (np.pi * d_w**2 * PPI) / denom)
    else:
        V_f = 0.0
    Ez = E_f * V_f * (np.cos(alpha_rad)**4) + E_m * (1 - V_f)
    return Ez

# ==================== session_state 初始化 ====================
if 'segments' not in st.session_state:
    st.session_state.segments = deepcopy(default_segments)
if 'L_total' not in st.session_state:
    st.session_state.L_total = 350.0

# 默认分段
default_segments = [
    {"start": 0, "end": 100, "layers": create_default_segment_layers()},
    {"start": 100, "end": 350, "layers": create_default_segment_layers()},
]

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("导管总长度")
    L_total = st.number_input("总长度 (mm)", min_value=1.0, value=st.session_state.L_total,
                              step=10.0, key="L_total_input")
    st.session_state.L_total = L_total

    st.header("分段管理")
    n_segments = st.number_input("分段数", min_value=1, max_value=20,
                                 value=len(st.session_state.segments), step=1,
                                 key="n_segments_input")
    if n_segments != len(st.session_state.segments):
        if n_segments > len(st.session_state.segments):
            for _ in range(n_segments - len(st.session_state.segments)):
                last_seg = st.session_state.segments[-1]
                new_start = last_seg['end']
                new_end = min(new_start + 10.0, L_total)
                st.session_state.segments.append({
                    "start": new_start,
                    "end": new_end,
                    "layers": deepcopy(last_seg['layers'])
                })
        else:
            st.session_state.segments = st.session_state.segments[:n_segments]
        st.rerun()

    # 编辑每个分段
    segments_to_save = []
    valid = True
    for i, seg in enumerate(st.session_state.segments):
        with st.expander(f"分段 {i+1}", expanded=(i == 0)):
            col1, col2 = st.columns(2)
            with col1:
                start = st.number_input(f"起点 (mm)", value=float(seg['start']),
                                        step=1.0, key=f"seg_{i}_start")
            with col2:
                end = st.number_input(f"终点 (mm)", value=float(seg['end']),
                                      step=1.0, key=f"seg_{i}_end")
            if end <= start:
                st.error("终点必须大于起点")
                valid = False

            st.markdown("**该段的层结构**")

            # 动态编辑每一层
            layers_list = []
            n_layers = st.number_input(f"该段层数", min_value=1, max_value=10,
                                       value=len(seg['layers']), step=1,
                                       key=f"seg_{i}_n_layers")
            # 调整层数
            current_layers = seg['layers']
            if n_layers != len(current_layers):
                if n_layers > len(current_layers):
                    for _ in range(n_layers - len(current_layers)):
                        current_layers.append({
                            "layer_type": "普通材料",
                            "r_in": current_layers[-1]['r_out'] if current_layers else 0.0,
                            "r_out": current_layers[-1]['r_out'] + 0.05 if current_layers else 0.1,
                            "material": "自定义",
                            "E_z": 0.0
                        })
                else:
                    current_layers = current_layers[:n_layers]
                seg['layers'] = current_layers

            # 逐层输入
            for j, layer in enumerate(seg['layers']):
                st.markdown(f"**第 {j+1} 层**")
                # 层类型选择
                layer_type = st.radio(
                    f"层类型",
                    ["普通材料", "编织层"],
                    horizontal=True,
                    key=f"seg_{i}_layer_{j}_type",
                    index=0 if layer.get('layer_type', '普通材料') == '普通材料' else 1
                )
                layer['layer_type'] = layer_type

                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    r_in = st.number_input(
                        f"内半径 (mm)",
                        value=float(layer['r_in']),
                        step=0.01, format="%.3f",
                        key=f"seg_{i}_layer_{j}_r_in"
                    )
                with col_r2:
                    r_out = st.number_input(
                        f"外半径 (mm)",
                        value=float(layer['r_out']),
                        step=0.01, format="%.3f",
                        key=f"seg_{i}_layer_{j}_r_out"
                    )
                if r_out <= r_in:
                    st.error(f"第 {j+1} 层外半径必须大于内半径")
                    valid = False
                layer['r_in'] = r_in
                layer['r_out'] = r_out

                if layer_type == "普通材料":
                    # 材料选择
                    material = st.selectbox(
                        "材料",
                        list(material_library.keys()),
                        key=f"seg_{i}_layer_{j}_material",
                        index=list(material_library.keys()).index(layer.get('material', '自定义'))
                    )
                    layer['material'] = material
                    default_E = material_library[material] if material != "自定义" else 0.0
                    E_z = st.number_input(
                        "轴向模量 (MPa)",
                        value=float(layer.get('E_z', default_E)),
                        step=100.0, format="%.1f",
                        key=f"seg_{i}_layer_{j}_Ez"
                    )
                    layer['E_z'] = E_z

                else:  # 编织层
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        d_w = st.number_input(
                            "编织丝直径 (mm)",
                            value=float(layer.get('d_w', 0.02)),
                            step=0.005, format="%.3f",
                            key=f"seg_{i}_layer_{j}_dw"
                        )
                        alpha = st.number_input(
                            "编织角 (度)",
                            value=float(layer.get('alpha', 45.0)),
                            step=1.0,
                            key=f"seg_{i}_layer_{j}_alpha"
                        )
                    with col_d2:
                        PPI = st.number_input(
                            "PPI (1/in)",
                            value=int(layer.get('PPI', 80)),
                            step=5,
                            key=f"seg_{i}_layer_{j}_PPI"
                        )
                        E_f = st.number_input(
                            "丝材模量 (MPa)",
                            value=float(layer.get('E_f', 200000)),
                            step=1000.0,
                            key=f"seg_{i}_layer_{j}_Ef"
                        )
                    E_m = st.number_input(
                        "基体模量 (MPa)",
                        value=float(layer.get('E_m', 30)),
                        step=1.0,
                        key=f"seg_{i}_layer_{j}_Em"
                    )
                    layer['d_w'] = d_w
                    layer['alpha'] = alpha
                    layer['PPI'] = PPI
                    layer['E_f'] = E_f
                    layer['E_m'] = E_m
                    # 计算等效轴向模量
                    Ez_calc = update_braid_Ez(layer)
                    layer['E_z'] = Ez_calc
                    st.success(f"编织层等效轴向模量 E_z = {Ez_calc:.1f} MPa")

                layers_list.append(layer)

            # 更新分段层数据
            seg['layers'] = layers_list

            # 检查相邻层半径连续性
            for j in range(1, len(layers_list)):
                if abs(layers_list[j]['r_in'] - layers_list[j-1]['r_out']) > 1e-6:
                    st.warning(f"第 {j+1} 层内半径 ({layers_list[j]['r_in']}) 与上一层外半径 ({layers_list[j-1]['r_out']}) 不一致，可能导致物理不连续")
                    # 不标记为无效，仅警告

            segments_to_save.append({
                "start": start,
                "end": end,
                "layers": layers_list
            })

    # 检查分段覆盖是否连续
    if valid and len(segments_to_save) > 0:
        sorted_segments = sorted(segments_to_save, key=lambda x: x['start'])
        if sorted_segments[0]['start'] > 0:
            st.warning(f"第一个分段起点应不小于0，当前为{sorted_segments[0]['start']}")
            valid = False
        for i in range(len(sorted_segments)-1):
            if abs(sorted_segments[i]['end'] - sorted_segments[i+1]['start']) > 1e-6:
                st.warning(f"分段 {i+1} 终点 ({sorted_segments[i]['end']}) 与分段 {i+2} 起点 ({sorted_segments[i+1]['start']}) 不连续")
                valid = False
        if sorted_segments[-1]['end'] < L_total:
            st.warning(f"最后一个分段终点应不小于总长度 {L_total}，当前为{sorted_segments[-1]['end']}")
            valid = False

    if st.button("保存修改", type="primary"):
        if valid:
            st.session_state.segments = segments_to_save
            st.success("参数已保存")
            st.rerun()
        else:
            st.error("请修正错误后再保存")

    if st.button("恢复示例数据"):
        st.session_state.segments = deepcopy(default_segments)
        st.session_state.L_total = 350.0
        st.rerun()

# ==================== 主区域 ====================
st.header("刚度沿长度分布")

if not st.session_state.segments:
    st.info("请在左侧添加分段数据")
else:
    plot_segments = st.session_state.segments

    x = np.linspace(0, st.session_state.L_total, 500)

    EA_arr = np.zeros_like(x)
    EI_arr = np.zeros_like(x)
    Kp_arr = np.zeros_like(x)

    for i, xi in enumerate(x):
        seg = None
        for s in plot_segments:
            if s['start'] <= xi < s['end']:
                seg = s
                break
        if seg is None:
            if xi < plot_segments[0]['start']:
                seg = plot_segments[0]
            else:
                seg = plot_segments[-1]
        try:
            EA, EI, Kp = compute_stiffness_at_x(seg['layers'])
            EA_arr[i] = EA
            EI_arr[i] = EI
            Kp_arr[i] = Kp
        except Exception as e:
            st.error(f"在 x={xi:.2f} 处计算失败：{e}")
            st.stop()

    # 绘图（调整布局避免文字重叠）
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    fig.suptitle("微导管刚度沿长度分布", y=0.98, fontsize=14)

    axes[0].plot(x, EA_arr, 'b-', linewidth=2)
    axes[0].set_ylabel('轴向刚度 EA (N)', fontsize=10)
    axes[0].grid(True)
    axes[0].set_title('Axial Stiffness', fontsize=12, pad=10)

    axes[1].plot(x, EI_arr, 'g-', linewidth=2)
    axes[1].set_ylabel('弯曲刚度 EI (N·mm²)', fontsize=10)
    axes[1].grid(True)
    axes[1].set_title('Bending Stiffness', fontsize=12, pad=10)

    axes[2].plot(x, Kp_arr, 'r-', linewidth=2)
    axes[2].set_xlabel('距远端位置 (mm)', fontsize=10)
    axes[2].set_ylabel('抗压扁刚度 Kp (N/mm)', fontsize=10)
    axes[2].grid(True)
    axes[2].set_title('Crush Stiffness (diametral compression)', fontsize=12, pad=10)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    st.pyplot(fig)

    # 显示分段数据表
    st.subheader("当前分段数据")
    for i, seg in enumerate(plot_segments):
        st.markdown(f"**分段 {i+1}：{seg['start']:.1f} – {seg['end']:.1f} mm**")
        # 将层列表转换为DataFrame显示
        df = pd.DataFrame(seg['layers'])
        st.dataframe(df, use_container_width=True)
