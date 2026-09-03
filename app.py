import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

st.set_page_config(page_title="微导管截面多层刚度分析", layout="wide")

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
def compute_layer_contributions(layers):
    """计算每一层对 EA, EI, Kp 的贡献"""
    EA_contrib = []
    EI_contrib = []
    Kp_contrib = []
    for layer in layers:
        r_in = layer['r_in']
        r_out = layer['r_out']
        E_z = layer['E_z']
        # 轴向刚度贡献
        EA_i = np.pi * E_z * (r_out**2 - r_in**2)
        # 弯曲刚度贡献
        EI_i = (np.pi / 4) * E_z * (r_out**4 - r_in**4)
        EA_contrib.append(EA_i)
        EI_contrib.append(EI_i)
        Kp_contrib.append(0.0)  # 先占位，后面计算总 Kp 后按比例分配

    EA_total = sum(EA_contrib)
    EI_total = sum(EI_contrib)
    if layers:
        r0 = layers[0]['r_in']
        rn = layers[-1]['r_out']
        R = (r0 + rn) / 2
        Kp_total = EI_total / (R**3 * (np.pi/2 - 4/np.pi))
        # 按 EI 比例分配 Kp 贡献
        if EI_total > 0:
            Kp_contrib = [ei / EI_total * Kp_total for ei in EI_contrib]
        else:
            Kp_contrib = [0.0] * len(layers)
    else:
        Kp_total = 0.0
    return EA_total, EI_total, Kp_total, EA_contrib, EI_contrib, Kp_contrib

# ==================== 默认层数据 ====================
def create_default_layers():
    return [
        {"layer_type": "普通材料", "r_in": 0.40, "r_out": 0.45,
         "material": "PTFE", "E_z": 500},
        {"layer_type": "编织层", "r_in": 0.45, "r_out": 0.50,
         "d_w": 0.02, "alpha": 45.0, "PPI": 80,
         "E_f": 200000, "E_m": 30,
         "E_z": None},  # 计算时更新
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
if 'layers' not in st.session_state:
    st.session_state.layers = create_default_layers()

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("截面层结构定义")

    n_layers = st.number_input("总层数", min_value=1, max_value=10,
                               value=len(st.session_state.layers), step=1,
                               key="n_layers_input")
    # 调整层数
    if n_layers != len(st.session_state.layers):
        if n_layers > len(st.session_state.layers):
            for _ in range(n_layers - len(st.session_state.layers)):
                last_layer = st.session_state.layers[-1]
                st.session_state.layers.append({
                    "layer_type": "普通材料",
                    "r_in": last_layer['r_out'],
                    "r_out": last_layer['r_out'] + 0.05,
                    "material": "自定义",
                    "E_z": 0.0
                })
        else:
            st.session_state.layers = st.session_state.layers[:n_layers]
        st.rerun()

    # 逐层编辑
    layers_to_save = []
    valid = True
    for i, layer in enumerate(st.session_state.layers):
        with st.expander(f"第 {i+1} 层", expanded=(i == 0)):
            # 层类型
            layer_type = st.radio(
                "层类型",
                ["普通材料", "编织层"],
                horizontal=True,
                key=f"layer_{i}_type",
                index=0 if layer.get('layer_type', '普通材料') == '普通材料' else 1
            )
            layer['layer_type'] = layer_type

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                r_in = st.number_input(
                    "内半径 (mm)",
                    value=float(layer['r_in']),
                    step=0.01, format="%.3f",
                    key=f"layer_{i}_rin"
                )
            with col_r2:
                r_out = st.number_input(
                    "外半径 (mm)",
                    value=float(layer['r_out']),
                    step=0.01, format="%.3f",
                    key=f"layer_{i}_rout"
                )
            if r_out <= r_in:
                st.error(f"第 {i+1} 层外半径必须大于内半径")
                valid = False
            layer['r_in'] = r_in
            layer['r_out'] = r_out

            if layer_type == "普通材料":
                material = st.selectbox(
                    "材料",
                    list(material_library.keys()),
                    key=f"layer_{i}_material",
                    index=list(material_library.keys()).index(layer.get('material', '自定义'))
                )
                layer['material'] = material
                default_E = material_library[material] if material != "自定义" else 0.0
                E_z = st.number_input(
                    "轴向模量 (MPa)",
                    value=float(layer.get('E_z', default_E)),
                    step=100.0, format="%.1f",
                    key=f"layer_{i}_Ez"
                )
                layer['E_z'] = E_z

            else:  # 编织层
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    d_w = st.number_input(
                        "编织丝直径 (mm)",
                        value=float(layer.get('d_w', 0.02)),
                        step=0.005, format="%.3f",
                        key=f"layer_{i}_dw"
                    )
                    alpha = st.number_input(
                        "编织角 (度)",
                        value=float(layer.get('alpha', 45.0)),
                        step=1.0,
                        key=f"layer_{i}_alpha"
                    )
                with col_d2:
                    PPI = st.number_input(
                        "PPI (1/in)",
                        value=int(layer.get('PPI', 80)),
                        step=5,
                        key=f"layer_{i}_PPI"
                    )
                    E_f = st.number_input(
                        "丝材模量 (MPa)",
                        value=float(layer.get('E_f', 200000)),
                        step=1000.0,
                        key=f"layer_{i}_Ef"
                    )
                E_m = st.number_input(
                    "基体模量 (MPa)",
                    value=float(layer.get('E_m', 30)),
                    step=1.0,
                    key=f"layer_{i}_Em"
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

            # 保存该层
            layers_to_save.append(layer)

    # 检查半径连续性
    for i in range(1, len(layers_to_save)):
        if abs(layers_to_save[i]['r_in'] - layers_to_save[i-1]['r_out']) > 1e-6:
            st.warning(f"第 {i+1} 层内半径 ({layers_to_save[i]['r_in']}) 与第 {i} 层外半径 ({layers_to_save[i-1]['r_out']}) 不一致")
            # 仅警告，不强制

    if st.button("保存修改", type="primary"):
        if valid:
            st.session_state.layers = layers_to_save
            st.success("参数已保存")
            st.rerun()
        else:
            st.error("请修正错误后再保存")

    if st.button("恢复示例数据"):
        st.session_state.layers = create_default_layers()
        st.rerun()

# ==================== 主区域 ====================
st.header("截面多层刚度分析")

if not st.session_state.layers:
    st.info("请在左侧定义至少一层")
else:
    layers = st.session_state.layers
    try:
        EA_total, EI_total, Kp_total, EA_contrib, EI_contrib, Kp_contrib = compute_layer_contributions(layers)
    except Exception as e:
        st.error(f"计算错误：{e}")
        st.stop()

    # 显示总刚度
    col1, col2, col3 = st.columns(3)
    col1.metric("总轴向刚度 EA", f"{EA_total:.2f} N")
    col2.metric("总弯曲刚度 EI", f"{EI_total:.2f} N·mm²")
    col3.metric("总抗压扁刚度 Kp", f"{Kp_total:.2f} N/mm")

    # 分两列显示截面示意图和堆叠条形图
    col_left, col_right = st.columns([1, 1.5])

    with col_left:
        st.subheader("导管截面示意图")
        # 绘制同心圆环
        fig_cross, ax_cross = plt.subplots(figsize=(4, 4))
        colors = plt.cm.tab10(np.linspace(0, 1, len(layers)))
        # 先画最内腔（白色）
        ax_cross.add_patch(plt.Circle((0, 0), layers[0]['r_in'], color='white', fill=True, linewidth=0.5))
        # 从内到外绘制每一层
        for i, layer in enumerate(layers):
            r_in = layer['r_in']
            r_out = layer['r_out']
            # 画填充环
            ring = plt.Circle((0, 0), r_out, color=colors[i], alpha=0.6)
            ax_cross.add_patch(ring)
            # 再画内圆覆盖形成环
            inner = plt.Circle((0, 0), r_in, color='white', fill=True)
            ax_cross.add_patch(inner)
            # 在环中间标注层号
            r_mid = (r_in + r_out) / 2
            ax_cross.text(0, r_mid, f"L{i+1}", ha='center', va='center', fontsize=9,
                          color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        ax_cross.set_xlim(-layers[-1]['r_out']*1.2, layers[-1]['r_out']*1.2)
        ax_cross.set_ylim(-layers[-1]['r_out']*1.2, layers[-1]['r_out']*1.2)
        ax_cross.set_aspect('equal')
        ax_cross.axis('off')
        st.pyplot(fig_cross)

    with col_right:
        st.subheader("各层刚度贡献")
        layer_labels = [f"Layer {i+1}" for i in range(len(layers))]
        colors = plt.cm.tab10(np.linspace(0, 1, len(layers)))

        fig_bar, axes_bar = plt.subplots(1, 3, figsize=(12, 4))
        fig_bar.suptitle("各层对刚度的贡献", y=1.02, fontsize=14)

        axes_bar[0].bar(layer_labels, EA_contrib, color=colors)
        axes_bar[0].set_title('Axial Stiffness (EA)', fontsize=12)
        axes_bar[0].set_ylabel('N')
        axes_bar[0].grid(axis='y', linestyle='--', alpha=0.7)

        axes_bar[1].bar(layer_labels, EI_contrib, color=colors)
        axes_bar[1].set_title('Bending Stiffness (EI)', fontsize=12)
        axes_bar[1].set_ylabel('N·mm²')
        axes_bar[1].grid(axis='y', linestyle='--', alpha=0.7)

        axes_bar[2].bar(layer_labels, Kp_contrib, color=colors)
        axes_bar[2].set_title('Crush Stiffness (Kp)', fontsize=12)
        axes_bar[2].set_ylabel('N/mm')
        axes_bar[2].grid(axis='y', linestyle='--', alpha=0.7)

        fig_bar.tight_layout(rect=[0, 0, 1, 0.95])
        st.pyplot(fig_bar)

    # 显示层参数表
    st.subheader("层参数明细")
    df = pd.DataFrame(layers)
    st.dataframe(df, use_container_width=True)
