import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="导丝刚度分析工具", layout="wide")

COLORS = ['green', 'blue', 'orange', 'purple', 'red', 'cyan', 'magenta', 'yellow', 'black', 'brown']

# ==================== 安全表达式求值 ====================
def safe_eval(expr, x_val):
    import re
    allowed = set("0123456789+-*/(). xXeE")
    if any(ch not in allowed for ch in expr):
        raise ValueError(f"表达式包含非法字符: {expr}")
    expr_sub = expr.replace('x', f'({x_val})')
    return float(eval(expr_sub))

# ==================== 海波管分段函数解析 ====================
def parse_hypo_functions(text):
    segments = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) != 4:
            continue
        try:
            start = float(parts[0])
            end = float(parts[1])
            b_expr = parts[2]
            Z_expr = parts[3]
            segments.append((start, end, b_expr, Z_expr))
        except:
            continue
    return segments

def calc_b_Z(x, segments):
    for start, end, b_expr, Z_expr in segments:
        if start <= x < end:
            b = safe_eval(b_expr, x)
            Z = safe_eval(Z_expr, x)
            return b, Z
    if x < segments[0][0]:
        b = safe_eval(segments[0][2], segments[0][0])
        Z = safe_eval(segments[0][3], segments[0][0])
    else:
        b = safe_eval(segments[-1][2], segments[-1][1])
        Z = safe_eval(segments[-1][3], segments[-1][1])
    return b, Z

# ==================== 计算函数 ====================
def compute_version(x, core_df, hypo_segments, params, eta_global):
    E_core = params['E_core']
    G_core = params['G_core']
    E_hypo = params['E_hypo']
    G_hypo = params['G_hypo']
    D_o = params['D_o']
    D_i = params['D_i']
    w_s = params['w_s']
    L_total = params['L_total']
    F = params['F']
    T0 = params['T0']
    spring_start = params['spring_start']
    spring_end = params['spring_end']
    glue_intervals = params['glue_intervals']

    # 完整海波管基准刚度
    I0 = np.pi / 64 * (D_o**4 - D_i**4)
    J0 = 2 * I0
    A0 = np.pi / 4 * (D_o**2 - D_i**2)
    EI0 = E_hypo * I0
    GJ0 = G_hypo * J0
    EA0 = E_hypo * A0

    n = len(x)
    d_core_arr = np.zeros(n)
    b_arr = np.zeros(n)
    Z_arr = np.zeros(n)
    eta_b_arr = np.zeros(n)
    eta_t_arr = np.zeros(n)
    eta_a_arr = np.zeros(n)

    # 芯丝直径插值
    def interp_core(x_val):
        for _, row in core_df.iterrows():
            if row['start'] <= x_val < row['end']:
                t = (x_val - row['start']) / (row['end'] - row['start'])
                return row['d_start'] + t * (row['d_end'] - row['d_start'])
        if x_val < core_df.iloc[0]['start']:
            return core_df.iloc[0]['d_start']
        else:
            return core_df.iloc[-1]['d_end']

    for i, xi in enumerate(x):
        d_core_arr[i] = interp_core(xi)
        b_val, Z_val = calc_b_Z(xi, hypo_segments)
        b_arr[i] = b_val
        Z_arr[i] = Z_val

        # 传递系数
        eta_b = eta_global['no_spring_b']
        eta_t = eta_global['no_spring_t']
        eta_a = eta_global['no_spring_a']

        if spring_start <= xi < spring_end:
            eta_b = eta_global['spring_b']
            eta_t = eta_global['spring_t']
            eta_a = eta_global['spring_a']

        for g_start, g_end, g_type in glue_intervals:
            if g_start <= xi < g_end:
                if g_type == 'full':
                    eta_b = eta_global['full_b']
                    eta_t = eta_global['full_t']
                    eta_a = eta_global['full_a']
                elif g_type == 'core_spring':
                    eta_b = eta_global['core_spring_b']
                    eta_t = eta_global['core_spring_t']
                    eta_a = eta_global['core_spring_a']
                elif g_type == 'core_hypo':
                    eta_b = eta_global['core_hypo_b']
                    eta_t = eta_global['core_hypo_t']
                    eta_a = eta_global['core_hypo_a']
                break

        eta_b_arr[i] = eta_b
        eta_t_arr[i] = eta_t
        eta_a_arr[i] = eta_a

    # 芯丝刚度
    EI_core = E_core * np.pi * d_core_arr**4 / 64
    GJ_core = G_core * np.pi * d_core_arr**4 / 32
    EA_core = E_core * np.pi * d_core_arr**2 / 4

    # 海波管折减系数
    Y = 0.5184 - b_arr
    denom = Z_arr - w_s
    denom_safe = np.where(denom > 0, denom, 1e-9)
    k = 1.0 / (1.0 + (w_s / denom_safe) * (Y / b_arr))
    k = np.where(denom > 0, k, 1.0)

    EI_hypo = k * EI0
    GJ_hypo = k * GJ0
    EA_hypo = k * EA0

    EI_total = EI_core + eta_b_arr * EI_hypo
    GJ_total = GJ_core + eta_t_arr * GJ_hypo
    EA_total = EA_core + eta_a_arr * EA_hypo

    # 应力计算
    M_total = F * x
    T_total = T0 * x / L_total
    ratio_b = (eta_b_arr * EI_hypo) / (EI_core + eta_b_arr * EI_hypo)
    ratio_t = (eta_t_arr * GJ_hypo) / (GJ_core + eta_t_arr * GJ_hypo)
    M_hypo = ratio_b * M_total
    T_hypo = ratio_t * T_total

    t_wall = (D_o - D_i) / 2
    r_m = (D_o + D_i) / 4
    sigma_bend = M_hypo / (2 * b_arr * t_wall * r_m)
    tau_tors = T_hypo / (2 * b_arr * t_wall * r_m)
    sigma_eq = np.sqrt(sigma_bend**2 + 3 * tau_tors**2)

    return EI_total, GJ_total, EA_total, sigma_bend, tau_tors, sigma_eq

# ==================== 默认数据 ====================
default_core_v1 = pd.DataFrame([
    {"start": 0, "end": 15, "d_start": 0.0508, "d_end": 0.0508},
    {"start": 15, "end": 25, "d_start": 0.0508, "d_end": 0.0762},
    {"start": 25, "end": 100, "d_start": 0.0762, "d_end": 0.0762},
    {"start": 100, "end": 160, "d_start": 0.0762, "d_end": 0.127},
    {"start": 160, "end": 350, "d_start": 0.127, "d_end": 0.127},
])
default_core_v3 = pd.DataFrame([
    {"start": 0, "end": 15, "d_start": 0.0508, "d_end": 0.0508},
    {"start": 15, "end": 25, "d_start": 0.0508, "d_end": 0.0889},
    {"start": 25, "end": 70, "d_start": 0.0889, "d_end": 0.0889},
    {"start": 70, "end": 126, "d_start": 0.0889, "d_end": 0.14224},
    {"start": 126, "end": 175, "d_start": 0.14224, "d_end": 0.14224},
    {"start": 175, "end": 350, "d_start": 0.14224, "d_end": 0.22098},
])

default_hypo_v1 = """0,10,0.036,0.058
10,20,0.049,0.0663
20,30,0.058,0.0747
30,40,0.071,0.083
40,50,0.086,0.0913
50,60,0.096,0.0997
60,70,0.102,0.108
70,80,0.112,0.1163
80,90,0.126,0.1246
90,350,0.152,0.133"""

default_hypo_v2 = """0,10,0.036,0.058
10,90,-0.000000154786*x**3+0.000017054434*x**2+0.0011531092*x+0.0229182506,-0.000008789096*x**2+0.0018164096*x+0.0407148136
90,350,0.152,0.133"""

default_hypo_v3 = """0,10,0.036,0.058
10,70,-0.000000154786*x**3+0.00001307278*x**2+0.0011531092*x+0.023316416,-37/3600000*x**2+0.0014388889*x+0.0446388888
70,145,0.115,0.095
145,180,-0.000030204*(x-145)**2+(0.037-1225*(-0.000030204))/35*(x-145)+0.115,0.095+0.038*(x-145)/35
180,350,0.152,0.133"""

# ==================== 初始化 session_state ====================
if 'saved_versions' not in st.session_state:
    st.session_state.saved_versions = []
if 'current_core' not in st.session_state:
    st.session_state.current_core = default_core_v1.copy()
if 'current_hypo_text' not in st.session_state:
    st.session_state.current_hypo_text = default_hypo_v1
if 'current_name' not in st.session_state:
    st.session_state.current_name = "Version 1 (Step)"
if 'current_spring' not in st.session_state:
    st.session_state.current_spring = (0, 150)
if 'current_glue' not in st.session_state:
    st.session_state.current_glue = "0,1,full\n90,100,core_spring\n345,346,core_hypo"

# 传递系数初始值（用于 number_input 的 value）
eta_defaults = {
    'full_b': 1.0, 'full_t': 1.0, 'full_a': 1.0,
    'core_spring_b': 0.9, 'core_spring_t': 0.9, 'core_spring_a': 0.0,
    'core_hypo_b': 1.0, 'core_hypo_t': 1.0, 'core_hypo_a': 1.0,
    'spring_b': 0.9, 'spring_t': 0.6, 'spring_a': 0.0,
    'no_spring_b': 0.85, 'no_spring_t': 0.35, 'no_spring_a': 0.0,
}
for k, v in eta_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("当前版本编辑")
    st.session_state.current_name = st.text_input("版本名称", value=st.session_state.current_name)

    st.subheader("材料参数")
    E_core = st.number_input("芯丝杨氏模量 (MPa)", value=200000, step=1000)
    G_core = st.number_input("芯丝剪切模量 (MPa)", value=77000, step=1000)
    E_hypo = st.number_input("海波管杨氏模量 (MPa)", value=50000, step=1000)
    G_hypo = st.number_input("海波管剪切模量 (MPa)", value=19231, step=1000)

    st.subheader("几何参数")
    D_o = st.number_input("海波管外径 (mm)", value=0.33, step=0.01)
    D_i = st.number_input("海波管内径 (mm)", value=0.23, step=0.01)
    w_s = st.number_input("槽宽 (mm)", value=0.03, step=0.01)
    L_total = st.number_input("导丝总长 (mm)", value=350, step=10)

    st.subheader("载荷参数")
    F = st.number_input("远端横向力 F (N)", value=0.001, step=0.001, format="%.4f")
    T0 = st.number_input("近端扭矩 T0 (N·mm)", value=1.0, step=0.1)

    st.subheader("弹簧圈范围")
    spring_start = st.number_input("弹簧圈起始位置 (mm)", value=st.session_state.current_spring[0], step=5)
    spring_end = st.number_input("弹簧圈结束位置 (mm)", value=st.session_state.current_spring[1], step=5)
    st.session_state.current_spring = (spring_start, spring_end)

    st.subheader("点胶区间")
    glue_text = st.text_area("格式: start,end,type (每行一个)", value=st.session_state.current_glue)
    st.session_state.current_glue = glue_text

    st.subheader("芯丝直径分段表")
    edited_core = st.data_editor(st.session_state.current_core, num_rows="dynamic")
    st.session_state.current_core = edited_core

    st.subheader("海波管开槽函数 (b(x), Z(x))")
    uploaded_file = st.file_uploader("上传海波管参数 Excel/CSV 文件 (可选)", type=["xlsx", "xls", "csv"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)
            required = ['start', 'end', 'b_expr', 'Z_expr']
            if all(col in df_upload.columns for col in required):
                lines = []
                for _, row in df_upload.iterrows():
                    lines.append(f"{row['start']},{row['end']},{row['b_expr']},{row['Z_expr']}")
                uploaded_text = "\n".join(lines)
                st.session_state.current_hypo_text = uploaded_text
                st.success("文件已加载，已填充下方文本框")
            else:
                st.error(f"文件缺少列，必需列: {', '.join(required)}")
        except Exception as e:
            st.error(f"读取文件出错: {e}")

    st.markdown("每行一个区间：`start,end,b_expr,Z_expr`，支持变量 `x`，支持 `+ - * / **` 和括号")
    hypo_text = st.text_area("海波管函数", value=st.session_state.current_hypo_text, height=200)
    st.session_state.current_hypo_text = hypo_text

    st.subheader("传递系数")
    with st.expander("完全点胶区 (full)"):
        full_b = st.number_input("弯曲", value=st.session_state.full_b, step=0.05)
        st.session_state.full_b = full_b
        full_t = st.number_input("扭转", value=st.session_state.full_t, step=0.05)
        st.session_state.full_t = full_t
        full_a = st.number_input("轴向", value=st.session_state.full_a, step=0.05)
        st.session_state.full_a = full_a
    with st.expander("芯丝+弹簧圈点胶区 (core_spring)"):
        cs_b = st.number_input("弯曲", value=st.session_state.core_spring_b, step=0.05)
        st.session_state.core_spring_b = cs_b
        cs_t = st.number_input("扭转", value=st.session_state.core_spring_t, step=0.05)
        st.session_state.core_spring_t = cs_t
        cs_a = st.number_input("轴向", value=st.session_state.core_spring_a, step=0.05)
        st.session_state.core_spring_a = cs_a
    with st.expander("芯丝+海波管点胶区 (core_hypo)"):
        ch_b = st.number_input("弯曲", value=st.session_state.core_hypo_b, step=0.05)
        st.session_state.core_hypo_b = ch_b
        ch_t = st.number_input("扭转", value=st.session_state.core_hypo_t, step=0.05)
        st.session_state.core_hypo_t = ch_t
        ch_a = st.number_input("轴向", value=st.session_state.core_hypo_a, step=0.05)
        st.session_state.core_hypo_a = ch_a
    with st.expander("有弹簧圈无点胶区"):
        sp_b = st.number_input("弯曲", value=st.session_state.spring_b, step=0.05)
        st.session_state.spring_b = sp_b
        sp_t = st.number_input("扭转", value=st.session_state.spring_t, step=0.05)
        st.session_state.spring_t = sp_t
        sp_a = st.number_input("轴向", value=st.session_state.spring_a, step=0.05)
        st.session_state.spring_a = sp_a
    with st.expander("无弹簧圈无点胶区"):
        ns_b = st.number_input("弯曲", value=st.session_state.no_spring_b, step=0.05)
        st.session_state.no_spring_b = ns_b
        ns_t = st.number_input("扭转", value=st.session_state.no_spring_t, step=0.05)
        st.session_state.no_spring_t = ns_t
        ns_a = st.number_input("轴向", value=st.session_state.no_spring_a, step=0.05)
        st.session_state.no_spring_a = ns_a

    if st.button("保存当前版本", type="primary"):
        glue_intervals = []
        if glue_text.strip():
            for line in glue_text.strip().splitlines():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) == 3:
                    try:
                        glue_intervals.append((float(parts[0]), float(parts[1]), parts[2]))
                    except:
                        pass
        hypo_segments = parse_hypo_functions(hypo_text)
        version = {
            'name': st.session_state.current_name,
            'E_core': E_core,
            'G_core': G_core,
            'E_hypo': E_hypo,
            'G_hypo': G_hypo,
            'D_o': D_o,
            'D_i': D_i,
            'w_s': w_s,
            'L_total': L_total,
            'F': F,
            'T0': T0,
            'spring_start': spring_start,
            'spring_end': spring_end,
            'glue_intervals': glue_intervals,
            'core_df': st.session_state.current_core.copy(),
            'hypo_segments': hypo_segments,
            'eta': {
                'full_b': full_b, 'full_t': full_t, 'full_a': full_a,
                'core_spring_b': cs_b, 'core_spring_t': cs_t, 'core_spring_a': cs_a,
                'core_hypo_b': ch_b, 'core_hypo_t': ch_t, 'core_hypo_a': ch_a,
                'spring_b': sp_b, 'spring_t': sp_t, 'spring_a': sp_a,
                'no_spring_b': ns_b, 'no_spring_t': ns_t, 'no_spring_a': ns_a,
            }
        }
        st.session_state.saved_versions.append(version)
        st.success(f"版本 '{version['name']}' 已保存")

    st.subheader("加载预设版本")
    col1, col2, col3 = st.columns(3)
    if col1.button("版本一"):
        st.session_state.current_name = "Version 1 (Step)"
        st.session_state.current_core = default_core_v1.copy()
        st.session_state.current_hypo_text = default_hypo_v1
        st.session_state.current_spring = (0, 150)
        st.session_state.current_glue = "0,1,full\n90,100,core_spring\n345,346,core_hypo"
        # 同步传递系数为版本一默认值
        for k, v in eta_defaults.items():
            st.session_state[k] = v
        st.rerun()
    if col2.button("版本二"):
        st.session_state.current_name = "Version 2 (Continuous)"
        st.session_state.current_core = default_core_v1.copy()
        st.session_state.current_hypo_text = default_hypo_v2
        st.session_state.current_spring = (0, 150)
        st.session_state.current_glue = "0,1,full\n90,100,core_spring\n345,346,core_hypo"
        for k, v in eta_defaults.items():
            st.session_state[k] = v
        st.rerun()
    if col3.button("版本三"):
        st.session_state.current_name = "Version 3 (New)"
        st.session_state.current_core = default_core_v3.copy()
        st.session_state.current_hypo_text = default_hypo_v3
        st.session_state.current_spring = (0, 120)
        st.session_state.current_glue = "0,1,full\n90,100,core_spring\n345,346,core_hypo"
        for k, v in eta_defaults.items():
            st.session_state[k] = v
        st.rerun()

# ==================== 主区域 ====================
st.header("已保存版本")
if not st.session_state.saved_versions:
    st.info("请在左侧编辑参数并点击“保存当前版本”。")
else:
    for idx, ver in enumerate(st.session_state.saved_versions):
        col1, col2, col3 = st.columns([3,1,1])
        new_name = col1.text_input(
            "版本名称",
            value=ver['name'],
            key=f"rename_{idx}",
            label_visibility="collapsed"
        )
        if new_name != ver['name']:
            ver['name'] = new_name

        if col2.button("加载到左侧", key=f"load_{idx}"):
            st.session_state.current_name = ver['name']
            st.session_state.current_core = ver['core_df'].copy()
            st.session_state.current_spring = (ver['spring_start'], ver['spring_end'])
            st.session_state.current_glue = "\n".join([f"{s},{e},{t}" for s,e,t in ver['glue_intervals']])
            hypo_lines = [f"{seg[0]},{seg[1]},{seg[2]},{seg[3]}" for seg in ver['hypo_segments']]
            st.session_state.current_hypo_text = "\n".join(hypo_lines)
            # 传递系数
            st.session_state.full_b = ver['eta']['full_b']
            st.session_state.full_t = ver['eta']['full_t']
            st.session_state.full_a = ver['eta']['full_a']
            st.session_state.core_spring_b = ver['eta']['core_spring_b']
            st.session_state.core_spring_t = ver['eta']['core_spring_t']
            st.session_state.core_spring_a = ver['eta']['core_spring_a']
            st.session_state.core_hypo_b = ver['eta']['core_hypo_b']
            st.session_state.core_hypo_t = ver['eta']['core_hypo_t']
            st.session_state.core_hypo_a = ver['eta']['core_hypo_a']
            st.session_state.spring_b = ver['eta']['spring_b']
            st.session_state.spring_t = ver['eta']['spring_t']
            st.session_state.spring_a = ver['eta']['spring_a']
            st.session_state.no_spring_b = ver['eta']['no_spring_b']
            st.session_state.no_spring_t = ver['eta']['no_spring_t']
            st.session_state.no_spring_a = ver['eta']['no_spring_a']
            st.rerun()
        if col3.button("删除", key=f"del_{idx}"):
            st.session_state.saved_versions.pop(idx)
            st.rerun()

    st.subheader("选择要对比的版本")
    selected_indices = []
    for idx, ver in enumerate(st.session_state.saved_versions):
        if st.checkbox(ver['name'], key=f"check_{idx}"):
            selected_indices.append(idx)

    if st.button("生成对比曲线", type="primary"):
        if not selected_indices:
            st.warning("请至少选择一个版本")
        else:
            x = np.linspace(0, L_total, 500)

            fig1, axes1 = plt.subplots(3, 1, figsize=(10, 12))
            fig1.suptitle("Stiffness Comparison")
            axes1[0].set_ylabel('Bending stiffness EI (N·mm²)')
            axes1[1].set_ylabel('Torsional stiffness GJ (N·mm²)')
            axes1[2].set_xlabel('Distance from distal end (mm)')
            axes1[2].set_ylabel('Axial stiffness EA (N)')
            for ax in axes1:
                ax.grid(True)

            fig2, axes2 = plt.subplots(3, 1, figsize=(10, 12))
            fig2.suptitle("Hypo-tube Connector Stress Comparison")
            axes2[0].set_ylabel('Bending normal stress (MPa)')
            axes2[1].set_ylabel('Torsional shear stress (MPa)')
            axes2[2].set_xlabel('Distance from distal end (mm)')
            axes2[2].set_ylabel('Von Mises stress (MPa)')
            for ax in axes2:
                ax.grid(True)

            for idx in selected_indices:
                ver = st.session_state.saved_versions[idx]
                color = COLORS[idx % len(COLORS)]
                label = ver['name']

                params = {
                    'E_core': ver['E_core'],
                    'G_core': ver['G_core'],
                    'E_hypo': ver['E_hypo'],
                    'G_hypo': ver['G_hypo'],
                    'D_o': ver['D_o'],
                    'D_i': ver['D_i'],
                    'w_s': ver['w_s'],
                    'L_total': ver['L_total'],
                    'F': ver['F'],
                    'T0': ver['T0'],
                    'spring_start': ver['spring_start'],
                    'spring_end': ver['spring_end'],
                    'glue_intervals': ver['glue_intervals'],
                }

                EI_total, GJ_total, EA_total, sigma_bend, tau_tors, sigma_eq = compute_version(
                    x, ver['core_df'], ver['hypo_segments'], params, ver['eta']
                )

                axes1[0].plot(x, EI_total, color=color, linewidth=2, label=label)
                axes1[1].plot(x, GJ_total, color=color, linewidth=2, label=label)
                axes1[2].plot(x, EA_total, color=color, linewidth=2, label=label)

                axes2[0].plot(x, sigma_bend, color=color, linewidth=2, label=label)
                axes2[1].plot(x, tau_tors, color=color, linewidth=2, label=label)
                axes2[2].plot(x, sigma_eq, color=color, linewidth=2, label=label)

            axes1[0].legend()
            axes1[1].legend()
            axes1[2].legend()
            axes2[0].legend()
            axes2[1].legend()
            axes2[2].legend()

            st.pyplot(fig1)
            st.pyplot(fig2)
