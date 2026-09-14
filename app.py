import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

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
            start = float(parts[0]); end = float(parts[1])
            b_expr = parts[2]; Z_expr = parts[3]
            segments.append((start, end, b_expr, Z_expr))
        except:
            continue
    return segments

def calc_b_Z_original(x, segments):
    for start, end, b_expr, Z_expr in segments:
        if start <= x < end:
            return safe_eval(b_expr, x), safe_eval(Z_expr, x)
    if x < segments[0][0]:
        return safe_eval(segments[0][2], segments[0][0]), safe_eval(segments[0][3], segments[0][0])
    else:
        return safe_eval(segments[-1][2], segments[-1][1]), safe_eval(segments[-1][3], segments[-1][1])

def calc_b_Z_smooth(x, segments, delta=2.0):
    for i, (start, end, b_expr, Z_expr) in enumerate(segments):
        if start <= x < end:
            for boundary in [start, end]:
                if boundary <= 0: continue
                if abs(x - boundary) < delta:
                    prev_seg = next_seg = None
                    for seg in segments:
                        if seg[1] == boundary: prev_seg = seg
                        if seg[0] == boundary: next_seg = seg
                    if prev_seg and next_seg:
                        eps = 1e-5
                        b_l = safe_eval(prev_seg[2], boundary - eps)
                        b_r = safe_eval(next_seg[2], boundary + eps)
                        Z_l = safe_eval(prev_seg[3], boundary - eps)
                        Z_r = safe_eval(next_seg[3], boundary + eps)
                        db_l = (safe_eval(prev_seg[2], boundary-eps) - safe_eval(prev_seg[2], boundary-2*eps))/eps
                        db_r = (safe_eval(next_seg[2], boundary+2*eps) - safe_eval(next_seg[2], boundary+eps))/eps
                        dZ_l = (safe_eval(prev_seg[3], boundary-eps) - safe_eval(prev_seg[3], boundary-2*eps))/eps
                        dZ_r = (safe_eval(next_seg[3], boundary+2*eps) - safe_eval(next_seg[3], boundary+eps))/eps
                        t = (x - (boundary-delta))/(2*delta)
                        h00 = 2*t**3-3*t**2+1; h10 = t**3-2*t**2+t
                        h01 = -2*t**3+3*t**2; h11 = t**3-t**2
                        b = h00*b_l + h10*(2*delta)*db_l + h01*b_r + h11*(2*delta)*db_r
                        Z = h00*Z_l + h10*(2*delta)*dZ_l + h01*Z_r + h11*(2*delta)*dZ_r
                        return b, Z
            return safe_eval(b_expr, x), safe_eval(Z_expr, x)
    if x < segments[0][0]:
        return safe_eval(segments[0][2], segments[0][0]), safe_eval(segments[0][3], segments[0][0])
    else:
        return safe_eval(segments[-1][2], segments[-1][1]), safe_eval(segments[-1][3], segments[-1][1])

# ==================== 芯丝插值 ====================
def interp_core(x_val, core_df, smooth=False):
    for _, row in core_df.iterrows():
        if row['start'] <= x_val < row['end']:
            d1, d2 = row['d_start'], row['d_end']
            if d1 == d2: return d1
            if smooth:
                L_t = row['end'] - row['start']
                t = max(0.0, min(1.0, (x_val - row['start'])/L_t))
                return d1 + (d2-d1)*(3*t**2 - 2*t**3)
            else:
                t = (x_val - row['start'])/(row['end'] - row['start'])
                return d1 + t*(d2-d1)
    if x_val < core_df.iloc[0]['start']: return core_df.iloc[0]['d_start']
    return core_df.iloc[-1]['d_end']

# ==================== 弹簧圈螺距模型 ====================
def get_pitch(x, x_stretch_start, x_coil_end, p_start, p_end, pitch_mode='constant', n_segments=3):
    if x < x_stretch_start or x > x_coil_end:
        return p_start
    if pitch_mode == 'constant':
        return p_start
    elif pitch_mode == 'segmented':
        seg_len = (x_coil_end - x_stretch_start) / n_segments
        idx = int((x - x_stretch_start) / seg_len)
        idx = min(idx, n_segments - 1)
        return p_start + (p_end - p_start) * (idx + 1) / n_segments
    else:
        t = (x - x_stretch_start) / (x_coil_end - x_stretch_start)
        return p_start + (p_end - p_start) * t

def eta_t_spring(x, glue_intervals, x_stretch_start, x_coil_end, p_start, p_end,
                 eta_base, eta_glue_peak, pitch_mode='constant', n_segments=3):
    if x <= 1: return 1.0
    if x <= x_stretch_start or x > x_coil_end:
        base = eta_base
    else:
        p = get_pitch(x, x_stretch_start, x_coil_end, p_start, p_end, pitch_mode, n_segments)
        base = eta_base * p_start / p
    for g_start, g_end, g_type in glue_intervals:
        if g_start <= x < g_end:
            return eta_glue_peak
    return base

# ==================== 计算函数 ====================
def compute_version(x, core_df, hypo_segments, params, eta_global,
                    smooth=False, spring_enabled=False, complex_params=None):
    E_core = params['E_core']; G_core = params['G_core']
    E_hypo = params['E_hypo']; G_hypo = params['G_hypo']
    D_o = params['D_o']; D_i = params['D_i']; w_s = params['w_s']
    L_total = params['L_total']; F = params['F']; T0 = params['T0']
    spring_start = params['spring_start']; spring_end = params['spring_end']
    glue_intervals = params['glue_intervals']

    I0 = np.pi/64*(D_o**4 - D_i**4); J0 = 2*I0
    A0 = np.pi/4*(D_o**2 - D_i**2)
    EI0 = E_hypo*I0; GJ0 = G_hypo*J0; EA0 = E_hypo*A0

    n = len(x)
    d_core_arr = np.zeros(n); b_arr = np.zeros(n); Z_arr = np.zeros(n)
    eta_b_arr = np.zeros(n); eta_t_arr = np.zeros(n); eta_a_arr = np.zeros(n)

    for i, xi in enumerate(x):
        d_core_arr[i] = interp_core(xi, core_df, smooth=smooth)
        if smooth:
            b_val, Z_val = calc_b_Z_smooth(xi, hypo_segments, delta=2.0)
        else:
            b_val, Z_val = calc_b_Z_original(xi, hypo_segments)
        b_arr[i] = b_val; Z_arr[i] = Z_val

        eta_b = eta_global['no_spring_b']; eta_a = eta_global['no_spring_a']
        if spring_start <= xi < spring_end:
            eta_b = eta_global['spring_b']; eta_a = eta_global['spring_a']
        for g_start, g_end, g_type in glue_intervals:
            if g_start <= xi < g_end:
                if g_type == 'full': eta_b = eta_global['full_b']; eta_a = eta_global['full_a']
                elif g_type == 'core_spring': eta_b = eta_global['core_spring_b']; eta_a = eta_global['core_spring_a']
                elif g_type == 'core_hypo': eta_b = eta_global['core_hypo_b']; eta_a = eta_global['core_hypo_a']
                break

        if spring_enabled and complex_params:
            eta_t = eta_t_spring(xi, glue_intervals,
                complex_params['x_stretch_start'], complex_params['x_coil_end'],
                complex_params['p_start'], complex_params['p_end'],
                eta_global['spring_t'], complex_params['eta_glue_peak'],
                complex_params.get('pitch_mode', 'constant'),
                complex_params.get('n_segments', 3))
        else:
            eta_t = eta_global['no_spring_t']
            if spring_start <= xi < spring_end: eta_t = eta_global['spring_t']
            for g_start, g_end, g_type in glue_intervals:
                if g_start <= xi < g_end:
                    if g_type == 'full': eta_t = eta_global['full_t']
                    elif g_type == 'core_spring': eta_t = eta_global['core_spring_t']
                    elif g_type == 'core_hypo': eta_t = eta_global['core_hypo_t']
                    break

        eta_b_arr[i] = eta_b; eta_t_arr[i] = eta_t; eta_a_arr[i] = eta_a

    EI_core = E_core*np.pi*d_core_arr**4/64
    GJ_core = G_core*np.pi*d_core_arr**4/32
    EA_core = E_core*np.pi*d_core_arr**2/4

    Y = 0.5184 - b_arr; denom = Z_arr - w_s
    denom_safe = np.where(denom > 0, denom, 1e-9)
    k = 1.0/(1.0 + (w_s/denom_safe)*(Y/b_arr))
    k = np.where(denom > 0, k, 1.0)

    EI_hypo = k*EI0; GJ_hypo = k*GJ0; EA_hypo = k*EA0
    EI_total = EI_core + eta_b_arr*EI_hypo
    GJ_total = GJ_core + eta_t_arr*GJ_hypo
    EA_total = EA_core + eta_a_arr*EA_hypo

    M_total = F*x; T_total = T0*x/L_total
    ratio_b = (eta_b_arr*EI_hypo)/(EI_core + eta_b_arr*EI_hypo)
    ratio_t = (eta_t_arr*GJ_hypo)/(GJ_core + eta_t_arr*GJ_hypo)
    M_hypo = ratio_b*M_total; T_hypo = ratio_t*T_total

    t_wall = (D_o - D_i)/2; r_m = (D_o + D_i)/4
    sigma_bend = M_hypo/(2*b_arr*t_wall*r_m)
    tau_tors = T_hypo/(2*b_arr*t_wall*r_m)
    sigma_eq = np.sqrt(sigma_bend**2 + 3*tau_tors**2)

    theta = cumulative_trapezoid(M_total/EI_total, x, initial=0)
    theta = theta - theta[-1]
    y_def = cumulative_trapezoid(theta, x, initial=0)
    y_def = y_def - y_def[-1]
    phi = cumulative_trapezoid(T_total/GJ_total, x, initial=0)

    return (EI_total, GJ_total, EA_total, sigma_bend, tau_tors, sigma_eq, y_def, phi)

# ==================== 参数改进建议（仅海波管+芯丝） ====================
def generate_suggestions(ver):
    suggestions = []
    hypo_segments = ver['hypo_segments']
    for i, seg in enumerate(hypo_segments):
        if i == 0: continue
        prev = hypo_segments[i-1]
        boundary = seg[0]
        if abs(prev[1] - boundary) > 1e-6:
            suggestions.append({
                'type': '海波管分段不连续', 'position': boundary,
                'description': f"前段结束于 {prev[1]} mm，后段开始于 {boundary} mm，存在间隙或重叠。",
                'formula': "调整区间使前段 end 等于后段 start。",
            })
        else:
            delta = 2.0; xb = boundary
            suggestions.append({
                'type': '海波管参数突变', 'position': boundary,
                'description': f"在 x={boundary} mm 处参数变化，建议使用三次样条过渡。",
                'formula': (f"在 [{xb-delta}, {xb+delta}] mm 区间内使用三次多项式：\n"
                            f"f(x) = a0 + a1*(x-{xb}) + a2*(x-{xb})^2 + a3*(x-{xb})^3\n"
                            f"系数由端点值和斜率匹配条件确定。"),
            })
    core_df = ver['core_df']
    for _, row in core_df.iterrows():
        if row['d_start'] != row['d_end']:
            L_t = row['end'] - row['start']; x1 = row['start']
            d1 = row['d_start']; d2 = row['d_end']
            suggestions.append({
                'type': '芯丝直径线性过渡', 'position': f"{x1}-{row['end']} mm",
                'description': "当前为线性过渡，建议改为 S 形曲线。",
                'formula': f"d(x) = {d1} + ({d2}-{d1}) * [3*((x-{x1})/{L_t})^2 - 2*((x-{x1})/{L_t})^3]",
            })
            break
    return suggestions

# ==================== 自动推荐点胶位置 ====================
def find_best_glue_position(ver, glue_length=10.0, search_start=80.0, search_end=200.0, step=2.0):
    x = np.linspace(0, ver['L_total'], 500)
    cp = ver.get('complex_params', {})
    x_stretch = cp.get('x_stretch_start', 80.0)
    x_coil = cp.get('x_coil_end', 120.0)
    pitch_mode = cp.get('pitch_mode', 'constant')

    base_params = {k: ver[k] for k in ['E_core','G_core','E_hypo','G_hypo','D_o','D_i','w_s','L_total','F','T0','spring_start','spring_end']}
    base_glue = [(0, 1, 'full')]
    for g in ver['glue_intervals']:
        if g[0] > 200: base_glue.append(g)

    results = []
    for gs in np.arange(search_start, search_end - glue_length, step):
        ge = gs + glue_length
        if pitch_mode != 'constant':
            if not (ge <= x_stretch or gs >= x_coil):
                continue
        test_glue = list(base_glue) + [(gs, ge, 'core_spring')]
        params = dict(base_params); params['glue_intervals'] = test_glue
        try:
            _, GJ, _, _, _, _, _, _ = compute_version(
                x, ver['core_df'], ver['hypo_segments'], params, ver['eta'],
                smooth=True, spring_enabled=True,
                complex_params={**cp, 'eta_glue_peak': 0.90}
            )
            dGJ = np.abs(np.diff(GJ) / np.diff(x))
            max_slope = np.max(dGJ)
            mask = (x[:-1] >= gs - 5) & (x[:-1] <= ge + 5)
            local_slope = np.max(dGJ[mask]) if np.any(mask) else max_slope
            results.append((gs, ge, max_slope, local_slope))
        except Exception:
            continue
    if not results:
        return None, None, None, []
    results.sort(key=lambda r: r[3] + 0.3 * r[2])
    best = results[0]
    return best[0], best[1], best[2], results

def get_recommended_glue_position(ver, glue_length=10.0):
    name = ver.get('name', '')
    if 'Version 1' in name or '版本一' in name:
        fixed_start = 85.0; fixed_end = 95.0
        x = np.linspace(0, ver['L_total'], 500)
        cp = ver.get('complex_params', {})
        base_params = {k: ver[k] for k in ['E_core','G_core','E_hypo','G_hypo','D_o','D_i','w_s','L_total','F','T0','spring_start','spring_end']}
        base_glue = [(0, 1, 'full')]
        for g in ver['glue_intervals']:
            if g[0] > 200: base_glue.append(g)
        test_glue = list(base_glue) + [(fixed_start, fixed_end, 'core_spring')]
        params = dict(base_params); params['glue_intervals'] = test_glue
        try:
            _, GJ, _, _, _, _, _, _ = compute_version(
                x, ver['core_df'], ver['hypo_segments'], params, ver['eta'],
                smooth=True, spring_enabled=True,
                complex_params={**cp, 'eta_glue_peak': 0.90}
            )
            dGJ = np.abs(np.diff(GJ) / np.diff(x))
            max_slope = np.max(dGJ)
        except Exception:
            max_slope = 0.0
        return fixed_start, fixed_end, max_slope, [(fixed_start, fixed_end, max_slope, max_slope)]
    else:
        return find_best_glue_position(ver, glue_length=glue_length, search_start=80.0, search_end=200.0, step=2.0)

# ==================== 默认数据 ====================
default_core_v1 = pd.DataFrame([
    {"start":0,"end":15,"d_start":0.0508,"d_end":0.0508},
    {"start":15,"end":25,"d_start":0.0508,"d_end":0.0762},
    {"start":25,"end":100,"d_start":0.0762,"d_end":0.0762},
    {"start":100,"end":160,"d_start":0.0762,"d_end":0.127},
    {"start":160,"end":350,"d_start":0.127,"d_end":0.127},
])
default_core_v3 = pd.DataFrame([
    {"start":0,"end":15,"d_start":0.0508,"d_end":0.0508},
    {"start":15,"end":25,"d_start":0.0508,"d_end":0.0889},
    {"start":25,"end":70,"d_start":0.0889,"d_end":0.0889},
    {"start":70,"end":126,"d_start":0.0889,"d_end":0.14224},
    {"start":126,"end":175,"d_start":0.14224,"d_end":0.14224},
    {"start":175,"end":350,"d_start":0.14224,"d_end":0.22098},
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

eta_keys = ['full_b','full_t','full_a','core_spring_b','core_spring_t','core_spring_a',
            'core_hypo_b','core_hypo_t','core_hypo_a','spring_b','spring_t','spring_a',
            'no_spring_b','no_spring_t','no_spring_a']
eta_defaults = {
    'full_b':1.0,'full_t':1.0,'full_a':1.0,
    'core_spring_b':0.9,'core_spring_t':0.9,'core_spring_a':0.0,
    'core_hypo_b':1.0,'core_hypo_t':1.0,'core_hypo_a':1.0,
    'spring_b':0.9,'spring_t':0.6,'spring_a':0.0,
    'no_spring_b':0.85,'no_spring_t':0.35,'no_spring_a':0.0,
}

if 'name_input' not in st.session_state: st.session_state.name_input = "Version 1 (Step)"
if 'core_editor' not in st.session_state: st.session_state.core_editor = default_core_v1.copy()
if 'hypo_text_input' not in st.session_state: st.session_state.hypo_text_input = default_hypo_v1
if 'spring_start_input' not in st.session_state: st.session_state.spring_start_input = 0
if 'spring_end_input' not in st.session_state: st.session_state.spring_end_input = 150
if 'glue_input' not in st.session_state: st.session_state.glue_input = "0,1,full\n90,100,core_spring\n345,346,core_hypo"
for k in eta_keys:
    if k + '_input' not in st.session_state: st.session_state[k + '_input'] = eta_defaults[k]
if 'x_stretch_start_input' not in st.session_state: st.session_state.x_stretch_start_input = 80.0
if 'x_coil_end_input' not in st.session_state: st.session_state.x_coil_end_input = 120.0
if 'p_start_input' not in st.session_state: st.session_state.p_start_input = 0.045
if 'p_end_input' not in st.session_state: st.session_state.p_end_input = 0.061
if 'pitch_mode_input' not in st.session_state: st.session_state.pitch_mode_input = "恒定螺距"
if 'n_segments_input' not in st.session_state: st.session_state.n_segments_input = 3

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("加载预设版本")
    col1, col2, col3 = st.columns(3)
    if col1.button("版本一"):
        st.session_state.name_input = "Version 1 (Step)"
        st.session_state.core_editor = default_core_v1.copy()
        st.session_state.hypo_text_input = default_hypo_v1
        st.session_state.spring_start_input = 0
        st.session_state.spring_end_input = 150
        st.session_state.glue_input = "0,1,full\n90,100,core_spring\n345,346,core_hypo"
        for k in eta_keys: st.session_state[k + '_input'] = eta_defaults[k]
        st.rerun()
    if col2.button("版本二"):
        st.session_state.name_input = "Version 2 (Continuous)"
        st.session_state.core_editor = default_core_v1.copy()
        st.session_state.hypo_text_input = default_hypo_v2
        st.session_state.spring_start_input = 0
        st.session_state.spring_end_input = 150
        st.session_state.glue_input = "0,1,full\n90,100,core_spring\n345,346,core_hypo"
        for k in eta_keys: st.session_state[k + '_input'] = eta_defaults[k]
        st.rerun()
    if col3.button("版本三"):
        st.session_state.name_input = "Version 3 (New)"
        st.session_state.core_editor = default_core_v3.copy()
        st.session_state.hypo_text_input = default_hypo_v3
        st.session_state.spring_start_input = 0
        st.session_state.spring_end_input = 120
        st.session_state.glue_input = "0,1,full\n90,100,core_spring\n345,346,core_hypo"
        for k in eta_keys: st.session_state[k + '_input'] = eta_defaults[k]
        st.rerun()

    st.divider()
    st.header("当前版本编辑")
    st.text_input("版本名称", key="name_input")

    st.subheader("材料参数")
    E_core = st.number_input("芯丝杨氏模量 (MPa)", value=200000, step=1000, key="E_core_input")
    G_core = st.number_input("芯丝剪切模量 (MPa)", value=77000, step=1000, key="G_core_input")
    E_hypo = st.number_input("海波管杨氏模量 (MPa)", value=50000, step=1000, key="E_hypo_input")
    G_hypo = st.number_input("海波管剪切模量 (MPa)", value=19231, step=1000, key="G_hypo_input")

    st.subheader("几何参数")
    D_o = st.number_input("海波管外径 (mm)", value=0.33, step=0.01, key="D_o_input")
    D_i = st.number_input("海波管内径 (mm)", value=0.23, step=0.01, key="D_i_input")
    w_s = st.number_input("槽宽 (mm)", value=0.03, step=0.01, key="w_s_input")
    L_total = st.number_input("导丝总长 (mm)", value=350, step=10, key="L_total_input")

    st.subheader("载荷参数")
    F = st.number_input("远端横向力 F (N)", value=0.001, step=0.001, format="%.4f", key="F_input")
    T0 = st.number_input("近端扭矩 T0 (N·mm)", value=1.0, step=0.1, key="T0_input")

    st.subheader("弹簧圈范围")
    spring_start = st.number_input("弹簧圈起始位置 (mm)", step=5, key="spring_start_input")
    spring_end = st.number_input("弹簧圈结束位置 (mm)", step=5, key="spring_end_input")

    st.subheader("点胶区间")
    st.text_area("格式: start,end,type (每行一个)", key="glue_input")

    st.subheader("芯丝直径分段表")
    st.data_editor(st.session_state.core_editor, num_rows="dynamic", key="core_editor_widget")

    st.subheader("海波管开槽函数")
    uploaded_file = st.file_uploader("上传 Excel/CSV (可选)", type=["xlsx","xls","csv"], key="file_uploader")
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            if all(col in df_upload.columns for col in ['start','end','b_expr','Z_expr']):
                lines = [f"{r['start']},{r['end']},{r['b_expr']},{r['Z_expr']}" for _,r in df_upload.iterrows()]
                st.session_state.hypo_text_input = "\n".join(lines)
                st.success("文件已加载")
            else:
                st.error("缺少列")
        except Exception as e:
            st.error(f"读取出错: {e}")

    st.text_area("海波管函数", key="hypo_text_input", height=200)

    st.subheader("传递系数")
    with st.expander("完全点胶区 (full)"):
        st.number_input("弯曲", step=0.05, key="full_b_input")
        st.number_input("扭转", step=0.05, key="full_t_input")
        st.number_input("轴向", step=0.05, key="full_a_input")
    with st.expander("芯丝+弹簧圈点胶区 (core_spring)"):
        st.number_input("弯曲", step=0.05, key="core_spring_b_input")
        st.number_input("扭转", step=0.05, key="core_spring_t_input")
        st.number_input("轴向", step=0.05, key="core_spring_a_input")
    with st.expander("芯丝+海波管点胶区 (core_hypo)"):
        st.number_input("弯曲", step=0.05, key="core_hypo_b_input")
        st.number_input("扭转", step=0.05, key="core_hypo_t_input")
        st.number_input("轴向", step=0.05, key="core_hypo_a_input")
    with st.expander("有弹簧圈无点胶区"):
        st.number_input("弯曲", step=0.05, key="spring_b_input")
        st.number_input("扭转", step=0.05, key="spring_t_input")
        st.number_input("轴向", step=0.05, key="spring_a_input")
    with st.expander("无弹簧圈无点胶区"):
        st.number_input("弯曲", step=0.05, key="no_spring_b_input")
        st.number_input("扭转", step=0.05, key="no_spring_t_input")
        st.number_input("轴向", step=0.05, key="no_spring_a_input")

    st.subheader("弹簧圈螺距模式（可选）")
    pitch_mode = st.selectbox("螺距模式", ["恒定螺距", "分段螺距", "平滑渐变"], key="pitch_mode_input")
    with st.expander("螺距参数"):
        st.number_input("拉伸起点 (mm)", step=1.0, key="x_stretch_start_input")
        st.number_input("弹簧圈末端 (mm)", step=1.0, key="x_coil_end_input")
        st.number_input("初始螺距 (mm)", step=0.001, format="%.4f", key="p_start_input")
        if pitch_mode != "恒定螺距":
            st.number_input("末端螺距 (mm)", step=0.001, format="%.4f", key="p_end_input")
        if pitch_mode == "分段螺距":
            st.number_input("分段数", step=1, min_value=2, max_value=10, key="n_segments_input")

    if st.button("保存当前版本", type="primary"):
        glue_intervals = []
        glue_text = st.session_state.glue_input
        if glue_text.strip():
            for line in glue_text.strip().splitlines():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) == 3:
                    try: glue_intervals.append((float(parts[0]), float(parts[1]), parts[2]))
                    except: pass
        hypo_segments = parse_hypo_functions(st.session_state.hypo_text_input)
        eta = {k: st.session_state[k + '_input'] for k in eta_keys}
        pitch_mode_key = {'恒定螺距': 'constant', '分段螺距': 'segmented', '平滑渐变': 'gradient'}[st.session_state.pitch_mode_input]
        version = {
            'name': st.session_state.name_input,
            'E_core': E_core, 'G_core': G_core, 'E_hypo': E_hypo, 'G_hypo': G_hypo,
            'D_o': D_o, 'D_i': D_i, 'w_s': w_s, 'L_total': L_total, 'F': F, 'T0': T0,
            'spring_start': spring_start, 'spring_end': spring_end,
            'glue_intervals': glue_intervals,
            'core_df': st.session_state.core_editor.copy(),
            'hypo_segments': hypo_segments,
            'eta': eta,
            'complex_params': {
                'x_stretch_start': st.session_state.x_stretch_start_input,
                'x_coil_end': st.session_state.x_coil_end_input,
                'p_start': st.session_state.p_start_input,
                'p_end': st.session_state.p_end_input,
                'eta_glue_peak': 0.90,
                'pitch_mode': pitch_mode_key,
                'n_segments': st.session_state.get('n_segments_input', 3),
            }
        }
        st.session_state.saved_versions.append(version)
        st.success(f"版本 '{version['name']}' 已保存")

# ==================== 主区域 ====================
st.header("已保存版本")
if not st.session_state.saved_versions:
    st.info("请在左侧编辑参数并点击“保存当前版本”。")
else:
    for idx, ver in enumerate(st.session_state.saved_versions):
        col1, col2, col3 = st.columns([3,1,1])
        new_name = col1.text_input("版本名称", value=ver['name'], key=f"rename_{idx}", label_visibility="collapsed")
        if new_name != ver['name']: ver['name'] = new_name

        if col2.button("加载到左侧", key=f"load_{idx}"):
            st.session_state.name_input = ver['name']
            st.session_state.core_editor = ver['core_df'].copy()
            st.session_state.hypo_text_input = "\n".join([f"{seg[0]},{seg[1]},{seg[2]},{seg[3]}" for seg in ver['hypo_segments']])
            st.session_state.spring_start_input = ver['spring_start']
            st.session_state.spring_end_input = ver['spring_end']
            st.session_state.glue_input = "\n".join([f"{s},{e},{t}" for s,e,t in ver['glue_intervals']])
            for k in eta_keys:
                st.session_state[k + '_input'] = ver['eta'][k]
            cp = ver.get('complex_params', {})
            if cp:
                st.session_state.x_stretch_start_input = cp.get('x_stretch_start', 80.0)
                st.session_state.x_coil_end_input = cp.get('x_coil_end', 120.0)
                st.session_state.p_start_input = cp.get('p_start', 0.045)
                st.session_state.p_end_input = cp.get('p_end', 0.061)
                pm = cp.get('pitch_mode', 'constant')
                st.session_state.pitch_mode_input = {'constant':'恒定螺距','segmented':'分段螺距','gradient':'平滑渐变'}[pm]
                st.session_state.n_segments_input = cp.get('n_segments', 3)
            st.rerun()
        if col3.button("删除", key=f"del_{idx}"):
            st.session_state.saved_versions.pop(idx)
            st.rerun()

    st.subheader("选择要对比的版本")
    selected_indices = []
    for idx, ver in enumerate(st.session_state.saved_versions):
        if st.checkbox(ver['name'], key=f"check_{idx}"):
            selected_indices.append(idx)

    # ========== 生成对比曲线 ==========
    if st.button("生成对比曲线", type="primary"):
        if not selected_indices:
            st.warning("请至少选择一个版本")
        else:
            x = np.linspace(0, st.session_state.L_total_input, 500)
            fig1, axes1 = plt.subplots(3,1,figsize=(10,12)); fig1.suptitle("Stiffness Comparison")
            axes1[0].set_ylabel('Bending stiffness EI (N·mm²)')
            axes1[1].set_ylabel('Torsional stiffness GJ (N·mm²)')
            axes1[2].set_xlabel('Distance from distal end (mm)'); axes1[2].set_ylabel('Axial stiffness EA (N)')
            for ax in axes1: ax.grid(True)
            fig2, axes2 = plt.subplots(3,1,figsize=(10,12)); fig2.suptitle("Stress Comparison")
            axes2[0].set_ylabel('Bending normal stress (MPa)'); axes2[1].set_ylabel('Torsional shear stress (MPa)')
            axes2[2].set_xlabel('Distance from distal end (mm)'); axes2[2].set_ylabel('Von Mises stress (MPa)')
            for ax in axes2: ax.grid(True)
            fig3, axes3 = plt.subplots(2,1,figsize=(10,8)); fig3.suptitle("Deformation Comparison")
            axes3[0].set_ylabel('Deflection (mm)'); axes3[1].set_ylabel('Twist angle (rad)')
            axes3[1].set_xlabel('Distance from distal end (mm)')
            for ax in axes3: ax.grid(True)

            for idx in selected_indices:
                ver = st.session_state.saved_versions[idx]
                color = COLORS[idx % len(COLORS)]; label = ver['name']
                params = {k: ver[k] for k in ['E_core','G_core','E_hypo','G_hypo','D_o','D_i','w_s','L_total','F','T0','spring_start','spring_end','glue_intervals']}
                EI,GJ,EA,sb,tt,se,yd,ph = compute_version(x, ver['core_df'], ver['hypo_segments'], params, ver['eta'])
                axes1[0].plot(x,EI,color=color,linewidth=2,label=label); axes1[1].plot(x,GJ,color=color,linewidth=2,label=label)
                axes1[2].plot(x,EA,color=color,linewidth=2,label=label)
                axes2[0].plot(x,sb,color=color,linewidth=2,label=label); axes2[1].plot(x,tt,color=color,linewidth=2,label=label)
                axes2[2].plot(x,se,color=color,linewidth=2,label=label)
                axes3[0].plot(x,yd,color=color,linewidth=2,label=label); axes3[1].plot(x,ph,color=color,linewidth=2,label=label)
            axes1[0].legend(); axes1[1].legend(); axes1[2].legend()
            axes2[0].legend(); axes2[1].legend(); axes2[2].legend()
            axes3[0].legend(); axes3[1].legend()
            st.pyplot(fig1); st.pyplot(fig2); st.pyplot(fig3)

    # ========== 生成参数改进建议（只涉及海波管+芯丝） ==========
    if st.button("生成参数改进建议"):
        if not st.session_state.saved_versions:
            st.warning("请先保存版本")
        else:
            for ver in st.session_state.saved_versions:
                st.subheader(f"版本：{ver['name']}")
                suggestions = generate_suggestions(ver)
                for s in suggestions:
                    st.markdown(f"**{s['type']}** (位置: {s['position']})")
                    st.write(s['description'])
                    st.markdown(f"```\n{s['formula']}\n```")
                st.divider()

            st.markdown("---")
            st.markdown("### 原版 vs 平滑改进（仅海波管 + 芯丝）")
            st.markdown("""
            - **原版（蓝色）**：原始海波管分段和芯丝直径过渡
            - **平滑版（橙色虚线）**：海波管 Hermite 平滑 + 芯丝 S 形过渡
            """)

            for ver in st.session_state.saved_versions:
                st.subheader(f"对比：{ver['name']}")
                x = np.linspace(0, ver['L_total'], 500)
                base_params = {k: ver[k] for k in ['E_core','G_core','E_hypo','G_hypo','D_o','D_i','w_s','L_total','F','T0','spring_start','spring_end']}

                params_orig = dict(base_params)
                params_orig['glue_intervals'] = ver['glue_intervals']
                EI_o, GJ_o, EA_o, sb_o, tt_o, se_o, y_o, phi_o = compute_version(
                    x, ver['core_df'], ver['hypo_segments'], params_orig, ver['eta'],
                    smooth=False, spring_enabled=False)

                params_smooth = dict(base_params)
                params_smooth['glue_intervals'] = ver['glue_intervals']
                EI_s, GJ_s, EA_s, sb_s, tt_s, se_s, y_s, phi_s = compute_version(
                    x, ver['core_df'], ver['hypo_segments'], params_smooth, ver['eta'],
                    smooth=True, spring_enabled=False)

                fig_stiff, axes_stiff = plt.subplots(3, 1, figsize=(11, 12))
                axes_stiff[0].plot(x, EI_o, label='Original', color='blue', linewidth=2)
                axes_stiff[0].plot(x, EI_s, label='Smooth (hypo+core)', color='orange', linestyle='--', linewidth=2)
                axes_stiff[0].set_ylabel('Bending stiffness EI (N·mm²)'); axes_stiff[0].grid(True); axes_stiff[0].legend()
                axes_stiff[0].set_title(f"{ver['name']} - Bending stiffness")

                axes_stiff[1].plot(x, GJ_o, label='Original', color='blue', linewidth=2)
                axes_stiff[1].plot(x, GJ_s, label='Smooth (hypo+core)', color='orange', linestyle='--', linewidth=2)
                axes_stiff[1].set_ylabel('Torsional stiffness GJ (N·mm²)'); axes_stiff[1].grid(True); axes_stiff[1].legend()
                axes_stiff[1].set_xlim(0, 200)

                axes_stiff[2].plot(x, EA_o, label='Original', color='blue', linewidth=2)
                axes_stiff[2].plot(x, EA_s, label='Smooth (hypo+core)', color='orange', linestyle='--', linewidth=2)
                axes_stiff[2].set_ylabel('Axial stiffness EA (N)'); axes_stiff[2].set_xlabel('Distance from distal end (mm)')
                axes_stiff[2].grid(True); axes_stiff[2].legend()
                st.pyplot(fig_stiff)

                fig_def, axes_def = plt.subplots(2, 1, figsize=(11, 7))
                axes_def[0].plot(x, y_o, label='Original', color='blue', linewidth=2)
                axes_def[0].plot(x, y_s, label='Smooth (hypo+core)', color='orange', linestyle='--', linewidth=2)
                axes_def[0].set_ylabel('Deflection (mm)'); axes_def[0].grid(True); axes_def[0].legend()
                axes_def[1].plot(x, phi_o, label='Original', color='blue', linewidth=2)
                axes_def[1].plot(x, phi_s, label='Smooth (hypo+core)', color='orange', linestyle='--', linewidth=2)
                axes_def[1].set_ylabel('Twist angle (rad)'); axes_def[1].set_xlabel('Distance from distal end (mm)')
                axes_def[1].grid(True); axes_def[1].legend()
                st.pyplot(fig_def)
                st.divider()

    # ========== 自动推荐最优方案 ==========
    glue_length_2 = st.number_input("点胶长度 (mm)", min_value=1.0, max_value=50.0, value=10.0, step=0.5, key="glue_length_input_2")
    if st.button("自动推荐最优方案"):
        if not st.session_state.saved_versions:
            st.warning("请先保存版本")
        else:
            st.markdown("### 自动扫描点胶位置并推荐最优方案（离头端≥80mm）")
            st.markdown("**Version 1 (Step) 固定推荐 85–95 mm**，其他版本自动搜索。")
            glue_length = glue_length_2

            for ver in st.session_state.saved_versions:
                st.subheader(ver['name'])
                cp = ver.get('complex_params', {})
                pitch_mode = cp.get('pitch_mode', 'constant')
                mode_name = {'constant':'恒定螺距','segmented':'分段螺距','gradient':'平滑渐变'}[pitch_mode]
                st.write(f"弹簧圈螺距模式：**{mode_name}**")

                with st.spinner(f"正在确定 {ver['name']} 的推荐点胶位置..."):
                    best_start, best_end, best_slope, results = get_recommended_glue_position(
                        ver, glue_length=glue_length
                    )

                if best_start is None:
                    st.warning("未找到可行位置。")
                    continue

                name = ver.get('name', '')
                if 'Version 1' in name or '版本一' in name:
                    st.markdown(f"**推荐点胶位置：{best_start:.0f} – {best_end:.0f} mm**")
                else:
                    st.markdown(f"**推荐点胶位置：{best_start:.0f} – {best_end:.0f} mm**（最大斜率 {best_slope:.3f} N·mm²/mm）")
                    if len(results) > 1:
                        st.markdown("**候选排名（前5）：**")
                        for i, (gs, ge, ms, ls) in enumerate(results[:5]):
                            st.write(f"{i+1}. {gs:.0f}–{ge:.0f} mm，局部斜率 {ls:.3f}，全局斜率 {ms:.3f}")

                x = np.linspace(0, ver['L_total'], 500)
                base_params = {k: ver[k] for k in ['E_core','G_core','E_hypo','G_hypo','D_o','D_i','w_s','L_total','F','T0','spring_start','spring_end']}
                cp_full = {**cp, 'eta_glue_peak': 0.90}

                base_glue_orig = [(0, 1, 'full')]
                for g in ver['glue_intervals']:
                    if g[0] > 1 and g[0] <= 200: base_glue_orig.append(g)
                for g in ver['glue_intervals']:
                    if g[0] > 200: base_glue_orig.append(g)
                params_orig = dict(base_params); params_orig['glue_intervals'] = base_glue_orig
                _, GJ_orig, _, _, _, _, y_orig, phi_orig = compute_version(
                    x, ver['core_df'], ver['hypo_segments'], params_orig, ver['eta'],
                    smooth=True, spring_enabled=True, complex_params=cp_full)

                base_glue_best = [(0, 1, 'full')]
                for g in ver['glue_intervals']:
                    if g[0] > 200: base_glue_best.append(g)
                base_glue_best.append((best_start, best_end, 'core_spring'))
                params_best = dict(base_params); params_best['glue_intervals'] = base_glue_best
                _, GJ_best, _, _, _, _, y_best, phi_best = compute_version(
                    x, ver['core_df'], ver['hypo_segments'], params_best, ver['eta'],
                    smooth=True, spring_enabled=True, complex_params=cp_full)

                fig, axes = plt.subplots(2, 1, figsize=(11, 8))
                axes[0].plot(x, GJ_orig, label='Original', color='blue', linewidth=2)
                axes[0].plot(x, GJ_best, label=f'Recommended: {best_start:.0f}-{best_end:.0f} mm', color='green', linestyle='--', linewidth=2)
                axes[0].set_ylabel('Torsional stiffness GJ (N·mm²)')
                axes[0].grid(True); axes[0].legend()
                axes[0].set_xlim(0, 200)
                axes[1].plot(x, phi_orig, label='Original', color='blue', linewidth=2)
                axes[1].plot(x, phi_best, label='Recommended', color='green', linestyle='--', linewidth=2)
                axes[1].set_ylabel('Twist angle (rad)')
                axes[1].set_xlabel('Distance from distal end (mm)')
                axes[1].grid(True); axes[1].legend()
                axes[1].set_xlim(0, 200)
                st.pyplot(fig)
                st.divider()
