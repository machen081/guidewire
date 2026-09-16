import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

st.set_page_config(page_title="导丝刚度分析工具", layout="wide")

COLORS = ['green', 'blue', 'orange', 'purple', 'red', 'cyan', 'magenta', 'yellow', 'black', 'brown']
P_REF = 0.045  # 参考螺距 mm

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

# ==================== 恒定螺距传递系数 ====================
def eta_t_constant(x, glue_intervals, eta_base, eta_glue_peak, pitch, p_ref=P_REF):
    if x <= 1: return 1.0
    if pitch <= 0:
        base = eta_base
    else:
        base = eta_base * p_ref / pitch
    base = min(base, 0.95)
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

        # ==================== 弯曲传递系数（关键修改） ====================
        # 未点胶区域：弯曲时芯丝与海波管可以独立弯曲，海波管贡献很弱
        # 点胶区域：复合截面共同弯曲，海波管完全参与
        if 0 <= xi <= 1:
            eta_b = 1.0          # 头端三者点胶
        elif 90 <= xi <= 100:
            eta_b = 0.9          # 芯丝+弹簧圈点胶
        elif 345 <= xi <= 346:
            eta_b = 1.0          # 芯丝+海波管点胶
        else:
            if 0 <= xi <= 150:
                eta_b = 0.5      # 有弹簧圈无点胶
            else:
                eta_b = 0.2      # 无弹簧圈无点胶
        # 如果当前版本自定义了 eta，也可以覆盖（保留原逻辑用于对比）
        # ================================================================

        eta_a = eta_global['no_spring_a']
        for g_start, g_end, g_type in glue_intervals:
            if g_start <= xi < g_end:
                if g_type == 'full': eta_b = eta_global['full_b']; eta_a = eta_global['full_a']
                elif g_type == 'core_spring': eta_b = eta_global['core_spring_b']; eta_a = eta_global['core_spring_a']
                elif g_type == 'core_hypo': eta_b = eta_global['core_hypo_b']; eta_a = eta_global['core_hypo_a']
                break

        if spring_enabled and complex_params:
            eta_t = eta_t_constant(xi, glue_intervals,
                eta_global['spring_t'], complex_params['eta_glue_peak'],
                complex_params.get('pitch', P_REF))
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
    M_hypo = ratio_b*M_total
    T_hypo = ratio_t*T_total
    T_core = T_total - T_hypo

    t_wall = (D_o - D_i)/2; r_m = (D_o + D_i)/4
    sigma_bend = M_hypo/(2*b_arr*t_wall*r_m)
    tau_tors = T_hypo/(2*b_arr*t_wall*r_m)
    sigma_eq = np.sqrt(sigma_bend**2 + 3*tau_tors**2)

    theta = cumulative_trapezoid(M_total/EI_total, x, initial=0)
    theta = theta - theta[-1]
    y_def = cumulative_trapezoid(theta, x, initial=0)
    y_def = y_def - y_def[-1]
    phi = cumulative_trapezoid(T_total/GJ_total, x, initial=0)

    return (EI_total, GJ_total, EA_total, sigma_bend, tau_tors, sigma_eq, y_def, phi,
            T_hypo, T_core, ratio_t, eta_t_arr)

# ==================== 参数改进建议 ====================
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
def find_best_glue_position(ver, glue_length=5.0, search_start=70.0, step=0.5, must_cover_range=None):
    x = np.linspace(0, ver['L_total'], 1500)
    x_coil = ver.get('spring_end', 120.0)
    search_end = ver['L_total'] - glue_length

    if must_cover_range is not None:
        lo, hi = must_cover_range
        effective_start = max(search_start, hi - glue_length)
        effective_end = min(search_end, lo)
        if effective_start > effective_end:
            return None, None, None, [], 'none'
        scan_start = effective_start
        scan_end = effective_end
    else:
        scan_start = search_start
        scan_end = search_end

    base_params = {k: ver[k] for k in ['E_core','G_core','E_hypo','G_hypo','D_o','D_i','w_s','L_total','F','T0','spring_start','spring_end']}
    base_glue = [(0, 1, 'full')]
    for g in ver['glue_intervals']:
        if g[0] > 200: base_glue.append(g)

    def evaluate(gs, ge):
        test_glue = list(base_glue) + [(gs, ge, 'core_spring')]
        params = dict(base_params); params['glue_intervals'] = test_glue
        try:
            res = compute_version(
                x, ver['core_df'], ver['hypo_segments'], params, ver['eta'],
                smooth=True, spring_enabled=True,
                complex_params={**ver.get('complex_params', {}), 'eta_glue_peak': 0.90}
            )
            GJ = res[1]
            dGJ = np.abs(np.gradient(GJ, x))

            mask_local = (x >= 70) & (x <= 110)
            max_slope_local = np.max(dGJ[mask_local]) if np.any(mask_local) else np.max(dGJ)
            max_slope_global = np.max(dGJ)

            score = max_slope_local + 0.3 * max_slope_global
            return (gs, ge, max_slope_local, max_slope_global, score)
        except Exception:
            return None

    def scan(constraint_fn):
        results = []
        if scan_end < scan_start:
            return results
        for gs in np.arange(scan_start, scan_end + step, step):
            ge = gs + glue_length
            if ge > ver['L_total']:
                continue
            if must_cover_range is not None:
                lo, hi = must_cover_range
                if not (gs <= lo and ge >= hi):
                    continue
            if not constraint_fn(gs, ge):
                continue
            r = evaluate(gs, ge)
            if r is not None:
                results.append(r)
        results.sort(key=lambda r: r[4])
        return results

    results = scan(lambda gs, ge: ge <= x_coil)
    if results:
        b = results[0]
        return b[0], b[1], b[2], results, 'strict'

    overrun = max(glue_length * 0.3, 5.0)
    results = scan(lambda gs, ge: ge <= x_coil + overrun)
    if results:
        b = results[0]
        return b[0], b[1], b[2], results, 'overrun'

    results = scan(lambda gs, ge: True)
    if results:
        b = results[0]
        return b[0], b[1], b[2], results, 'any'

    return None, None, None, [], 'none'

def get_recommended_glue_position(ver, glue_length=5.0):
    name = ver.get('name', '')
    is_version1 = ('Version 1' in name) or ('版本一' in name)

    if is_version1:
        return find_best_glue_position(ver, glue_length=glue_length,
                                       search_start=70.0, step=0.5,
                                       must_cover_range=(89.0, 91.0))
    else:
        return find_best_glue_position(ver, glue_length=glue_length,
                                       search_start=80.0, step=0.5, must_cover_range=None)

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
    'spring_b':0.5,'spring_t':0.6,'spring_a':0.0,
    'no_spring_b':0.2,'no_spring_t':0.35,'no_spring_a':0.0,
}

if 'name_input' not in st.session_state: st.session_state.name_input = "Version 1 (Step)"
if 'core_editor' not in st.session_state: st.session_state.core_editor = default_core_v1.copy()
if 'hypo_text_input' not in st.session_state: st.session_state.hypo_text_input = default_hypo_v1
if 'spring_start_input' not in st.session_state: st.session_state.spring_start_input = 0
if 'spring_end_input' not in st.session_state: st.session_state.spring_end_input = 150
if 'glue_input' not in st.session_state: st.session_state.glue_input = "0,1,full\n90,100,core_spring\n345,346,core_hypo"
if 'pitch_input' not in st.session_state: st.session_state.pitch_input = 0.045
for k in eta_keys:
    if k + '_input' not in st.session_state: st.session_state[k + '_input'] = eta_defaults[k]

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
        st.session_state.pitch_input = 0.045
        for k in eta_keys: st.session_state[k + '_input'] = eta_defaults[k]
        st.rerun()
    if col2.button("版本二"):
        st.session_state.name_input = "Version 2 (Continuous)"
        st.session_state.core_editor = default_core_v1.copy()
        st.session_state.hypo_text_input = default_hypo_v2
        st.session_state.spring_start_input = 0
        st.session_state.spring_end_input = 150
        st.session_state.glue_input = "0,1,full\n90,100,core_spring\n345,346,core_hypo"
        st.session_state.pitch_input = 0.045
        for k in eta_keys: st.session_state[k + '_input'] = eta_defaults[k]
        st.rerun()
    if col3.button("版本三"):
        st.session_state.name_input = "Version 3 (New)"
        st.session_state.core_editor = default_core_v3.copy()
        st.session_state.hypo_text_input = default_hypo_v3
        st.session_state.spring_start_input = 0
        st.session_state.spring_end_input = 120
        st.session_state.glue_input = "0,1,full\n90,100,core_spring\n345,346,core_hypo"
        st.session_state.pitch_input = 0.045
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

    st.subheader("弹簧圈螺距")
    st.number_input("螺距 (mm)", min_value=0.010, max_value=0.200, step=0.001, format="%.4f", key="pitch_input")

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
                'eta_glue_peak': 0.90,
                'pitch': st.session_state.pitch_input,
            }
        }
        st.session_state.saved_versions.append(version)
        st.success(f"版本 '{version['name']}' 已保存")

# ==================== 主区域 ====================
st.header("Guidewire Multi-layer Stiffness Analysis")

with st.expander("📖 使用流程说明（点击展开）", expanded=False):
    st.markdown("""
### 一、定义导丝结构

本工具用于分析由**芯丝、弹簧圈、海波管、点胶区**组成的导丝，沿轴向的刚度、应力、变形分布。

- **芯丝**：实心圆截面，不锈钢材料，直径沿长度分段变化（线性过渡）。
- **海波管**：镍钛材料，切割开槽，通过开槽参数折减等效刚度。
- **弹簧圈**：通过机械互锁和点胶影响海波管与芯丝之间的耦合。
- **点胶区**：通过改变传递系数，局部增强或减弱耦合。

### 二、三种预设版本

点击侧边栏顶部的三个按钮，可快速加载不同设计方案：

1. **Version 1 (Step)**：海波管开槽参数每 10 mm 阶梯变化。
2. **Version 2 (Continuous)**：海波管 10–90 mm 连续多项式过渡。
3. **Version 3 (New)**：新设计，弹簧圈、芯丝、海波管参数均不同。

### 三、弯曲传递系数（关键）

**弯曲传递系数 \(\eta_{\text{bend}}\)** 表示海波管在弯曲中参与承载的比例：

| 区域 | \(\eta_{\text{bend}}\) | 物理含义 |
|------|------------------------|----------|
| 完全点胶 | 1.0 | 芯丝+弹簧圈+海波管三层粘接，复合截面共同弯曲 |
| 芯丝+弹簧圈点胶 | 0.9 | 弹簧圈固定，海波管部分跟随 |
| 芯丝+海波管点胶 | 1.0 | 两层粘接，完全共同弯曲 |
| 有弹簧圈无点胶 | 0.5 | 弹簧圈填充间隙，提供一定支撑和摩擦 |
| 无弹簧圈无点胶 | 0.2 | 芯丝与海波管仅靠接触摩擦，几乎不耦合 |

**弯曲时，未点胶区域芯丝与海波管可以相对滑动**，海波管的抗弯能力很难被芯丝调动，所以弯曲传递系数远低于扭转传递系数。

### 四、扭转传递系数

扭转时芯丝与海波管通过摩擦和机械互锁耦合，取值比弯曲高：

| 区域 | \(\eta_{\text{tors}}\) |
|------|------------------------|
| 完全点胶 | 1.0 |
| 芯丝+弹簧圈点胶 | 0.9 |
| 有弹簧圈无点胶 | 0.6 |
| 无弹簧圈无点胶 | 0.35 |

螺距影响：\(\eta_t = \eta_{\text{base}} \times p_{\text{ref}} / p\)，螺距越大，耦合越弱。

### 五、轴向传递系数

轴向刚度主要靠芯丝，海波管只在点胶处才参与：

| 区域 | \(\eta_{\text{axial}}\) |
|------|-------------------------|
| 完全点胶 | 1.0 |
| 芯丝+海波管点胶 | 1.0 |
| 其他区域 | 0.0 |

### 六、参数编辑与保存

- 在侧边栏编辑材料、几何、载荷、弹簧圈、点胶、芯丝、海波管、传递系数。
- 点击**“保存当前版本”**将当前参数存入列表。
- 可保存多个版本，在主区域勾选对比。

### 七、生成对比曲线

勾选已保存版本后点击**“生成对比曲线”**，可得到：

- 弯曲刚度 EI、扭转刚度 GJ、轴向刚度 EA
- 海波管连接筋弯曲正应力、扭转剪切应力、Von Mises 等效应力
- 弯曲挠度、扭转角

### 八、生成参数改进建议

点击**“生成参数改进建议”**后，程序依次显示：

1. **改进建议**：海波管分段突变、芯丝直径线性过渡的量化建议。
2. **原版 vs 平滑改进**：海波管分段用 Hermite 过渡、芯丝直径用 S 形过渡。
3. **自动推荐点胶位置**：在 70–110 mm 区间以 0.5 mm 步长扫描，评分指标为该区间的最大斜率。Version 1 强制覆盖 89–91 mm。
4. **原版 vs 推荐版**：显示推荐点胶位置对扭转刚度、扭转角和力传递的影响。

### 九、最小弯曲半径

导丝的最小弯曲半径 \(R_{\min}\) 与弯曲刚度和许用应变相关：

\[
R_{\min} = \frac{EI_{\text{total}}}{M_{\max}} \quad \text{或} \quad R_{\min} = \frac{r_{\text{outer}}}{\varepsilon_{\text{allow}}}
\]

**点胶长度增加 → 高刚度段变长 → 整体弯曲刚度提高 → 相同弯矩下曲率变小 → 最小弯曲半径增大。**

如果希望最小弯曲半径不要增大太多，可以：
- 缩短点胶长度；
- 在点胶区局部减小海波管连接筋宽度；
- 使用低模量胶水；
- 把点胶区放在弯矩较小的位置（如近端）。
    """)

# ==================== 已保存版本 ====================
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
            st.session_state.pitch_input = cp.get('pitch', P_REF)
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
                res = compute_version(x, ver['core_df'], ver['hypo_segments'], params, ver['eta'])
                EI,GJ,EA,sb,tt,se,yd,ph = res[0],res[1],res[2],res[3],res[4],res[5],res[6],res[7]
                axes1[0].plot(x,EI,color=color,linewidth=2,label=label); axes1[1].plot(x,GJ,color=color,linewidth=2,label=label)
                axes1[2].plot(x,EA,color=color,linewidth=2,label=label)
                axes2[0].plot(x,sb,color=color,linewidth=2,label=label); axes2[1].plot(x,tt,color=color,linewidth=2,label=label)
                axes2[2].plot(x,se,color=color,linewidth=2,label=label)
                axes3[0].plot(x,yd,color=color,linewidth=2,label=label); axes3[1].plot(x,ph,color=color,linewidth=2,label=label)
            axes1[0].legend(); axes1[1].legend(); axes1[2].legend()
            axes2[0].legend(); axes2[1].legend(); axes2[2].legend()
            axes3[0].legend(); axes3[1].legend()
            st.pyplot(fig1); st.pyplot(fig2); st.pyplot(fig3)

    glue_length_input = st.number_input("点胶长度 (mm)", min_value=1.0, max_value=50.0, value=5.0, step=0.5, key="glue_length_input_combined")

    if st.button("生成参数改进建议"):
        if not st.session_state.saved_versions:
            st.warning("请先保存版本")
        else:
            glue_length = glue_length_input

            for ver in st.session_state.saved_versions:
                st.markdown(f"## 版本：{ver['name']}")

                st.markdown("### 改进建议（海波管 + 芯丝）")
                suggestions = generate_suggestions(ver)
                for s in suggestions:
                    st.markdown(f"**{s['type']}** (位置: {s['position']})")
                    st.write(s['description'])
                    st.markdown(f"```\n{s['formula']}\n```")
                st.divider()

                st.markdown("### 原版 vs 平滑改进（仅海波管 + 芯丝）")
                x = np.linspace(0, ver['L_total'], 500)
                base_params = {k: ver[k] for k in ['E_core','G_core','E_hypo','G_hypo','D_o','D_i','w_s','L_total','F','T0','spring_start','spring_end']}

                params_orig = dict(base_params)
                params_orig['glue_intervals'] = ver['glue_intervals']
                res_o = compute_version(x, ver['core_df'], ver['hypo_segments'], params_orig, ver['eta'],
                                        smooth=False, spring_enabled=False)
                EI_o, GJ_o, EA_o, y_o, phi_o = res_o[0], res_o[1], res_o[2], res_o[6], res_o[7]

                params_smooth = dict(base_params)
                params_smooth['glue_intervals'] = ver['glue_intervals']
                res_s = compute_version(x, ver['core_df'], ver['hypo_segments'], params_smooth, ver['eta'],
                                        smooth=True, spring_enabled=False)
                EI_s, GJ_s, EA_s, y_s, phi_s = res_s[0], res_s[1], res_s[2], res_s[6], res_s[7]

                fig_stiff, axes_stiff = plt.subplots(3, 1, figsize=(11, 12))
                axes_stiff[0].plot(x, EI_o, label='Original', color='blue', linewidth=2)
                axes_stiff[0].plot(x, EI_s, label='Smooth', color='orange', linestyle='--', linewidth=2)
                axes_stiff[0].set_ylabel('Bending stiffness EI (N·mm²)'); axes_stiff[0].grid(True); axes_stiff[0].legend()
                axes_stiff[0].set_title(f"{ver['name']} - Bending stiffness")
                axes_stiff[1].plot(x, GJ_o, label='Original', color='blue', linewidth=2)
                axes_stiff[1].plot(x, GJ_s, label='Smooth', color='orange', linestyle='--', linewidth=2)
                axes_stiff[1].set_ylabel('Torsional stiffness GJ (N·mm²)'); axes_stiff[1].grid(True); axes_stiff[1].legend()
                axes_stiff[1].set_xlim(0, 200)
                axes_stiff[2].plot(x, EA_o, label='Original', color='blue', linewidth=2)
                axes_stiff[2].plot(x, EA_s, label='Smooth', color='orange', linestyle='--', linewidth=2)
                axes_stiff[2].set_ylabel('Axial stiffness EA (N)'); axes_stiff[2].set_xlabel('Distance from distal end (mm)')
                axes_stiff[2].grid(True); axes_stiff[2].legend()
                st.pyplot(fig_stiff)

                fig_def, axes_def = plt.subplots(2, 1, figsize=(11, 7))
                axes_def[0].plot(x, y_o, label='Original', color='blue', linewidth=2)
                axes_def[0].plot(x, y_s, label='Smooth', color='orange', linestyle='--', linewidth=2)
                axes_def[0].set_ylabel('Deflection (mm)'); axes_def[0].grid(True); axes_def[0].legend()
                axes_def[1].plot(x, phi_o, label='Original', color='blue', linewidth=2)
                axes_def[1].plot(x, phi_s, label='Smooth', color='orange', linestyle='--', linewidth=2)
                axes_def[1].set_ylabel('Twist angle (rad)'); axes_def[1].set_xlabel('Distance from distal end (mm)')
                axes_def[1].grid(True); axes_def[1].legend()
                st.pyplot(fig_def)
                st.divider()

                st.markdown("### 自动推荐点胶位置（目标：70–110 mm 区间内最大斜率最小）")
                cp = ver.get('complex_params', {})
                x_coil = ver.get('spring_end', 120.0)
                st.write(f"弹簧圈末端：**{x_coil:.0f} mm**　|　螺距：**{cp.get('pitch', P_REF):.4f} mm**　|　点胶长度：**{glue_length:.1f} mm**")

                with st.spinner(f"正在扫描 {ver['name']} 的推荐点胶位置..."):
                    best_start, best_end, best_score, results, status = get_recommended_glue_position(
                        ver, glue_length=glue_length
                    )

                if status == 'none' or best_start is None:
                    st.error("无法找到可行位置（可能点胶长度太小，无法覆盖 89-91 mm）。")
                    st.divider()
                    continue

                status_msgs = {
                    'strict': ('success', '严格约束满足（终点≤弹簧圈末端）'),
                    'overrun': ('warning', '严格约束下无可行位置，点胶终点已略微超过弹簧圈末端'),
                    'any': ('warning', '约束极紧，已在所有可能位置中选择最优'),
                }
                level, msg = status_msgs.get(status, ('info', ''))
                if level == 'success': st.success(msg)
                elif level == 'warning': st.warning(msg)
                elif level == 'info': st.info(msg)

                st.markdown(f"**推荐点胶位置：{best_start:.1f} – {best_end:.1f} mm**（70–110 mm 局部最大斜率 {best_score:.4f}）")

                if len(results) > 1:
                    st.markdown("**候选排名（按 70–110 mm 局部最大斜率最小）：**")
                    for i, r in enumerate(results[:15]):
                        gs, ge, ms_local, ms_global, sc = r
                        st.write(f"{i+1}. {gs:.1f}–{ge:.1f} mm，局部最大斜率 {ms_local:.4f}，全局最大斜率 {ms_global:.4f}，总分 {sc:.4f}")

                cp_full = {**cp, 'eta_glue_peak': 0.90}

                base_glue_orig = [(0, 1, 'full')]
                for g in ver['glue_intervals']:
                    if g[0] > 1 and g[0] <= 200: base_glue_orig.append(g)
                for g in ver['glue_intervals']:
                    if g[0] > 200: base_glue_orig.append(g)
                params_orig2 = dict(base_params); params_orig2['glue_intervals'] = base_glue_orig
                res_orig = compute_version(x, ver['core_df'], ver['hypo_segments'], params_orig2, ver['eta'],
                                           smooth=True, spring_enabled=True, complex_params=cp_full)
                GJ_orig, phi_orig, rt_orig = res_orig[1], res_orig[7], res_orig[10]

                base_glue_best = [(0, 1, 'full')]
                for g in ver['glue_intervals']:
                    if g[0] > 200: base_glue_best.append(g)
                base_glue_best.append((best_start, best_end, 'core_spring'))
                params_best = dict(base_params); params_best['glue_intervals'] = base_glue_best
                res_best = compute_version(x, ver['core_df'], ver['hypo_segments'], params_best, ver['eta'],
                                           smooth=True, spring_enabled=True, complex_params=cp_full)
                GJ_best, phi_best, Th_best, Tc_best, rt_best = res_best[1], res_best[7], res_best[8], res_best[9], res_best[10]

                fig, axes = plt.subplots(2, 1, figsize=(11, 8))
                axes[0].plot(x, GJ_orig, label='Original', color='blue', linewidth=2)
                axes[0].plot(x, GJ_best, label=f'Recommended: {best_start:.1f}-{best_end:.1f} mm', color='green', linestyle='--', linewidth=2)
                axes[0].axvline(90, color='red', linestyle=':', alpha=0.4, label='90 mm')
                axes[0].set_ylabel('Torsional stiffness GJ (N·mm²)')
                axes[0].grid(True); axes[0].legend()
                axes[0].set_xlim(0, 200)
                axes[1].plot(x, phi_orig, label='Original', color='blue', linewidth=2)
                axes[1].plot(x, phi_best, label='Recommended', color='green', linestyle='--', linewidth=2)
                axes[1].axvline(90, color='red', linestyle=':', alpha=0.4)
                axes[1].set_ylabel('Twist angle (rad)')
                axes[1].set_xlabel('Distance from distal end (mm)')
                axes[1].grid(True); axes[1].legend()
                axes[1].set_xlim(0, 200)
                st.pyplot(fig)

                st.markdown("#### 力传递：海波管承担的扭矩比例")
                fig_ft, axes_ft = plt.subplots(1, 1, figsize=(11, 4))
                axes_ft.plot(x, rt_orig, label='Original', color='blue', linewidth=2)
                axes_ft.plot(x, rt_best, label='Recommended', color='green', linestyle='--', linewidth=2)
                axes_ft.axvline(best_start, color='green', linestyle=':', alpha=0.5)
                axes_ft.axvline(best_end, color='green', linestyle=':', alpha=0.5)
                axes_ft.axvline(90, color='red', linestyle=':', alpha=0.4)
                axes_ft.set_ylabel('T_hypo / T_total')
                axes_ft.set_xlabel('Distance from distal end (mm)')
                axes_ft.grid(True); axes_ft.legend()
                axes_ft.set_xlim(0, 200)
                st.pyplot(fig_ft)

                st.divider()
