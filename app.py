import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

st.set_page_config(page_title="导丝刚度分析工具", layout="wide")

COLORS = ['green', 'blue', 'orange', 'purple', 'red', 'cyan', 'magenta', 'yellow', 'black', 'brown']
P_REF = 0.045

# ==================== 安全表达式求值 ====================
def safe_eval(expr, x_val):
    allowed = set("0123456789+-*/(). xXeE")
    if any(ch not in allowed for ch in expr):
        raise ValueError(f"表达式包含非法字符: {expr}")
    expr_sub = expr.replace('X', 'x').replace('x', f'({x_val})')
    return float(eval(expr_sub))

def safe_eval_array(expr, x_arr):
    return np.array([safe_eval(expr, xi) for xi in x_arr], dtype=float)

# ==================== 移动平均 ====================
def moving_average(arr, window):
    window = int(window)
    if window < 1:
        return np.asarray(arr, dtype=float).copy()
    if window % 2 == 0:
        window += 1
    s = pd.Series(np.asarray(arr, dtype=float))
    return s.rolling(window=window, center=True, min_periods=1).mean().to_numpy()

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
                        dZ_r = (safe_eval(next_seg[3], boundary+2*eps) - safe_eval(next_seg[3], boundary-eps))/eps
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

# ==================== 构造位置相关的传递系数 η(x) ====================
def build_eta_arrays(x_arr, glue_intervals, spring_start, spring_end, pitch,
                     eta_global, use_pitch_corr=False):
    n = len(x_arr)
    eta_b_x = np.zeros(n); eta_t_x = np.zeros(n); eta_a_x = np.zeros(n)
    for i, xi in enumerate(x_arr):
        eta_b = eta_global['no_spring_b']
        eta_t = eta_global['no_spring_t']
        eta_a = eta_global['no_spring_a']

        in_spring = spring_start <= xi < spring_end
        if in_spring:
            eta_b = eta_global['spring_b']
            eta_t = eta_global['spring_t']
            eta_a = eta_global['spring_a']

        in_glue = False
        for g_start, g_end, g_type in glue_intervals:
            if g_start <= xi < g_end:
                in_glue = True
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

        if use_pitch_corr and in_spring and not in_glue and pitch > 0:
            eta_t = min(eta_t * P_REF / pitch, 0.95)

        eta_b_x[i] = eta_b; eta_t_x[i] = eta_t; eta_a_x[i] = eta_a
    return eta_b_x, eta_t_x, eta_a_x

# ==================== 计算函数（正问题） ====================
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

        if 0 <= xi <= 1:
            eta_b = 1.0
        elif 90 <= xi <= 100:
            eta_b = 0.9
        elif 345 <= xi <= 346:
            eta_b = 1.0
        else:
            if 0 <= xi <= 150:
                eta_b = 0.9
            else:
                eta_b = 0.85

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

    Y = 0.5184 - b_arr
    Z_safe = np.where(Z_arr > 0, Z_arr, 1e-9)
    b_safe = np.where(b_arr > 0, b_arr, 1e-9)
    k = 1.0 / (1.0 + (w_s / Z_safe) * (Y / b_safe))
    k = np.where((Z_arr > 0) & (b_arr > 0) & (b_arr < 0.5184), k, np.nan)
    k = np.clip(k, 0.0, 1.0)
    k = np.nan_to_num(k, nan=0.0)

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
    'spring_b':0.9,'spring_t':0.6,'spring_a':0.0,
    'no_spring_b':0.85,'no_spring_t':0.35,'no_spring_a':0.0,
}

if 'name_input' not in st.session_state: st.session_state.name_input = "Version 1 (Step)"
if 'core_editor_data' not in st.session_state: st.session_state.core_editor_data = default_core_v1.copy()
if 'core_editor_version' not in st.session_state: st.session_state.core_editor_version = 0
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

    def load_preset(name, core_df, hypo_text, s_start, s_end, glue, pitch):
        st.session_state.name_input = name
        st.session_state.core_editor_data = core_df.copy()
        st.session_state.core_editor_version += 1
        st.session_state.hypo_text_input = hypo_text
        st.session_state.spring_start_input = s_start
        st.session_state.spring_end_input = s_end
        st.session_state.glue_input = glue
        st.session_state.pitch_input = pitch
        for kk in eta_keys: st.session_state[kk + '_input'] = eta_defaults[kk]

    if col1.button("版本一"):
        load_preset("Version 1 (Step)", default_core_v1, default_hypo_v1, 0, 150,
                    "0,1,full\n90,100,core_spring\n345,346,core_hypo", 0.045)
        st.rerun()
    if col2.button("版本二"):
        load_preset("Version 2 (Continuous)", default_core_v1, default_hypo_v2, 0, 150,
                    "0,1,full\n90,100,core_spring\n345,346,core_hypo", 0.045)
        st.rerun()
    if col3.button("版本三"):
        load_preset("Version 3 (New)", default_core_v3, default_hypo_v3, 0, 120,
                    "0,1,full\n90,100,core_spring\n345,346,core_hypo", 0.045)
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
    edited_core_df = st.data_editor(
        st.session_state.core_editor_data,
        num_rows="dynamic",
        key=f"core_editor_widget_{st.session_state.core_editor_version}"
    )
    st.session_state.core_editor_data = edited_core_df

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
            'core_df': st.session_state.core_editor_data.copy(),
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
    st.markdown(r"""
# 导丝刚度分析工具使用说明

## 0. 快速开始

1. 加载预设：左侧边栏“版本一 / 版本二 / 版本三”。
2. 修改参数：材料、几何、载荷、弹簧圈、点胶、芯丝、海波管、传递系数。
3. 保存版本：使用左侧保存功能，将参数存入“已保存版本”。
4. 查看正问题结果：勾选版本后查看 EI、GJ、EA、应力、变形对比曲线。
5. 查看改进建议：平滑方案、推荐点胶位置、原版/推荐版对比。
6. 反问题设计：主区域底部“反问题求解”支持两种模式。

## 反问题两种模式

- **反解连接筋宽度 b(x)**：固定 Z、w_s，给定目标 EI/GJ/EA 和芯丝直径 d(x)，反解 b(x)。
- **反解芯丝直径 d(x)**：固定海波管函数 b(x)、Z(x)，给定目标 EI/GJ/EA，反解 d(x)。

传递系数 η_b、η_t、η_a 自动读取侧边栏设置，与正问题一致。
""")

    st.markdown(r"""
---

## 1. 正问题

正问题的目标：给定几何、材料、载荷、点胶、弹簧圈参数，计算导丝各位置的 EI、GJ、EA、应力、挠度、扭转角。

### 关键公式

完整海波管刚度：

- EI0 = E_hypo * pi * (D_o^4 - D_i^4) / 64
- GJ0 = G_hypo * pi * (D_o^4 - D_i^4) / 32
- EA0 = E_hypo * pi * (D_o^2 - D_i^2) / 4

开槽折减系数（Z 为轴向周期）：

k = 1 / (1 + (w_s / Z) * (0.5184 - b) / b)

总刚度：

- EI_total = EI_core + eta_b * k * EI0
- GJ_total = GJ_core + eta_t * k * GJ0
- EA_total = EA_core + eta_a * k * EA0

---

## 2. 反问题：反解连接筋宽度 b(x)

固定 Z 和 w_s，给定目标 EI/GJ/EA 和芯丝直径 d(x)。

由目标刚度反推 k：

k_S(x) = (S_target(x) - S_core(x)) / (eta_S(x) * S_0)

反解 b：

b = 0.5184 * k * w_s / (k * w_s + (1 - k) * Z)

---

## 3. 反问题：反解芯丝直径 d(x)

固定海波管函数 b(x)、Z(x)（从侧边栏读取），给定目标 EI/GJ/EA。

先算海波管等效刚度：

- EI_hypo = k(x) * EI0
- GJ_hypo = k(x) * GJ0
- EA_hypo = k(x) * EA0

目标芯丝刚度：

- EI_core_target = EI_target - eta_b(x) * EI_hypo
- GJ_core_target = GJ_target - eta_t(x) * GJ_hypo
- EA_core_target = EA_target - eta_a(x) * EA_hypo

反解 d（解析，无需迭代）：

- d_EI = (64 * EI_core_target / (E_core * pi))^(1/4)
- d_GJ = (32 * GJ_core_target / (G_core * pi))^(1/4)
- d_EA = (4 * EA_core_target / (E_core * pi))^(1/2)

物理约束：

- 目标芯丝刚度 > 0，否则无解
- d > 0
- d < D_i（芯丝必须在海波管内）

---

## 4. 物理约束与常见问题

| 约束 | 说明 |
|---|---|
| 目标刚度 > 芯丝刚度（反解 b） | 否则无解 |
| 目标刚度 > 海波管贡献（反解 d） | 否则无解 |
| 0 < k < 1 | 折减系数合理范围 |
| 0 < b < 0.5184 | 连接筋宽在半周长内 |
| Z > w_s | 轴向周期大于槽宽 |
| d < D_i | 芯丝必须装在海波管内 |
""")

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
            st.session_state.core_editor_data = ver['core_df'].copy()
            st.session_state.core_editor_version += 1
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

                params_orig = dict(base_params); params_orig['glue_intervals'] = ver['glue_intervals']
                res_o = compute_version(x, ver['core_df'], ver['hypo_segments'], params_orig, ver['eta'],
                                        smooth=False, spring_enabled=False)
                EI_o, GJ_o, EA_o, y_o, phi_o = res_o[0], res_o[1], res_o[2], res_o[6], res_o[7]

                params_smooth = dict(base_params); params_smooth['glue_intervals'] = ver['glue_intervals']
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

# ==================== 反问题求解 ====================
st.divider()
st.header("🔧 反问题求解：从目标刚度反推几何参数")

inv_mode = st.radio(
    "反解对象",
    ["反解连接筋宽度 b(x)（固定 Z, w_s）",
     "反解芯丝直径 d(x)（固定海波管函数）"],
    key="inv_mode"
)

with st.expander("📖 反问题求解说明（点击展开）", expanded=False):
    st.markdown(r"""
### 一、反解对象

- **反解连接筋宽度 b(x)**：固定轴向周期 Z 和槽宽 w_s，给定目标 EI/GJ/EA 和芯丝直径 d(x)，反解 b(x)。
- **反解芯丝直径 d(x)**：固定海波管函数 b(x)、Z(x)（从侧边栏读取），给定目标 EI/GJ/EA，反解 d(x)。

### 二、反解公式

**折减系数（正问题）：**

k = 1 / (1 + (w_s / Z) * (0.5184 - b) / b)

**反解 b：**

k_S(x) = (S_target(x) - S_core(x)) / (eta_S(x) * S_0)

b = 0.5184 * k * w_s / (k * w_s + (1 - k) * Z)

**反解 d：**

先算海波管等效刚度：

- EI_hypo = k(x) * EI0
- GJ_hypo = k(x) * GJ0
- EA_hypo = k(x) * EA0

再算目标芯丝刚度：

- EI_core_target = EI_target - eta_b(x) * EI_hypo
- GJ_core_target = GJ_target - eta_t(x) * GJ_hypo
- EA_core_target = EA_target - eta_a(x) * EA_hypo

反解 d：

- d_EI = (64 * EI_core_target / (E_core * pi))^(1/4)
- d_GJ = (32 * GJ_core_target / (G_core * pi))^(1/4)
- d_EA = (4 * EA_core_target / (E_core * pi))^(1/2)

### 三、传递系数的处理

反问题会自动读取侧边栏的“点胶区间”“弹簧圈范围”“传递系数”，构造位置相关的 η_b(x)、η_t(x)、η_a(x)，与正问题一致。

### 四、物理约束

| 约束 | 说明 |
|---|---|
| 目标刚度 > 芯丝刚度（反解 b） | 否则无解 |
| 目标刚度 > 海波管贡献（反解 d） | 否则无解 |
| 0 < k < 1 | 折减系数合理范围 |
| 0 < b < 0.5184 | 连接筋宽在半周长内 |
| Z > w_s | 轴向周期大于槽宽 |
| d < D_i | 芯丝必须装在海波管内 |
    """)

col_inv1, col_inv2 = st.columns(2)

with col_inv1:
    inv_EI_expr = st.text_area(
        "目标弯曲刚度 EI_target(x)，单位 N·mm²",
        value="5 + 0.05*x",
        height=90, key="inv_EI_expr"
    )
    inv_GJ_expr = st.text_area(
        "目标扭转刚度 GJ_target(x)，单位 N·mm²（可选）",
        value="", height=90, key="inv_GJ_expr"
    )
    inv_EA_expr = st.text_area(
        "目标轴向刚度 EA_target(x)，单位 N（可选）",
        value="", height=90, key="inv_EA_expr"
    )

with col_inv2:
    inv_primary = st.radio("反解主导刚度", ["EI", "GJ", "EA"], horizontal=True, key="inv_primary")
    inv_smooth_window = st.number_input("移动平均窗口（采样点数，偶数自动+1）",
                                        min_value=1, max_value=101, value=11, step=1, key="inv_smooth_window")
    inv_use_pitch_corr = st.checkbox("弹簧圈区未点胶位置应用螺距修正（扭转）", value=False, key="inv_pitch_corr")

# ---- 反解 b 的专属参数 ----
if "反解连接筋宽度" in inv_mode:
    st.markdown("**反解 b 的固定参数**")
    col_z1, col_z2 = st.columns(2)
    with col_z1:
        inv_Z = st.number_input("轴向周期 Z（相邻切槽中心距，mm）", value=0.133, step=0.001, format="%.4f", key="inv_Z")
    with col_z2:
        inv_w_s = st.number_input("槽宽 w_s（固定值，mm）", value=0.03, step=0.001, format="%.4f", key="inv_w_s")
    inv_d_expr = st.text_area(
        "芯丝直径 d(x)，单位 mm",
        value="0.05 + 0.0003*x",
        height=90, key="inv_d_expr"
    )
else:
    st.markdown("**反解 d 的固定参数**")
    col_z1, col_z2 = st.columns(2)
    with col_z1:
        inv_w_s_d = st.number_input("槽宽 w_s（固定值，mm）", value=0.03, step=0.001, format="%.4f", key="inv_w_s_d")
    with col_z2:
        inv_d_max = st.number_input("芯丝直径上限（通常取 D_i，mm）", value=0.23, step=0.01, format="%.4f", key="inv_d_max")
    st.info("海波管函数 b(x)、Z(x) 来自左侧边栏“海波管函数”。")

st.info("反问题使用的传递系数来自左侧边栏的“点胶区间”“弹簧圈范围”“传递系数”。如需修改，请先调整左侧参数。")

# ==================== 反解 b ====================
if "反解连接筋宽度" in inv_mode:
    if st.button("求解反问题（反解 b）", key="inverse_solve_b_btn", type="primary"):
        try:
            inv_E_core = st.session_state.get('E_core_input', 200000.0)
            inv_G_core = st.session_state.get('G_core_input', 77000.0)
            inv_E_hypo = st.session_state.get('E_hypo_input', 50000.0)
            inv_G_hypo = st.session_state.get('G_hypo_input', 19231.0)
            inv_D_o = st.session_state.get('D_o_input', 0.33)
            inv_D_i = st.session_state.get('D_i_input', 0.23)
            inv_L_total = float(st.session_state.get('L_total_input', 350.0))

            if inv_Z <= inv_w_s:
                st.error(f"轴向周期 Z = {inv_Z} mm 必须大于槽宽 w_s = {inv_w_s} mm。")
                st.stop()

            x_inv = np.linspace(0, inv_L_total, 500)

            d_core_inv = safe_eval_array(inv_d_expr, x_inv)
            if np.any(d_core_inv <= 0):
                st.warning("芯丝直径 d(x) 存在 ≤ 0 的位置，已钳制到 1e-6 mm。")
                d_core_inv = np.maximum(d_core_inv, 1e-6)

            EI_core = inv_E_core * np.pi * d_core_inv**4 / 64.0
            GJ_core = inv_G_core * np.pi * d_core_inv**4 / 32.0
            EA_core = inv_E_core * np.pi * d_core_inv**2 / 4.0

            I_0 = np.pi / 64.0 * (inv_D_o**4 - inv_D_i**4)
            J_0 = 2.0 * I_0
            A_0 = np.pi / 4.0 * (inv_D_o**2 - inv_D_i**2)
            EI_0 = inv_E_hypo * I_0
            GJ_0 = inv_G_hypo * J_0
            EA_0 = inv_E_hypo * A_0

            EI_target = safe_eval_array(inv_EI_expr, x_inv) if inv_EI_expr.strip() else None
            GJ_target = safe_eval_array(inv_GJ_expr, x_inv) if inv_GJ_expr.strip() else None
            EA_target = safe_eval_array(inv_EA_expr, x_inv) if inv_EA_expr.strip() else None

            if EI_target is None and GJ_target is None and EA_target is None:
                st.error("至少需要提供一个目标刚度表达式。")
                st.stop()

            # 从侧边栏读取传递系数设置
            glue_intervals_inv = []
            glue_text_inv = st.session_state.get('glue_input', '')
            if glue_text_inv.strip():
                for line in glue_text_inv.strip().splitlines():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) == 3:
                        try: glue_intervals_inv.append((float(parts[0]), float(parts[1]), parts[2]))
                        except: pass

            spring_start_inv = st.session_state.get('spring_start_input', 0)
            spring_end_inv = st.session_state.get('spring_end_input', 150)
            pitch_inv = st.session_state.get('pitch_input', P_REF)
            eta_global_inv = {k: st.session_state.get(k + '_input', eta_defaults[k]) for k in eta_keys}

            eta_b_x, eta_t_x, eta_a_x = build_eta_arrays(
                x_inv, glue_intervals_inv, spring_start_inv, spring_end_inv,
                pitch_inv, eta_global_inv, use_pitch_corr=inv_use_pitch_corr
            )

            def k_from_EI(EI_t): return (EI_t - EI_core) / (eta_b_x * EI_0)
            def k_from_GJ(GJ_t): return (GJ_t - GJ_core) / (eta_t_x * GJ_0)
            def k_from_EA(EA_t): return (EA_t - EA_core) / (eta_a_x * EA_0)

            k_EI = k_from_EI(EI_target) if EI_target is not None else None
            k_GJ = k_from_GJ(GJ_target) if GJ_target is not None else None
            k_EA = k_from_EA(EA_target) if EA_target is not None else None

            k_map = {'EI': k_EI, 'GJ': k_GJ, 'EA': k_EA}
            k_used = k_map[inv_primary]
            if k_used is None:
                st.error(f"主导刚度 {inv_primary} 的目标表达式未填写。")
                st.stop()

            valid_mask = np.isfinite(k_used) & (k_used > 1e-6) & (k_used < 1.0)

            with np.errstate(divide='ignore', invalid='ignore'):
                b_inv = (0.5184 * k_used * inv_w_s) / (k_used * inv_w_s + (1.0 - k_used) * inv_Z)
            b_inv = np.where(valid_mask, b_inv, np.nan)

            b_smooth = moving_average(b_inv, int(inv_smooth_window))

            with np.errstate(divide='ignore', invalid='ignore'):
                Y_s = 0.5184 - b_smooth
                k_smooth = 1.0 / (1.0 + (inv_w_s / inv_Z) * (Y_s / b_smooth))
            k_smooth = np.where(
                np.isfinite(k_smooth) & (b_smooth > 0) & (b_smooth < 0.5184),
                np.clip(k_smooth, 0.0, 1.0),
                np.nan
            )

            EI_actual = EI_core + eta_b_x * (k_smooth * EI_0)
            GJ_actual = GJ_core + eta_t_x * (k_smooth * GJ_0)
            EA_actual = EA_core + eta_a_x * (k_smooth * EA_0)

            # 一致性检查
            k_available = [k for k in [k_EI, k_GJ, k_EA] if k is not None]
            if len(k_available) >= 2:
                k_stack = np.vstack(k_available)
                with np.errstate(invalid='ignore'):
                    k_spread = np.nanmax(k_stack, axis=0) - np.nanmin(k_stack, axis=0)
                spread_max = np.nanmax(k_spread) if np.any(np.isfinite(k_spread)) else 0.0
                if spread_max > 0.15:
                    st.warning(f"⚠️ 不同目标推导出的 k(x) 最大差异为 {spread_max:.3f}，目标可能物理不一致。")
                else:
                    st.success(f"✅ 不同目标推导出的 k(x) 最大差异仅 {spread_max:.3f}，一致性良好。")

            def rel_err(a, t):
                if t is None: return None
                with np.errstate(divide='ignore', invalid='ignore'):
                    return np.abs(a - t) / np.maximum(np.abs(t), 1e-9) * 100.0

            err_EI = rel_err(EI_actual, EI_target)
            err_GJ = rel_err(GJ_actual, GJ_target)
            err_EA = rel_err(EA_actual, EA_target)

            def safe_metric(e):
                if e is None: return "—"
                v = np.nanmax(e) if np.any(np.isfinite(e)) else float('nan')
                return f"{v:.2f}%" if np.isfinite(v) else "—"

            st.subheader("误差统计")
            c1, c2, c3 = st.columns(3)
            c1.metric("EI 最大相对误差", safe_metric(err_EI))
            c2.metric("GJ 最大相对误差", safe_metric(err_GJ))
            c3.metric("EA 最大相对误差", safe_metric(err_EA))

            fig_inv, axes_inv = plt.subplots(7, 1, figsize=(11, 26))
            fig_inv.suptitle("Inverse Problem: Recover b(x)", fontsize=14)

            axes_inv[0].plot(x_inv, EI_core, 'g:', linewidth=1.5, label='Core EI alone')
            if EI_target is not None: axes_inv[0].plot(x_inv, EI_target, 'b-', linewidth=2, label='Target EI')
            axes_inv[0].plot(x_inv, EI_actual, 'r--', linewidth=2, label='Actual EI')
            axes_inv[0].set_ylabel('EI (N·mm²)'); axes_inv[0].grid(True); axes_inv[0].legend()
            axes_inv[0].set_title('Bending Stiffness EI')

            axes_inv[1].plot(x_inv, GJ_core, 'g:', linewidth=1.5, label='Core GJ alone')
            if GJ_target is not None: axes_inv[1].plot(x_inv, GJ_target, 'b-', linewidth=2, label='Target GJ')
            axes_inv[1].plot(x_inv, GJ_actual, 'r--', linewidth=2, label='Actual GJ')
            axes_inv[1].set_ylabel('GJ (N·mm²)'); axes_inv[1].grid(True); axes_inv[1].legend()
            axes_inv[1].set_title('Torsional Stiffness GJ')

            axes_inv[2].plot(x_inv, EA_core, 'g:', linewidth=1.5, label='Core EA alone')
            if EA_target is not None: axes_inv[2].plot(x_inv, EA_target, 'b-', linewidth=2, label='Target EA')
            axes_inv[2].plot(x_inv, EA_actual, 'r--', linewidth=2, label='Actual EA')
            axes_inv[2].set_ylabel('EA (N)'); axes_inv[2].grid(True); axes_inv[2].legend()
            axes_inv[2].set_title('Axial Stiffness EA')

            axes_inv[3].plot(x_inv, b_inv, color='lightgray', linewidth=1, label='Raw b')
            axes_inv[3].plot(x_inv, b_smooth, 'b-', linewidth=2, label='Smoothed b')
            axes_inv[3].axhline(0.5184, color='red', linestyle=':', alpha=0.5, label='b_max = 0.5184')
            axes_inv[3].axhline(0.0, color='red', linestyle=':', alpha=0.5)
            axes_inv[3].set_ylabel('b (mm)'); axes_inv[3].grid(True); axes_inv[3].legend()
            axes_inv[3].set_title('Connector Width b(x)')

            if k_EI is not None: axes_inv[4].plot(x_inv, k_EI, color='cyan', linewidth=1.5, label='k from EI')
            if k_GJ is not None: axes_inv[4].plot(x_inv, k_GJ, color='magenta', linewidth=1.5, label='k from GJ')
            if k_EA is not None: axes_inv[4].plot(x_inv, k_EA, color='orange', linewidth=1.5, label='k from EA')
            axes_inv[4].plot(x_inv, k_smooth, 'r-', linewidth=2, label='k used')
            axes_inv[4].axhline(0, color='black', linestyle=':', alpha=0.4)
            axes_inv[4].axhline(1, color='black', linestyle=':', alpha=0.4)
            axes_inv[4].set_ylabel('k'); axes_inv[4].grid(True); axes_inv[4].legend()
            axes_inv[4].set_title('Reduction Factor k(x)')

            axes_inv[5].plot(x_inv, d_core_inv, 'g-', linewidth=2, label='Core diameter d(x)')
            axes_inv[5].set_ylabel('d (mm)'); axes_inv[5].grid(True); axes_inv[5].legend()
            axes_inv[5].set_title('Core Diameter d(x)')

            axes_inv[6].plot(x_inv, eta_b_x, 'b-', linewidth=2, label='eta_b(x)')
            axes_inv[6].plot(x_inv, eta_t_x, 'r--', linewidth=2, label='eta_t(x)')
            axes_inv[6].plot(x_inv, eta_a_x, 'g-.', linewidth=2, label='eta_a(x)')
            axes_inv[6].set_xlabel('Distance from distal end (mm)')
            axes_inv[6].set_ylabel('eta'); axes_inv[6].grid(True); axes_inv[6].legend()
            axes_inv[6].set_title('Transfer Coefficients eta(x)')

            fig_inv.tight_layout(rect=[0, 0, 1, 0.98])
            st.pyplot(fig_inv)

            st.subheader("反解结果（每 10 mm 采样）")
            result_dict = {
                '位置 (mm)': np.round(x_inv, 2),
                '芯丝直径 d (mm)': np.round(d_core_inv, 4),
                '连接筋宽 b (mm)': np.round(b_smooth, 4),
                '折减系数 k': np.round(k_smooth, 4),
            }
            if EI_target is not None:
                result_dict['目标 EI'] = np.round(EI_target, 3); result_dict['实际 EI'] = np.round(EI_actual, 3)
            if GJ_target is not None:
                result_dict['目标 GJ'] = np.round(GJ_target, 3); result_dict['实际 GJ'] = np.round(GJ_actual, 3)
            if EA_target is not None:
                result_dict['目标 EA'] = np.round(EA_target, 3); result_dict['实际 EA'] = np.round(EA_actual, 3)

            inv_result_df = pd.DataFrame(result_dict)
            step = max(1, len(inv_result_df) // 35)
            st.dataframe(inv_result_df.iloc[::step], use_container_width=True)
            st.download_button(
                label="下载完整反解结果 CSV",
                data=inv_result_df.to_csv(index=False).encode('utf-8-sig'),
                file_name="inverse_b_result.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"反问题求解出错：{e}")
            st.exception(e)

# ==================== 反解 d ====================
else:
    if st.button("求解反问题（反解 d）", key="inverse_solve_d_btn", type="primary"):
        try:
            inv_E_core = st.session_state.get('E_core_input', 200000.0)
            inv_G_core = st.session_state.get('G_core_input', 77000.0)
            inv_E_hypo = st.session_state.get('E_hypo_input', 50000.0)
            inv_G_hypo = st.session_state.get('G_hypo_input', 19231.0)
            inv_D_o = st.session_state.get('D_o_input', 0.33)
            inv_D_i = st.session_state.get('D_i_input', 0.23)
            inv_L_total = float(st.session_state.get('L_total_input', 350.0))
            w_s = inv_w_s_d

            # 读取侧边栏海波管函数
            hypo_text = st.session_state.get('hypo_text_input', '')
            hypo_segments = parse_hypo_functions(hypo_text)
            if not hypo_segments:
                st.error("侧边栏海波管函数为空或格式错误。")
                st.stop()

            x_inv = np.linspace(0, inv_L_total, 500)

            # 解析 b(x)、Z(x)
            b_arr = np.zeros_like(x_inv)
            Z_arr = np.zeros_like(x_inv)
            for i, xi in enumerate(x_inv):
                bv, Zv = calc_b_Z_smooth(xi, hypo_segments, delta=2.0)
                b_arr[i] = bv; Z_arr[i] = Zv

            # 计算 k(x)
            Y = 0.5184 - b_arr
            Z_safe = np.where(Z_arr > 0, Z_arr, 1e-9)
            b_safe = np.where(b_arr > 0, b_arr, 1e-9)
            k_x = 1.0 / (1.0 + (w_s / Z_safe) * (Y / b_safe))
            k_x = np.where((Z_arr > 0) & (b_arr > 0) & (b_arr < 0.5184), k_x, np.nan)
            k_x = np.clip(k_x, 0.0, 1.0)

            # 完整海波管刚度
            I_0 = np.pi / 64.0 * (inv_D_o**4 - inv_D_i**4)
            J_0 = 2.0 * I_0
            A_0 = np.pi / 4.0 * (inv_D_o**2 - inv_D_i**2)
            EI_0 = inv_E_hypo * I_0
            GJ_0 = inv_G_hypo * J_0
            EA_0 = inv_E_hypo * A_0

            # 海波管等效刚度
            EI_hypo = k_x * EI_0
            GJ_hypo = k_x * GJ_0
            EA_hypo = k_x * EA_0

            # 读取侧边栏传递系数设置
            glue_intervals_inv = []
            glue_text_inv = st.session_state.get('glue_input', '')
            if glue_text_inv.strip():
                for line in glue_text_inv.strip().splitlines():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) == 3:
                        try: glue_intervals_inv.append((float(parts[0]), float(parts[1]), parts[2]))
                        except: pass

            spring_start_inv = st.session_state.get('spring_start_input', 0)
            spring_end_inv = st.session_state.get('spring_end_input', 150)
            pitch_inv = st.session_state.get('pitch_input', P_REF)
            eta_global_inv = {k: st.session_state.get(k + '_input', eta_defaults[k]) for k in eta_keys}

            eta_b_x, eta_t_x, eta_a_x = build_eta_arrays(
                x_inv, glue_intervals_inv, spring_start_inv, spring_end_inv,
                pitch_inv, eta_global_inv, use_pitch_corr=inv_use_pitch_corr
            )

            # 目标刚度
            EI_target = safe_eval_array(inv_EI_expr, x_inv) if inv_EI_expr.strip() else None
            GJ_target = safe_eval_array(inv_GJ_expr, x_inv) if inv_GJ_expr.strip() else None
            EA_target = safe_eval_array(inv_EA_expr, x_inv) if inv_EA_expr.strip() else None

            if EI_target is None and GJ_target is None and EA_target is None:
                st.error("至少需要提供一个目标刚度表达式。")
                st.stop()

            # 目标芯丝刚度
            def d_from_EI(EI_t):
                EI_ct = EI_t - eta_b_x * EI_hypo
                with np.errstate(invalid='ignore'):
                    val = np.where(EI_ct > 0, (64.0 * EI_ct / (inv_E_core * np.pi)) ** 0.25, np.nan)
                return val

            def d_from_GJ(GJ_t):
                GJ_ct = GJ_t - eta_t_x * GJ_hypo
                with np.errstate(invalid='ignore'):
                    val = np.where(GJ_ct > 0, (32.0 * GJ_ct / (inv_G_core * np.pi)) ** 0.25, np.nan)
                return val

            def d_from_EA(EA_t):
                EA_ct = EA_t - eta_a_x * EA_hypo
                with np.errstate(invalid='ignore'):
                    val = np.where(EA_ct > 0, (4.0 * EA_ct / (inv_E_core * np.pi)) ** 0.5, np.nan)
                return val

            d_EI = d_from_EI(EI_target) if EI_target is not None else None
            d_GJ = d_from_GJ(GJ_target) if GJ_target is not None else None
            d_EA = d_from_EA(EA_target) if EA_target is not None else None

            d_map = {'EI': d_EI, 'GJ': d_GJ, 'EA': d_EA}
            d_used = d_map[inv_primary]
            if d_used is None:
                st.error(f"主导刚度 {inv_primary} 的目标表达式未填写。")
                st.stop()

            valid_mask = np.isfinite(d_used) & (d_used > 1e-6) & (d_used < inv_d_max)

            n_invalid = int(np.sum(~valid_mask))
            if n_invalid > 0:
                st.warning(f"有 {n_invalid}/{len(x_inv)} 个位置反解出的 d 超出 (0, {inv_d_max})，将标记为无解（NaN）。")

            d_inv = np.where(valid_mask, d_used, np.nan)
            d_smooth = moving_average(d_inv, int(inv_smooth_window))
            # 平滑后重新检查越界
            d_smooth = np.where((d_smooth > 0) & (d_smooth < inv_d_max), d_smooth, np.nan)

            # 用平滑后的 d 重算三种刚度
            EI_core_actual = inv_E_core * np.pi * d_smooth**4 / 64.0
            GJ_core_actual = inv_G_core * np.pi * d_smooth**4 / 32.0
            EA_core_actual = inv_E_core * np.pi * d_smooth**2 / 4.0

            EI_actual = EI_core_actual + eta_b_x * EI_hypo
            GJ_actual = GJ_core_actual + eta_t_x * GJ_hypo
            EA_actual = EA_core_actual + eta_a_x * EA_hypo

            # 一致性检查
            d_available = [d for d in [d_EI, d_GJ, d_EA] if d is not None]
            if len(d_available) >= 2:
                d_stack = np.vstack(d_available)
                with np.errstate(invalid='ignore'):
                    d_spread = np.nanmax(d_stack, axis=0) - np.nanmin(d_stack, axis=0)
                spread_max = np.nanmax(d_spread) if np.any(np.isfinite(d_spread)) else 0.0
                if spread_max > 0.02:
                    st.warning(f"⚠️ 不同目标推导出的 d(x) 最大差异为 {spread_max:.4f} mm，目标可能物理不一致。")
                else:
                    st.success(f"✅ 不同目标推导出的 d(x) 最大差异仅 {spread_max:.4f} mm，一致性良好。")

            def rel_err(a, t):
                if t is None: return None
                with np.errstate(divide='ignore', invalid='ignore'):
                    return np.abs(a - t) / np.maximum(np.abs(t), 1e-9) * 100.0

            err_EI = rel_err(EI_actual, EI_target)
            err_GJ = rel_err(GJ_actual, GJ_target)
            err_EA = rel_err(EA_actual, EA_target)

            def safe_metric(e):
                if e is None: return "—"
                v = np.nanmax(e) if np.any(np.isfinite(e)) else float('nan')
                return f"{v:.2f}%" if np.isfinite(v) else "—"

            st.subheader("误差统计")
            c1, c2, c3 = st.columns(3)
            c1.metric("EI 最大相对误差", safe_metric(err_EI))
            c2.metric("GJ 最大相对误差", safe_metric(err_GJ))
            c3.metric("EA 最大相对误差", safe_metric(err_EA))

            fig_inv, axes_inv = plt.subplots(7, 1, figsize=(11, 26))
            fig_inv.suptitle("Inverse Problem: Recover Core Diameter d(x)", fontsize=14)

            axes_inv[0].plot(x_inv, EI_hypo, 'g:', linewidth=1.5, label='Hypo EI contribution')
            if EI_target is not None: axes_inv[0].plot(x_inv, EI_target, 'b-', linewidth=2, label='Target EI')
            axes_inv[0].plot(x_inv, EI_actual, 'r--', linewidth=2, label='Actual EI (smoothed d)')
            axes_inv[0].set_ylabel('EI (N·mm²)'); axes_inv[0].grid(True); axes_inv[0].legend()
            axes_inv[0].set_title('Bending Stiffness EI')

            axes_inv[1].plot(x_inv, GJ_hypo, 'g:', linewidth=1.5, label='Hypo GJ contribution')
            if GJ_target is not None: axes_inv[1].plot(x_inv, GJ_target, 'b-', linewidth=2, label='Target GJ')
            axes_inv[1].plot(x_inv, GJ_actual, 'r--', linewidth=2, label='Actual GJ')
            axes_inv[1].set_ylabel('GJ (N·mm²)'); axes_inv[1].grid(True); axes_inv[1].legend()
            axes_inv[1].set_title('Torsional Stiffness GJ')

            axes_inv[2].plot(x_inv, EA_hypo, 'g:', linewidth=1.5, label='Hypo EA contribution')
            if EA_target is not None: axes_inv[2].plot(x_inv, EA_target, 'b-', linewidth=2, label='Target EA')
            axes_inv[2].plot(x_inv, EA_actual, 'r--', linewidth=2, label='Actual EA')
            axes_inv[2].set_ylabel('EA (N)'); axes_inv[2].grid(True); axes_inv[2].legend()
            axes_inv[2].set_title('Axial Stiffness EA')

            axes_inv[3].plot(x_inv, b_arr, 'b-', linewidth=2, label='b(x) from hypo function')
            axes_inv[3].axhline(0.5184, color='red', linestyle=':', alpha=0.5)
            axes_inv[3].set_ylabel('b (mm)'); axes_inv[3].grid(True); axes_inv[3].legend()
            axes_inv[3].set_title('Hypotube Connector Width b(x) (input)')

            axes_inv[4].plot(x_inv, k_x, 'r-', linewidth=2, label='k(x) from b, Z')
            axes_inv[4].axhline(0, color='black', linestyle=':', alpha=0.4)
            axes_inv[4].axhline(1, color='black', linestyle=':', alpha=0.4)
            axes_inv[4].set_ylabel('k'); axes_inv[4].grid(True); axes_inv[4].legend()
            axes_inv[4].set_title('Reduction Factor k(x) (from hypo)')

            axes_inv[5].plot(x_inv, d_inv, color='lightgray', linewidth=1, label='Raw d')
            axes_inv[5].plot(x_inv, d_smooth, 'g-', linewidth=2, label='Smoothed d')
            axes_inv[5].axhline(inv_d_max, color='red', linestyle=':', alpha=0.5, label=f'd_max = {inv_d_max}')
            axes_inv[5].axhline(0.0, color='red', linestyle=':', alpha=0.5)
            axes_inv[5].set_ylabel('d (mm)'); axes_inv[5].grid(True); axes_inv[5].legend()
            axes_inv[5].set_title('Core Diameter d(x) (recovered)')

            axes_inv[6].plot(x_inv, eta_b_x, 'b-', linewidth=2, label='eta_b(x)')
            axes_inv[6].plot(x_inv, eta_t_x, 'r--', linewidth=2, label='eta_t(x)')
            axes_inv[6].plot(x_inv, eta_a_x, 'g-.', linewidth=2, label='eta_a(x)')
            axes_inv[6].set_xlabel('Distance from distal end (mm)')
            axes_inv[6].set_ylabel('eta'); axes_inv[6].grid(True); axes_inv[6].legend()
            axes_inv[6].set_title('Transfer Coefficients eta(x)')

            fig_inv.tight_layout(rect=[0, 0, 1, 0.98])
            st.pyplot(fig_inv)

            st.subheader("反解结果（每 10 mm 采样）")
            result_dict = {
                '位置 (mm)': np.round(x_inv, 2),
                '海波管 b (mm)': np.round(b_arr, 4),
                '折减系数 k': np.round(k_x, 4),
                '芯丝直径 d (mm)': np.round(d_smooth, 4),
                'eta_b(x)': np.round(eta_b_x, 3),
                'eta_t(x)': np.round(eta_t_x, 3),
                'eta_a(x)': np.round(eta_a_x, 3),
            }
            if EI_target is not None:
                result_dict['目标 EI'] = np.round(EI_target, 3); result_dict['实际 EI'] = np.round(EI_actual, 3)
            if GJ_target is not None:
                result_dict['目标 GJ'] = np.round(GJ_target, 3); result_dict['实际 GJ'] = np.round(GJ_actual, 3)
            if EA_target is not None:
                result_dict['目标 EA'] = np.round(EA_target, 3); result_dict['实际 EA'] = np.round(EA_actual, 3)

            inv_result_df = pd.DataFrame(result_dict)
            step = max(1, len(inv_result_df) // 35)
            st.dataframe(inv_result_df.iloc[::step], use_container_width=True)
            st.download_button(
                label="下载完整反解结果 CSV",
                data=inv_result_df.to_csv(index=False).encode('utf-8-sig'),
                file_name="inverse_d_result.csv",
                mime="text/csv"
            )

            # 可行性检查
            st.subheader("可行性检查报告")
            check_items = []
            d_finite = np.isfinite(d_smooth)
            if not np.any(d_finite):
                check_items.append("❌ 全部位置无解（d 全为 NaN），可能目标刚度低于海波管贡献")
            else:
                if np.any(d_finite & (d_smooth <= 0)):
                    check_items.append("❌ 存在 d ≤ 0 的位置")
                else:
                    check_items.append("✅ 所有有效位置 d > 0")
                if np.any(d_finite & (d_smooth >= inv_d_max)):
                    check_items.append(f"❌ 存在 d ≥ {inv_d_max} 的位置，超过芯丝直径上限")
                else:
                    check_items.append(f"✅ 所有有效位置 d < {inv_d_max}")
            n_nan = int(np.sum(~d_finite))
            if n_nan > 0:
                check_items.append(f"⚠️ {n_nan}/{len(d_smooth)} 个位置无解（已标记 NaN）")

            for tag, e in [("EI", err_EI), ("GJ", err_GJ), ("EA", err_EA)]:
                if e is None: continue
                e_finite = e[np.isfinite(e)]
                if len(e_finite) == 0:
                    check_items.append(f"⚠️ {tag} 误差无法计算")
                else:
                    emax = e_finite.max()
                    if emax > 20:
                        check_items.append(f"⚠️ {tag} 最大相对误差 {emax:.2f}% 较大")
                    else:
                        check_items.append(f"✅ {tag} 最大相对误差 {emax:.2f}% 在可接受范围")

            for item in check_items:
                st.write(item)

        except Exception as e:
            st.error(f"反问题求解出错：{e}")
            st.exception(e)
