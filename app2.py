import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
import json
from datetime import datetime

st.set_page_config(page_title="导丝刚度分析工具", layout="wide")

# ==================== pending 机制 ====================
if '_pending_import' in st.session_state:
    pending = st.session_state.pop('_pending_import')
    for k in pending.get('clear_keys', []):
        st.session_state.pop(k, None)
    for k, v in pending.get('set_keys', {}).items():
        st.session_state[k] = v
    if 'core_df' in pending:
        st.session_state.core_editor_data = pending['core_df']
        st.session_state.core_editor_version = st.session_state.get('core_editor_version', 0) + 1
    if 'saved_versions' in pending:
        st.session_state.saved_versions = pending['saved_versions']

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

def moving_average(arr, window):
    window = int(window)
    if window < 1:
        return np.asarray(arr, dtype=float).copy()
    if window % 2 == 0:
        window += 1
    s = pd.Series(np.asarray(arr, dtype=float))
    return s.rolling(window=window, center=True, min_periods=1).mean().to_numpy()

# ==================== 海波管分段函数解析 ====================
def parse_hypo_functions(text, default_N=2):
    segments = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) == 4:
            start_s, end_s, b_expr, Z_expr = parts
            N = default_N
        elif len(parts) == 5:
            start_s, end_s, b_expr, Z_expr, N_s = parts
            try:
                N = int(round(float(N_s)))
                if N < 1:
                    N = default_N
            except:
                N = default_N
        else:
            continue
        try:
            start = float(start_s); end = float(end_s)
            segments.append((start, end, b_expr, Z_expr, N))
        except:
            continue
    return segments

def format_hypo_segments(segments):
    lines = []
    for seg in segments:
        if len(seg) >= 5:
            lines.append(f"{seg[0]},{seg[1]},{seg[2]},{seg[3]},{seg[4]}")
        else:
            lines.append(f"{seg[0]},{seg[1]},{seg[2]},{seg[3]}")
    return "\n".join(lines)

def calc_b_Z_original(x, segments, default_N=2):
    for start, end, b_expr, Z_expr, N in segments:
        if start <= x < end:
            return safe_eval(b_expr, x), safe_eval(Z_expr, x), N
    if x < segments[0][0]:
        seg = segments[0]
        return safe_eval(seg[2], seg[0]), safe_eval(seg[3], seg[0]), seg[4]
    else:
        seg = segments[-1]
        return safe_eval(seg[2], seg[1]), safe_eval(seg[3], seg[1]), seg[4]

def calc_b_Z_smooth(x, segments, delta=2.0, default_N=2):
    for i, (start, end, b_expr, Z_expr, N_cur) in enumerate(segments):
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
                        return b, Z, N_cur
            return safe_eval(b_expr, x), safe_eval(Z_expr, x), N_cur
    if x < segments[0][0]:
        seg = segments[0]
        return safe_eval(seg[2], seg[0]), safe_eval(seg[3], seg[0]), seg[4]
    else:
        seg = segments[-1]
        return safe_eval(seg[2], seg[1]), safe_eval(seg[3], seg[1]), seg[4]

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
                    eta_b = eta_global['full_b']; eta_t = eta_global['full_t']; eta_a = eta_global['full_a']
                elif g_type == 'core_spring':
                    eta_b = eta_global['core_spring_b']; eta_t = eta_global['core_spring_t']; eta_a = eta_global['core_spring_a']
                elif g_type == 'core_hypo':
                    eta_b = eta_global['core_hypo_b']; eta_t = eta_global['core_hypo_t']; eta_a = eta_global['core_hypo_a']
                break

        if use_pitch_corr and in_spring and not in_glue and pitch > 0:
            eta_t = min(eta_t * P_REF / pitch, 0.95)

        eta_b_x[i] = eta_b; eta_t_x[i] = eta_t; eta_a_x[i] = eta_a
    return eta_b_x, eta_t_x, eta_a_x

# ==================== 正问题 ====================
def compute_version(x, core_df, hypo_segments, params, eta_global,
                    smooth=False, spring_enabled=False, complex_params=None):
    E_core = params['E_core']; G_core = params['G_core']
    E_hypo = params['E_hypo']; G_hypo = params['G_hypo']
    D_o = params['D_o']; D_i = params['D_i']; w_s = params['w_s']
    L_total = params['L_total']; F = params['F']; T0 = params['T0']
    spring_start = params['spring_start']; spring_end = params['spring_end']
    glue_intervals = params['glue_intervals']
    n_beam_default = params.get('n_beam', 2)

    I0 = np.pi/64*(D_o**4 - D_i**4); J0 = 2*I0
    A0 = np.pi/4*(D_o**2 - D_i**2)
    EI0 = E_hypo*I0; GJ0 = G_hypo*J0; EA0 = E_hypo*A0
    C_full = np.pi * D_o

    n = len(x)
    d_core_arr = np.zeros(n); b_arr = np.zeros(n); Z_arr = np.zeros(n)
    N_arr = np.zeros(n, dtype=int)
    eta_b_arr = np.zeros(n); eta_t_arr = np.zeros(n); eta_a_arr = np.zeros(n)

    for i, xi in enumerate(x):
        d_core_arr[i] = interp_core(xi, core_df, smooth=smooth)
        if smooth:
            b_val, Z_val, N_val = calc_b_Z_smooth(xi, hypo_segments, delta=2.0, default_N=n_beam_default)
        else:
            b_val, Z_val, N_val = calc_b_Z_original(xi, hypo_segments, default_N=n_beam_default)
        b_arr[i] = b_val; Z_arr[i] = Z_val; N_arr[i] = N_val

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

    b_total = N_arr * b_arr
    Y = C_full - b_total
    Z_safe = np.where(Z_arr > 0, Z_arr, 1e-9)
    b_safe = np.where(b_total > 0, b_total, 1e-9)
    k = 1.0 / (1.0 + (w_s / Z_safe) * (Y / b_safe))
    k = np.where((Z_arr > 0) & (b_arr > 0) & (b_total < C_full), k, np.nan)
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
    sigma_bend = M_hypo/(2*np.maximum(b_total,1e-9)*t_wall*r_m)
    tau_tors = T_hypo/(2*np.maximum(b_total,1e-9)*t_wall*r_m)
    sigma_eq = np.sqrt(sigma_bend**2 + 3*tau_tors**2)

    theta = cumulative_trapezoid(M_total/EI_total, x, initial=0)
    theta = theta - theta[-1]
    y_def = cumulative_trapezoid(theta, x, initial=0)
    y_def = y_def - y_def[-1]
    phi = cumulative_trapezoid(T_total/GJ_total, x, initial=0)

    return (EI_total, GJ_total, EA_total, sigma_bend, tau_tors, sigma_eq, y_def, phi,
            T_hypo, T_core, ratio_t, eta_t_arr)

def three_point_deflection_at(z, EI_local, P, L, a):
    M = np.where(z <= L/2, P*z/2, P*(L-z)/2)
    if a <= L/2:
        m = np.where(z <= a, (L-a)/L*z,
             np.where(z <= L/2, (L-a)/L*z - (z-a),
                      a/L*(L-z)))
    else:
        a_sym = L - a
        m = np.where(z <= a_sym, (L-a_sym)/L*z,
             np.where(z <= L/2, (L-a_sym)/L*z - (z-a_sym),
                      a_sym/L*(L-z)))
    return np.trapezoid(M * m / EI_local, z)

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

default_hypo_v1 = """0,10,0.036,0.058,3
10,20,0.049,0.0663,3
20,30,0.058,0.0747,2
30,40,0.071,0.083,2
40,50,0.086,0.0913,2
50,60,0.096,0.0997,2
60,70,0.102,0.108,2
70,80,0.112,0.1163,1
80,90,0.126,0.1246,1
90,350,0.152,0.133,1"""

default_hypo_v2 = """0,10,0.036,0.058,3
10,90,-0.000000154786*x**3+0.000017054434*x**2+0.0011531092*x+0.0229182506,-0.000008789096*x**2+0.0018164096*x+0.0407148136,2
90,350,0.152,0.133,1"""

default_hypo_v3 = """0,10,0.036,0.058,3
10,70,-0.000000154786*x**3+0.00001307278*x**2+0.0011531092*x+0.023316416,-37/3600000*x**2+0.0014388889*x+0.0446388888,2
70,145,0.115,0.095,2
145,180,-0.000030204*(x-145)**2+(0.037-1225*(-0.000030204))/35*(x-145)+0.115,0.095+0.038*(x-145)/35,1
180,350,0.152,0.133,1"""

# ==================== 初始化 ====================
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
if 'n_beam_mode' not in st.session_state: st.session_state.n_beam_mode = "双梁（全周长 2 个连接筋）"
for k in eta_keys:
    if k + '_input' not in st.session_state: st.session_state[k + '_input'] = eta_defaults[k]

# ==================== JSON 导入导出 ====================
MODEL_FILE_VERSION = 1
APP_ID = "guidewire_stiffness_analysis"
N_BEAM_CHOICES = ["单梁（全周长 1 个连接筋）",
                  "双梁（全周长 2 个连接筋）",
                  "三梁（全周长 3 个连接筋）"]

def _to_native(v):
    if v is None: return None
    try:
        if v is pd.NA or v is pd.NaT: return None
    except Exception:
        pass
    if isinstance(v, np.bool_): return bool(v)
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.floating):
        fv = float(v)
        return None if pd.isna(fv) else fv
    if isinstance(v, float):
        return None if pd.isna(v) else v
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_to_native(x) for x in v]
    return v

def safe_float(v, default=None):
    try:
        if v is None: return default
        if isinstance(v, str) and v.strip() == '': return default
        fv = float(v)
        if pd.isna(fv): return default
        return fv
    except (ValueError, TypeError):
        return default

def core_df_to_json(df):
    if df is None or df.empty: return []
    records = []
    for _, row in df.iterrows():
        rec = {str(c): _to_native(row[c]) for c in df.columns}
        records.append(rec)
    return records

def json_to_core_df(records, warn_list, prefix=""):
    if not records:
        return default_core_v1.copy()
    try:
        df = pd.DataFrame(records)
        expected = ["start", "end", "d_start", "d_end"]
        for c in expected:
            if c not in df.columns:
                warn_list.append(f"{prefix}芯丝表缺少列 {c}，已补 0")
                df[c] = 0.0
        df = df[expected].copy()
        for c in expected:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
        return df
    except Exception as e:
        warn_list.append(f"{prefix}芯丝表解析失败：{e}，已用默认")
        return default_core_v1.copy()

def version_to_json(ver):
    return {
        'name': str(ver.get('name', '')),
        'E_core': _to_native(ver.get('E_core')),
        'G_core': _to_native(ver.get('G_core')),
        'E_hypo': _to_native(ver.get('E_hypo')),
        'G_hypo': _to_native(ver.get('G_hypo')),
        'D_o': _to_native(ver.get('D_o')),
        'D_i': _to_native(ver.get('D_i')),
        'w_s': _to_native(ver.get('w_s')),
        'L_total': _to_native(ver.get('L_total')),
        'F': _to_native(ver.get('F')),
        'T0': _to_native(ver.get('T0')),
        'spring_start': _to_native(ver.get('spring_start')),
        'spring_end': _to_native(ver.get('spring_end')),
        'glue_intervals': [[_to_native(g[0]), _to_native(g[1]), str(g[2])]
                          for g in ver.get('glue_intervals', [])],
        'core_df': core_df_to_json(ver.get('core_df')),
        'hypo_segments': [[_to_native(s[0]), _to_native(s[1]), str(s[2]), str(s[3]), int(s[4])]
                          for s in ver.get('hypo_segments', [])],
        'eta': {k: _to_native(v) for k, v in ver.get('eta', {}).items()},
        'n_beam': int(ver.get('n_beam', 2)),
        'complex_params': {k: _to_native(v)
                          for k, v in ver.get('complex_params', {}).items()},
    }

def json_to_version(obj, warn_list, prefix=""):
    if not isinstance(obj, dict):
        warn_list.append(f"{prefix}版本不是对象，已跳过")
        return None
    try:
        name = obj.get('name', '')
        if not isinstance(name, str):
            name = str(name)
        if not name:
            name = "Unnamed"

        glue_in = obj.get('glue_intervals', [])
        glue = []
        if isinstance(glue_in, list):
            for item in glue_in:
                try:
                    if isinstance(item, list) and len(item) >= 3:
                        glue.append((float(item[0]), float(item[1]), str(item[2])))
                    elif isinstance(item, dict):
                        glue.append((float(item.get('start', 0)),
                                     float(item.get('end', 1)),
                                     str(item.get('type', 'full'))))
                except Exception:
                    pass

        hypo_in = obj.get('hypo_segments', [])
        if not isinstance(hypo_in, list):
            warn_list.append(f"{prefix}hypo_segments 不是列表，已重置")
            hypo_in = []
        hypo_segments = []
        for item in hypo_in:
            try:
                if isinstance(item, list) and len(item) >= 4:
                    s = float(item[0]); e = float(item[1])
                    b_expr = str(item[2]); Z_expr = str(item[3])
                    N = int(item[4]) if len(item) >= 5 else 2
                    hypo_segments.append((s, e, b_expr, Z_expr, N))
            except Exception:
                pass
        if not hypo_segments:
            warn_list.append(f"{prefix}海波管函数为空，已用默认")
            hypo_segments = parse_hypo_functions(default_hypo_v1, default_N=2)

        eta_in = obj.get('eta', {})
        if not isinstance(eta_in, dict): eta_in = {}
        eta = {k: safe_float(eta_in.get(k), eta_defaults[k]) for k in eta_keys}

        core_df = json_to_core_df(obj.get('core_df', []), warn_list, prefix=prefix)

        cp_in = obj.get('complex_params', {})
        if not isinstance(cp_in, dict): cp_in = {}

        return {
            'name': name,
            'E_core': safe_float(obj.get('E_core'), 200000.0),
            'G_core': safe_float(obj.get('G_core'), 77000.0),
            'E_hypo': safe_float(obj.get('E_hypo'), 50000.0),
            'G_hypo': safe_float(obj.get('G_hypo'), 19231.0),
            'D_o': safe_float(obj.get('D_o'), 0.33),
            'D_i': safe_float(obj.get('D_i'), 0.23),
            'w_s': safe_float(obj.get('w_s'), 0.03),
            'L_total': safe_float(obj.get('L_total'), 350.0),
            'F': safe_float(obj.get('F'), 0.001),
            'T0': safe_float(obj.get('T0'), 1.0),
            'spring_start': safe_float(obj.get('spring_start'), 0),
            'spring_end': safe_float(obj.get('spring_end'), 150),
            'glue_intervals': glue,
            'core_df': core_df,
            'hypo_segments': hypo_segments,
            'eta': eta,
            'n_beam': int(obj.get('n_beam', 2)),
            'complex_params': {
                'eta_glue_peak': safe_float(cp_in.get('eta_glue_peak'), 0.90),
                'pitch': safe_float(cp_in.get('pitch'), P_REF),
            },
        }
    except Exception as e:
        warn_list.append(f"{prefix}版本解析失败：{e}")
        return None

def current_state_to_json():
    core_data = st.session_state.get('core_editor_data', default_core_v1)
    return {
        'name_input': str(st.session_state.get('name_input', '')),
        'E_core': _to_native(st.session_state.get('E_core_input', 200000.0)),
        'G_core': _to_native(st.session_state.get('G_core_input', 77000.0)),
        'E_hypo': _to_native(st.session_state.get('E_hypo_input', 50000.0)),
        'G_hypo': _to_native(st.session_state.get('G_hypo_input', 19231.0)),
        'D_o': _to_native(st.session_state.get('D_o_input', 0.33)),
        'D_i': _to_native(st.session_state.get('D_i_input', 0.23)),
        'w_s': _to_native(st.session_state.get('w_s_input', 0.03)),
        'L_total': _to_native(st.session_state.get('L_total_input', 350.0)),
        'F': _to_native(st.session_state.get('F_input', 0.001)),
        'T0': _to_native(st.session_state.get('T0_input', 1.0)),
        'spring_start': _to_native(st.session_state.get('spring_start_input', 0)),
        'spring_end': _to_native(st.session_state.get('spring_end_input', 150)),
        'glue_input': str(st.session_state.get('glue_input', '')),
        'hypo_text_input': str(st.session_state.get('hypo_text_input', '')),
        'pitch_input': _to_native(st.session_state.get('pitch_input', P_REF)),
        'n_beam_mode': str(st.session_state.get('n_beam_mode', N_BEAM_CHOICES[1])),
        'core_df': core_df_to_json(core_data),
        'eta_inputs': {k: _to_native(st.session_state.get(k + '_input', eta_defaults[k]))
                       for k in eta_keys},
    }

def build_export_payload():
    return {
        'app': APP_ID,
        'model_file_version': MODEL_FILE_VERSION,
        'exported_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'current': current_state_to_json(),
        'saved_versions': [version_to_json(v) for v in st.session_state.saved_versions],
    }

def parse_model_payload(payload):
    warn_list = []
    if not isinstance(payload, dict):
        raise ValueError("文件内容不是有效的 JSON 对象")
    if payload.get('app') not in (None, APP_ID):
        warn_list.append(f"该文件的来源标识不是本工具（app={payload.get('app')!r}），仍尝试导入。")
    mfv = payload.get('model_file_version', 0)
    try:
        mfv = int(mfv)
    except Exception:
        mfv = 0
    if mfv and mfv > MODEL_FILE_VERSION:
        warn_list.append(f"文件版本 {mfv} 高于当前支持的 {MODEL_FILE_VERSION}，可能不兼容。")

    cur = payload.get('current', {})
    current_state = {}
    if isinstance(cur, dict):
        current_state['name_input'] = str(cur.get('name_input', 'Imported'))
        num_map = [
            ('E_core', 200000.0, 'E_core_input'),
            ('G_core', 77000.0, 'G_core_input'),
            ('E_hypo', 50000.0, 'E_hypo_input'),
            ('G_hypo', 19231.0, 'G_hypo_input'),
            ('D_o', 0.33, 'D_o_input'),
            ('D_i', 0.23, 'D_i_input'),
            ('w_s', 0.03, 'w_s_input'),
            ('L_total', 350.0, 'L_total_input'),
            ('F', 0.001, 'F_input'),
            ('T0', 1.0, 'T0_input'),
            ('spring_start', 0, 'spring_start_input'),
            ('spring_end', 150, 'spring_end_input'),
            ('pitch_input', P_REF, 'pitch_input'),
        ]
        for src_key, default_v, sess_key in num_map:
            current_state[sess_key] = safe_float(cur.get(src_key), default_v)
        current_state['glue_input'] = str(cur.get('glue_input', ''))
        current_state['hypo_text_input'] = str(cur.get('hypo_text_input', ''))
        nbm = cur.get('n_beam_mode', N_BEAM_CHOICES[1])
        if nbm not in N_BEAM_CHOICES:
            nbm = N_BEAM_CHOICES[1]
        current_state['n_beam_mode'] = nbm
        current_state['core_df'] = json_to_core_df(cur.get('core_df', []), warn_list, prefix="[当前] ")
        eta_in = cur.get('eta_inputs', {})
        if not isinstance(eta_in, dict): eta_in = {}
        for k in eta_keys:
            current_state[k + '_input'] = safe_float(eta_in.get(k), eta_defaults[k])
    else:
        warn_list.append("缺少 current 字段，当前工作区使用默认值。")

    versions_in = payload.get('saved_versions', [])
    if not isinstance(versions_in, list):
        warn_list.append("saved_versions 不是列表，已忽略。")
        versions_in = []
    saved_versions = []
    for i, vobj in enumerate(versions_in):
        v = json_to_version(vobj, warn_list, prefix=f"版本 {i+1}: ")
        if v is not None:
            saved_versions.append(v)

    return current_state, saved_versions, warn_list

def parse_json_text(raw_text):
    if raw_text is None:
        raise ValueError("输入为空")
    text = raw_text
    if isinstance(text, (bytes, bytearray)):
        text = text.decode('utf-8-sig')
    else:
        text = str(text)
    text = text.strip().lstrip('\ufeff')
    if not text:
        raise ValueError("输入内容为空")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 格式错误：第 {e.lineno} 行第 {e.colno} 列 — {e.msg}")
    return parse_model_payload(payload)

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("加载预设版本")
    col1, col2, col3 = st.columns(3)

    def load_preset(name, core_df, hypo_text, s_start, s_end, glue, pitch,
                    n_beam_mode=N_BEAM_CHOICES[1]):
        st.session_state.name_input = name
        st.session_state.core_editor_data = core_df.copy()
        st.session_state.core_editor_version += 1
        st.session_state.hypo_text_input = hypo_text
        st.session_state.spring_start_input = s_start
        st.session_state.spring_end_input = s_end
        st.session_state.glue_input = glue
        st.session_state.pitch_input = pitch
        st.session_state.n_beam_mode = n_beam_mode
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

    st.subheader("海波管梁模式（默认值）")
    n_beam_mode = st.radio(
        "海波管函数未指定 N 时使用的默认 N",
        N_BEAM_CHOICES,
        key="n_beam_mode"
    )
    n_beam = {N_BEAM_CHOICES[0]: 1, N_BEAM_CHOICES[1]: 2, N_BEAM_CHOICES[2]: 3}[n_beam_mode]
    st.caption(f"全局默认 N = {n_beam}（每个区间可用第 5 列覆盖）")

    st.subheader("载荷参数")
    F = st.number_input("远端横向力 F (N)", value=0.001, step=0.001, format="%.4f", key="F_input")
    T0 = st.number_input("近端扭矩 T0 (N·mm)", value=1.0, step=0.1, key="T0_input")

    st.subheader("弹簧圈范围")
    spring_start = st.number_input("弹簧圈起始位置 (mm)", step=5, key="spring_start_input")
    spring_end = st.number_input("弹簧圈结束位置 (mm)", step=5, key="spring_end_input")

    st.subheader("弹簧圈螺距")
    st.number_input("螺距 (mm)", min_value=0.010, max_value=0.200, step=0.001,
                    format="%.4f", key="pitch_input")

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
    st.caption("格式: start,end,b_expr,Z_expr[,N]  |  N=1单梁, 2双梁, 3三梁；不写时用默认 N")
    uploaded_file = st.file_uploader("上传 Excel/CSV (可选)", type=["xlsx","xls","csv"], key="file_uploader")
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            if all(col in df_upload.columns for col in ['start','end','b_expr','Z_expr']):
                has_N = 'N' in df_upload.columns
                lines = []
                for _, r in df_upload.iterrows():
                    if has_N:
                        lines.append(f"{r['start']},{r['end']},{r['b_expr']},{r['Z_expr']},{int(r['N'])}")
                    else:
                        lines.append(f"{r['start']},{r['end']},{r['b_expr']},{r['Z_expr']}")
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
        hypo_segments = parse_hypo_functions(st.session_state.hypo_text_input, default_N=n_beam)
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
            'n_beam': n_beam,
            'complex_params': {
                'eta_glue_peak': 0.90,
                'pitch': st.session_state.pitch_input,
            }
        }
        st.session_state.saved_versions.append(version)
        st.success(f"版本 '{version['name']}' 已保存")

    # ==================== 模型导入 / 导出 ====================
    st.divider()
    st.header("📦 模型导入 / 导出")
    st.caption("导出当前工作区 + 所有已保存版本，导入后完整恢复。")

    export_payload = build_export_payload()
    export_bytes = json.dumps(export_payload, ensure_ascii=False,
                              indent=2, default=str).encode('utf-8')
    st.download_button(
        "⬇️ 导出模型 (JSON)",
        data=export_bytes,
        file_name=f"guidewire_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        key="dl_model_json"
    )

    with st.expander("⬆️ 导入模型", expanded=False):
        import_mode = st.radio("方式", ["📁 上传文件", "📝 粘贴文本"],
                               key="import_mode_radio", horizontal=True)
        raw_text = None
        source_label = ""
        if import_mode == "📁 上传文件":
            up = st.file_uploader("选择 JSON", type=['json'], key="model_uploader")
            if up is not None:
                try:
                    raw_text = up.getvalue().decode('utf-8-sig')
                    source_label = up.name
                except Exception as e:
                    st.error(f"读取失败：{e}")
        else:
            paste = st.text_area("粘贴 JSON", value="", height=200, key="model_paste_area")
            if paste and paste.strip():
                raw_text = paste; source_label = "粘贴的文本"

        if raw_text is not None:
            try:
                new_cur, new_versions, imp_warnings = parse_json_text(raw_text)
            except Exception as e:
                st.error(f"解析失败：{e}")
            else:
                st.info(f"**来源**：{source_label} | "
                        f"**版本数**：{len(new_versions)} 个 | "
                        f"**当前工作区核心列**：{len(new_cur.get('core_df', []))} 行")
                for w in imp_warnings:
                    st.warning(w)

                version_action = "ignore"
                if len(new_versions) > 0:
                    mode = st.radio(
                        f"文件中含 {len(new_versions)} 个版本，本地有 {len(st.session_state.saved_versions)} 个",
                        ["① 覆盖", "② 合并", "③ 忽略"], key="version_import_mode")
                    if mode.startswith("①"): version_action = "replace"
                    elif mode.startswith("②"): version_action = "merge"

                if st.button("✅ 确认导入", key="confirm_import_model", type="primary"):
                    # 清空所有 widget 键（强制刷新）
                    clear_keys = [
                        'name_input',
                        'E_core_input','G_core_input','E_hypo_input','G_hypo_input',
                        'D_o_input','D_i_input','w_s_input','L_total_input',
                        'F_input','T0_input',
                        'spring_start_input','spring_end_input',
                        'pitch_input',
                        'glue_input','hypo_text_input','n_beam_mode',
                    ] + [k + '_input' for k in eta_keys]

                    # 确定最终 saved_versions
                    if version_action == "replace":
                        final_versions = new_versions
                    elif version_action == "merge":
                        merged = list(st.session_state.saved_versions)
                        for nv in new_versions:
                            found = False
                            for idx, ex in enumerate(merged):
                                if ex.get('name') == nv.get('name'):
                                    merged[idx] = nv
                                    found = True
                                    break
                            if not found:
                                merged.append(nv)
                        final_versions = merged
                    else:
                        final_versions = st.session_state.saved_versions

                    # 构造 pending
                    set_keys = {}
                    for k, v in new_cur.items():
                        if k == 'core_df':
                            continue
                        set_keys[k] = v

                    st.session_state['_pending_import'] = {
                        'clear_keys': clear_keys,
                        'set_keys': set_keys,
                        'core_df': new_cur.get('core_df', default_core_v1.copy()),
                        'saved_versions': final_versions,
                    }
                    st.rerun()

# ==================== 主区域 ====================
st.header("Guidewire Multi-layer Stiffness Analysis")

st.header("已保存版本")
if not st.session_state.saved_versions:
    st.info("请在左侧编辑参数并点击“保存当前版本”。")
else:
    for idx, ver in enumerate(st.session_state.saved_versions):
        col1, col2, col3 = st.columns([3,1,1])
        new_name = col1.text_input("版本名称", value=ver['name'],
                                    key=f"rename_{idx}", label_visibility="collapsed")
        if new_name != ver['name']: ver['name'] = new_name
        N_list = sorted(set([seg[4] for seg in ver['hypo_segments']]))
        col1.caption(f"N = {N_list}")

        if col2.button("加载到左侧", key=f"load_{idx}"):
            st.session_state.name_input = ver['name']
            st.session_state.core_editor_data = ver['core_df'].copy()
            st.session_state.core_editor_version += 1
            st.session_state.hypo_text_input = format_hypo_segments(ver['hypo_segments'])
            st.session_state.spring_start_input = ver['spring_start']
            st.session_state.spring_end_input = ver['spring_end']
            st.session_state.glue_input = "\n".join([f"{s},{e},{t}" for s,e,t in ver['glue_intervals']])
            for k in eta_keys:
                st.session_state[k + '_input'] = ver['eta'][k]
            cp = ver.get('complex_params', {})
            st.session_state.pitch_input = cp.get('pitch', P_REF)
            nb = ver.get('n_beam', 2)
            st.session_state.n_beam_mode = {1: N_BEAM_CHOICES[0], 2: N_BEAM_CHOICES[1], 3: N_BEAM_CHOICES[2]}[nb]
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
            axes1[0].set_ylabel('EI (N·mm²)'); axes1[1].set_ylabel('GJ (N·mm²)')
            axes1[2].set_xlabel('Distance (mm)'); axes1[2].set_ylabel('EA (N)')
            for ax in axes1: ax.grid(True)
            fig2, axes2 = plt.subplots(3,1,figsize=(10,12)); fig2.suptitle("Stress Comparison")
            axes2[0].set_ylabel('sigma_b (MPa)'); axes2[1].set_ylabel('tau (MPa)')
            axes2[2].set_xlabel('Distance (mm)'); axes2[2].set_ylabel('Von Mises (MPa)')
            for ax in axes2: ax.grid(True)

            for idx in selected_indices:
                ver = st.session_state.saved_versions[idx]
                color = COLORS[idx % len(COLORS)]
                label = f"{ver['name']}"
                params = {k: ver[k] for k in ['E_core','G_core','E_hypo','G_hypo','D_o','D_i','w_s','L_total','F','T0','spring_start','spring_end','glue_intervals']}
                params['n_beam'] = ver.get('n_beam', 2)
                res = compute_version(x, ver['core_df'], ver['hypo_segments'], params, ver['eta'])
                axes1[0].plot(x, res[0], color=color, linewidth=2, label=label)
                axes1[1].plot(x, res[1], color=color, linewidth=2, label=label)
                axes1[2].plot(x, res[2], color=color, linewidth=2, label=label)
                axes2[0].plot(x, res[3], color=color, linewidth=2, label=label)
                axes2[1].plot(x, res[4], color=color, linewidth=2, label=label)
                axes2[2].plot(x, res[5], color=color, linewidth=2, label=label)
            for ax in list(axes1) + list(axes2): ax.legend()
            st.pyplot(fig1); st.pyplot(fig2)

# ==================== 反问题 ====================
st.divider()
st.header("🔧 反问题求解")

inv_mode = st.radio("反解对象",
    ["反解连接筋宽度 b(x)", "反解芯丝直径 d(x)"],
    key="inv_mode", horizontal=True)

col_inv1, col_inv2 = st.columns(2)
with col_inv1:
    inv_EI_expr = st.text_area("目标 EI_target(x) (N·mm²)", value="5 + 0.05*x", height=90, key="inv_EI_expr")
    inv_GJ_expr = st.text_area("目标 GJ_target(x) (N·mm²，可选)", value="", height=90, key="inv_GJ_expr")
    inv_EA_expr = st.text_area("目标 EA_target(x) (N，可选)", value="", height=90, key="inv_EA_expr")

with col_inv2:
    inv_primary = st.radio("主导刚度", ["EI", "GJ", "EA"], horizontal=True, key="inv_primary")
    inv_smooth_window = st.number_input("移动平均窗口", min_value=1, max_value=101, value=11, step=1, key="inv_smooth_window")
    inv_use_pitch_corr = st.checkbox("弹簧圈区未点胶螺距修正（扭转）", value=False, key="inv_pitch_corr")

if "反解连接筋宽度" in inv_mode:
    col_z1, col_z2, col_z3 = st.columns(3)
    with col_z1:
        inv_Z = st.number_input("轴向周期 Z (mm)", value=0.133, step=0.001, format="%.4f", key="inv_Z")
    with col_z2:
        inv_w_s = st.number_input("槽宽 w_s (mm)", value=0.03, step=0.001, format="%.4f", key="inv_w_s")
    with col_z3:
        inv_default_N = st.number_input("默认 N（海波管函数未指定时）", min_value=1, max_value=10,
                                        value=int(n_beam), step=1, key="inv_default_N")
    inv_d_expr = st.text_area("芯丝直径 d(x) (mm)", value="0.05 + 0.0003*x", height=90, key="inv_d_expr")
else:
    col_z1, col_z2 = st.columns(2)
    with col_z1:
        inv_w_s_d = st.number_input("槽宽 w_s (mm)", value=0.03, step=0.001, format="%.4f", key="inv_w_s_d")
    with col_z2:
        inv_d_max = st.number_input("芯丝直径上限 (mm)", value=0.23, step=0.01, format="%.4f", key="inv_d_max")
    inv_default_N_d = st.number_input("默认 N（海波管函数未指定时）", min_value=1, max_value=10,
                                      value=int(n_beam), step=1, key="inv_default_N_d")

st.info("反问题自动复用侧边栏的点胶区间、弹簧圈范围、传递系数和海波管函数。")

# ---- 反解 b ----
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
                st.error("Z 必须大于 w_s")
                st.stop()

            C_full = np.pi * inv_D_o
            x_inv = np.linspace(0, inv_L_total, 500)

            hypo_segments_inv = parse_hypo_functions(
                st.session_state.get('hypo_text_input', ''),
                default_N=int(inv_default_N))
            if not hypo_segments_inv:
                st.error("侧边栏海波管函数为空。")
                st.stop()
            N_x = np.zeros_like(x_inv, dtype=int)
            for i, xi in enumerate(x_inv):
                _, _, N_val = calc_b_Z_original(xi, hypo_segments_inv, default_N=int(inv_default_N))
                N_x[i] = N_val

            d_core_inv = safe_eval_array(inv_d_expr, x_inv)
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
                st.error("至少需要一个目标刚度。")
                st.stop()

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
                st.error(f"主导刚度 {inv_primary} 的目标未填写。")
                st.stop()

            valid_mask = np.isfinite(k_used) & (k_used > 1e-6) & (k_used < 1.0)

            with np.errstate(divide='ignore', invalid='ignore'):
                A_inv = (1.0 - k_used) / k_used * inv_Z / inv_w_s
                b_inv = C_full / (N_x * (A_inv + 1.0))
            b_inv = np.where(valid_mask, b_inv, np.nan)

            b_smooth = moving_average(b_inv, int(inv_smooth_window))

            with np.errstate(divide='ignore', invalid='ignore'):
                b_total_s = N_x * b_smooth
                Y_s = C_full - b_total_s
                k_smooth = 1.0 / (1.0 + (inv_w_s / inv_Z) * (Y_s / b_total_s))
            k_smooth = np.where(
                np.isfinite(k_smooth) & (b_smooth > 0) & (b_total_s < C_full),
                np.clip(k_smooth, 0.0, 1.0),
                np.nan
            )

            EI_actual = EI_core + eta_b_x * (k_smooth * EI_0)
            GJ_actual = GJ_core + eta_t_x * (k_smooth * GJ_0)
            EA_actual = EA_core + eta_a_x * (k_smooth * EA_0)

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

            st.subheader(f"误差统计（主导={inv_primary}）")
            c1, c2, c3 = st.columns(3)
            c1.metric("EI 最大相对误差", safe_metric(err_EI))
            c2.metric("GJ 最大相对误差", safe_metric(err_GJ))
            c3.metric("EA 最大相对误差", safe_metric(err_EA))

            fig_inv, axes_inv = plt.subplots(7, 1, figsize=(11, 26))
            fig_inv.suptitle("Inverse Problem: Recover b(x)", fontsize=14)

            axes_inv[0].plot(x_inv, EI_core, 'g:', linewidth=1.5, label='Core EI')
            if EI_target is not None: axes_inv[0].plot(x_inv, EI_target, 'b-', linewidth=2, label='Target EI')
            axes_inv[0].plot(x_inv, EI_actual, 'r--', linewidth=2, label='Actual EI')
            axes_inv[0].set_ylabel('EI'); axes_inv[0].grid(True); axes_inv[0].legend()

            axes_inv[1].plot(x_inv, GJ_core, 'g:', linewidth=1.5, label='Core GJ')
            if GJ_target is not None: axes_inv[1].plot(x_inv, GJ_target, 'b-', linewidth=2, label='Target GJ')
            axes_inv[1].plot(x_inv, GJ_actual, 'r--', linewidth=2, label='Actual GJ')
            axes_inv[1].set_ylabel('GJ'); axes_inv[1].grid(True); axes_inv[1].legend()

            axes_inv[2].plot(x_inv, EA_core, 'g:', linewidth=1.5, label='Core EA')
            if EA_target is not None: axes_inv[2].plot(x_inv, EA_target, 'b-', linewidth=2, label='Target EA')
            axes_inv[2].plot(x_inv, EA_actual, 'r--', linewidth=2, label='Actual EA')
            axes_inv[2].set_ylabel('EA'); axes_inv[2].grid(True); axes_inv[2].legend()

            axes_inv[3].plot(x_inv, b_inv, color='lightgray', linewidth=1, label='Raw b')
            axes_inv[3].plot(x_inv, b_smooth, 'b-', linewidth=2, label='Smoothed b (per rib)')
            axes_inv[3].set_ylabel('b (mm)'); axes_inv[3].grid(True); axes_inv[3].legend()

            if k_EI is not None: axes_inv[4].plot(x_inv, k_EI, color='cyan', linewidth=1.5, label='k from EI')
            if k_GJ is not None: axes_inv[4].plot(x_inv, k_GJ, color='magenta', linewidth=1.5, label='k from GJ')
            if k_EA is not None: axes_inv[4].plot(x_inv, k_EA, color='orange', linewidth=1.5, label='k from EA')
            axes_inv[4].plot(x_inv, k_smooth, 'r-', linewidth=2, label='k used')
            axes_inv[4].axhline(0, color='k', linestyle=':', alpha=0.3)
            axes_inv[4].axhline(1, color='k', linestyle=':', alpha=0.3)
            axes_inv[4].set_ylabel('k'); axes_inv[4].grid(True); axes_inv[4].legend()

            axes_inv[5].plot(x_inv, d_core_inv, 'g-', linewidth=2, label='Core d(x)')
            axes_inv[5].set_ylabel('d (mm)'); axes_inv[5].grid(True); axes_inv[5].legend()

            axes_inv[6].plot(x_inv, N_x, 'b-', linewidth=2, label='N(x)')
            axes_inv[6].set_xlabel('Distance from distal end (mm)')
            axes_inv[6].set_ylabel('N')
            axes_inv[6].set_yticks([1, 2, 3])
            axes_inv[6].grid(True); axes_inv[6].legend()

            fig_inv.tight_layout(rect=[0, 0, 1, 0.98])
            st.pyplot(fig_inv)

            st.subheader("反解结果")
            result_dict = {
                '位置 (mm)': np.round(x_inv, 2),
                'N(x)': N_x,
                '芯丝 d (mm)': np.round(d_core_inv, 4),
                '连接筋 b（每根, mm）': np.round(b_smooth, 4),
                '总宽 N·b (mm)': np.round(N_x * b_smooth, 4),
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
            st.download_button("下载完整反解结果 CSV",
                data=inv_result_df.to_csv(index=False).encode('utf-8-sig'),
                file_name="inverse_b_result.csv", mime="text/csv")

        except Exception as e:
            st.error(f"反问题求解出错：{e}")
            st.exception(e)

# ---- 反解 d ----
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
            C_full = np.pi * inv_D_o

            hypo_segments_inv = parse_hypo_functions(
                st.session_state.get('hypo_text_input', ''),
                default_N=int(inv_default_N_d))
            if not hypo_segments_inv:
                st.error("侧边栏海波管函数为空。")
                st.stop()

            x_inv = np.linspace(0, inv_L_total, 500)
            b_arr = np.zeros_like(x_inv)
            Z_arr = np.zeros_like(x_inv)
            N_x = np.zeros_like(x_inv, dtype=int)
            for i, xi in enumerate(x_inv):
                bv, Zv, Nv = calc_b_Z_smooth(xi, hypo_segments_inv, delta=2.0,
                                             default_N=int(inv_default_N_d))
                b_arr[i] = bv; Z_arr[i] = Zv; N_x[i] = Nv

            b_total_x = N_x * b_arr
            Y = C_full - b_total_x
            Z_safe = np.where(Z_arr > 0, Z_arr, 1e-9)
            b_safe = np.where(b_total_x > 0, b_total_x, 1e-9)
            k_x = 1.0 / (1.0 + (w_s / Z_safe) * (Y / b_safe))
            k_x = np.where((Z_arr > 0) & (b_arr > 0) & (b_total_x < C_full), k_x, np.nan)
            k_x = np.clip(k_x, 0.0, 1.0)

            I_0 = np.pi / 64.0 * (inv_D_o**4 - inv_D_i**4)
            J_0 = 2.0 * I_0
            A_0 = np.pi / 4.0 * (inv_D_o**2 - inv_D_i**2)
            EI_0 = inv_E_hypo * I_0
            GJ_0 = inv_G_hypo * J_0
            EA_0 = inv_E_hypo * A_0

            EI_hypo = k_x * EI_0
            GJ_hypo = k_x * GJ_0
            EA_hypo = k_x * EA_0

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

            EI_target = safe_eval_array(inv_EI_expr, x_inv) if inv_EI_expr.strip() else None
            GJ_target = safe_eval_array(inv_GJ_expr, x_inv) if inv_GJ_expr.strip() else None
            EA_target = safe_eval_array(inv_EA_expr, x_inv) if inv_EA_expr.strip() else None

            if EI_target is None and GJ_target is None and EA_target is None:
                st.error("至少需要一个目标刚度。")
                st.stop()

            def d_from_EI(EI_t):
                EI_ct = EI_t - eta_b_x * EI_hypo
                with np.errstate(invalid='ignore'):
                    return np.where(EI_ct > 0, (64.0 * EI_ct / (inv_E_core * np.pi)) ** 0.25, np.nan)

            def d_from_GJ(GJ_t):
                GJ_ct = GJ_t - eta_t_x * GJ_hypo
                with np.errstate(invalid='ignore'):
                    return np.where(GJ_ct > 0, (32.0 * GJ_ct / (inv_G_core * np.pi)) ** 0.25, np.nan)

            def d_from_EA(EA_t):
                EA_ct = EA_t - eta_a_x * EA_hypo
                with np.errstate(invalid='ignore'):
                    return np.where(EA_ct > 0, (4.0 * EA_ct / (inv_E_core * np.pi)) ** 0.5, np.nan)

            d_EI = d_from_EI(EI_target) if EI_target is not None else None
            d_GJ = d_from_GJ(GJ_target) if GJ_target is not None else None
            d_EA = d_from_EA(EA_target) if EA_target is not None else None

            d_map = {'EI': d_EI, 'GJ': d_GJ, 'EA': d_EA}
            d_used = d_map[inv_primary]
            if d_used is None:
                st.error(f"主导刚度 {inv_primary} 的目标未填写。")
                st.stop()

            valid_mask = np.isfinite(d_used) & (d_used > 1e-6) & (d_used < inv_d_max)
            d_inv = np.where(valid_mask, d_used, np.nan)
            d_smooth = moving_average(d_inv, int(inv_smooth_window))
            d_smooth = np.where((d_smooth > 0) & (d_smooth < inv_d_max), d_smooth, np.nan)

            EI_core_actual = inv_E_core * np.pi * d_smooth**4 / 64.0
            GJ_core_actual = inv_G_core * np.pi * d_smooth**4 / 32.0
            EA_core_actual = inv_E_core * np.pi * d_smooth**2 / 4.0

            EI_actual = EI_core_actual + eta_b_x * EI_hypo
            GJ_actual = GJ_core_actual + eta_t_x * GJ_hypo
            EA_actual = EA_core_actual + eta_a_x * EA_hypo

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

            st.subheader(f"误差统计（主导={inv_primary}）")
            c1, c2, c3 = st.columns(3)
            c1.metric("EI 最大相对误差", safe_metric(err_EI))
            c2.metric("GJ 最大相对误差", safe_metric(err_GJ))
            c3.metric("EA 最大相对误差", safe_metric(err_EA))

            fig_inv, axes_inv = plt.subplots(6, 1, figsize=(11, 22))
            fig_inv.suptitle("Inverse Problem: Recover d(x)", fontsize=14)

            axes_inv[0].plot(x_inv, EI_hypo, 'g:', linewidth=1.5, label='Hypo EI')
            if EI_target is not None: axes_inv[0].plot(x_inv, EI_target, 'b-', linewidth=2, label='Target EI')
            axes_inv[0].plot(x_inv, EI_actual, 'r--', linewidth=2, label='Actual EI')
            axes_inv[0].set_ylabel('EI'); axes_inv[0].grid(True); axes_inv[0].legend()

            axes_inv[1].plot(x_inv, GJ_hypo, 'g:', linewidth=1.5, label='Hypo GJ')
            if GJ_target is not None: axes_inv[1].plot(x_inv, GJ_target, 'b-', linewidth=2, label='Target GJ')
            axes_inv[1].plot(x_inv, GJ_actual, 'r--', linewidth=2, label='Actual GJ')
            axes_inv[1].set_ylabel('GJ'); axes_inv[1].grid(True); axes_inv[1].legend()

            axes_inv[2].plot(x_inv, EA_hypo, 'g:', linewidth=1.5, label='Hypo EA')
            if EA_target is not None: axes_inv[2].plot(x_inv, EA_target, 'b-', linewidth=2, label='Target EA')
            axes_inv[2].plot(x_inv, EA_actual, 'r--', linewidth=2, label='Actual EA')
            axes_inv[2].set_ylabel('EA'); axes_inv[2].grid(True); axes_inv[2].legend()

            axes_inv[3].plot(x_inv, b_arr, 'b-', linewidth=2, label='b(x) (per rib)')
            axes_inv[3].set_ylabel('b (mm)'); axes_inv[3].grid(True); axes_inv[3].legend()

            axes_inv[4].plot(x_inv, d_inv, color='lightgray', linewidth=1, label='Raw d')
            axes_inv[4].plot(x_inv, d_smooth, 'g-', linewidth=2, label='Smoothed d')
            axes_inv[4].set_ylabel('d (mm)'); axes_inv[4].grid(True); axes_inv[4].legend()

            axes_inv[5].plot(x_inv, N_x, 'b-', linewidth=2, label='N(x)')
            axes_inv[5].set_xlabel('Distance from distal end (mm)')
            axes_inv[5].set_ylabel('N')
            axes_inv[5].set_yticks([1, 2, 3])
            axes_inv[5].grid(True); axes_inv[5].legend()

            fig_inv.tight_layout(rect=[0, 0, 1, 0.98])
            st.pyplot(fig_inv)

            st.subheader("反解结果")
            result_dict = {
                '位置 (mm)': np.round(x_inv, 2),
                'N(x)': N_x,
                '海波管 b（每根, mm）': np.round(b_arr, 4),
                '折减系数 k': np.round(k_x, 4),
                '芯丝 d (mm)': np.round(d_smooth, 4),
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
            st.download_button("下载完整反解结果 CSV",
                data=inv_result_df.to_csv(index=False).encode('utf-8-sig'),
                file_name="inverse_d_result.csv", mime="text/csv")

        except Exception as e:
            st.error(f"反问题求解出错：{e}")
            st.exception(e)

# ==================== 三点弯曲对比 ====================
st.divider()
st.header("📐 三点弯曲对比")

col_bp1, col_bp2, col_bp3 = st.columns(3)
with col_bp1:
    bp_L = st.number_input("跨距 L (mm)", value=30.0, step=1.0, format="%.2f", key="bp_L")
    bp_x1 = st.number_input("跨距左端 x1 (mm)", value=0.0, step=1.0, format="%.2f", key="bp_x1")
with col_bp2:
    bp_P = st.number_input("中心力 P (N)", value=0.5, step=0.05, format="%.4f", key="bp_P")
    bp_measure_str = st.text_input("测量点（距左支撑, mm）",
                                   value="5,7.5,10,12.5,15,17.5,20,22.5,25",
                                   key="bp_measure_str")
with col_bp3:
    st.caption("使用侧边栏海波管函数，N 由函数第 5 列决定")

if st.button("计算三点弯曲", key="bp_calc_btn", type="primary"):
    try:
        bp_params = {
            'E_core': st.session_state.get('E_core_input', 200000.0),
            'G_core': st.session_state.get('G_core_input', 77000.0),
            'E_hypo': st.session_state.get('E_hypo_input', 50000.0),
            'G_hypo': st.session_state.get('G_hypo_input', 19231.0),
            'D_o': st.session_state.get('D_o_input', 0.33),
            'D_i': st.session_state.get('D_i_input', 0.23),
            'w_s': st.session_state.get('w_s_input', 0.03),
            'L_total': float(st.session_state.get('L_total_input', 350.0)),
            'F': st.session_state.get('F_input', 0.001),
            'T0': st.session_state.get('T0_input', 1.0),
            'spring_start': st.session_state.get('spring_start_input', 0),
            'spring_end': st.session_state.get('spring_end_input', 150),
            'n_beam': st.session_state.get('n_beam', 2),
        }

        bp_glue_intervals = []
        bp_glue_text = st.session_state.get('glue_input', '')
        if bp_glue_text.strip():
            for line in bp_glue_text.strip().splitlines():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) == 3:
                    try: bp_glue_intervals.append((float(parts[0]), float(parts[1]), parts[2]))
                    except: pass
        bp_params['glue_intervals'] = bp_glue_intervals

        bp_hypo_segments = parse_hypo_functions(
            st.session_state.get('hypo_text_input', ''),
            default_N=bp_params['n_beam'])
        if not bp_hypo_segments:
            st.error("海波管函数为空。")
            st.stop()

        bp_core_df = st.session_state.get('core_editor_data', default_core_v1)
        bp_eta = {k: st.session_state.get(k + '_input', eta_defaults[k]) for k in eta_keys}

        x_full = np.linspace(0, bp_params['L_total'], 1000)
        res = compute_version(x_full, bp_core_df, bp_hypo_segments, bp_params, bp_eta)
        EI_full = res[0]

        z = np.linspace(0, bp_L, 2001)
        x_local = bp_x1 + z
        EI_local = np.interp(x_local, x_full, EI_full)

        try:
            measure_points = [float(v.strip()) for v in bp_measure_str.split(',') if v.strip()]
        except Exception:
            st.error("测量点格式错误。")
            st.stop()

        delta_arr = np.array([three_point_deflection_at(z, EI_local, bp_P, bp_L, a) for a in measure_points])
        delta_C = three_point_deflection_at(z, EI_local, bp_P, bp_L, bp_L/2)
        EI_eff = bp_P * bp_L**3 / (48 * delta_C) if delta_C > 0 else float('nan')

        fig, axes = plt.subplots(2, 1, figsize=(11, 8))
        axes[0].plot(x_full, EI_full, 'b-', linewidth=2, label='EI(x)')
        axes[0].axvspan(bp_x1, bp_x1 + bp_L, alpha=0.15, color='orange', label='Span')
        axes[0].set_xlabel('Position along guidewire (mm)')
        axes[0].set_ylabel('EI (N·mm²)')
        axes[0].set_title('Bending Stiffness EI(x)')
        axes[0].grid(True); axes[0].legend()
        axes[0].set_xlim(0, min(bp_params['L_total'], 200))

        axes[1].plot(z, EI_local, 'b-', linewidth=2, label='EI local')
        axes[1].set_xlabel('Position within span z (mm)')
        axes[1].set_ylabel('EI (N·mm²)')
        axes[1].set_title(f'Local EI within Span (x1 = {bp_x1} mm)')
        axes[1].grid(True); axes[1].legend()
        st.pyplot(fig)

        st.subheader("三点弯曲挠度")
        df_delta = pd.DataFrame({
            '测量点 (mm)': measure_points,
            '挠度 (mm)': np.round(delta_arr, 5),
        })
        st.dataframe(df_delta, use_container_width=True)

        st.metric("等效弯曲刚度 EI_eff (N·mm²)", f"{EI_eff:.4f}")

        fig2, ax2 = plt.subplots(1, 1, figsize=(11, 5))
        defl_curve = np.array([three_point_deflection_at(z, EI_local, bp_P, bp_L, a) for a in z])
        ax2.plot(z, defl_curve, 'b-', linewidth=2, label='Deflection curve')
        ax2.scatter(measure_points, delta_arr, color='red', s=50, zorder=5, label='Measure points')
        ax2.set_xlabel('Position within span z (mm)')
        ax2.set_ylabel('Deflection (mm)')
        ax2.set_title(f'Three-point Bending Deflection (P = {bp_P} N, L = {bp_L} mm)')
        ax2.grid(True); ax2.legend()
        st.pyplot(fig2)

        from scipy.optimize import brentq
        def f(P_try):
            return three_point_deflection_at(z, EI_local, P_try, bp_L, bp_L/2) - 4.0
        try:
            P_req = brentq(f, 0.001, 200.0)
            st.metric("下压 4 mm 所需力 P", f"{P_req:.4f} N")
        except Exception:
            st.info("无法反推下压 4 mm 的力")

    except Exception as e:
        st.error(f"三点弯曲计算出错：{e}")
        st.exception(e)
