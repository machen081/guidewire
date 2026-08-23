
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

st.set_page_config(page_title="导丝刚度分析工具（函数输入）", layout="wide")

# ==================== 颜色列表 ====================
COLORS = ['green', 'blue', 'orange', 'purple', 'red', 'cyan', 'magenta', 'yellow', 'black', 'brown']

# ==================== 安全表达式求值 ====================
def safe_eval(expr, x_val):
   """
   安全地计算一个包含 x 的表达式。
   只允许数字、运算符、括号、x、数学函数等。
   """
   # 白名单：允许的字符
   allowed = set("0123456789+-*/(). xXeE")
   # 检查表达式是否包含非法字符
   for ch in expr:
       if ch not in allowed:
           raise ValueError(f"表达式包含非法字符: {ch}")
   # 将 x 替换为具体的数值
   expr_sub = expr.replace('x', f'({x_val})')
   try:
       return float(eval(expr_sub))
   except:
       raise ValueError(f"无法计算表达式: {expr}")

# ==================== 海波管分段函数解析 ====================
def parse_hypo_functions(text):
   """
   解析海波管分段函数文本。
   每行格式：start, end, b_expr, Z_expr
   返回列表 [(start, end, b_expr, Z_expr), ...]
   """
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
   """
   根据分段函数计算 x 处的 b 和 Z。
   """
   for start, end, b_expr, Z_expr in segments:
       if start  0, 1.0 / (1.0 + (w_s / denom) * (Y / b_arr)), 1.0)

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

# ==================== 初始化 session_state ====================
if 'saved_versions' not in st.session_state:
   st.session_state.saved_versions = []

if 'current_hypo_text' not in st.session_state:
   st.session_state.current_hypo_text = "0,10,0.036,0.058\n10,90,-0.000000154786*x**3+0.000017054434*x**2+0.0011531092*x+0.0229182506,-0.000008789096*x**2+0.0018164096*x+0.0407148136\n90,350,0.152,0.133"

# 默认芯丝表格
default_core = pd.DataFrame([
   {"start": 0, "end": 15, "d_start": 0.0508, "d_end": 0.0508},
   {"start": 15, "end": 25, "d_start": 0.0508, "d_end": 0.0762},
   {"start": 25, "end": 100, "d_start": 0.0762, "d_end": 0.0762},
   {"start": 100, "end": 160, "d_start": 0.0762, "d_end": 0.127},
   {"start": 160, "end": 350, "d_start": 0.127, "d_end": 0.127},
])

if 'current_core' not in st.session_state:
   st.session_state.current_core = default_core.copy()

if 'current_name' not in st.session_state:
   st.session_state.current_name = "版本二（函数输入）"

if 'current_spring' not in st.session_state:
   st.session_state.current_spring = (0, 150)

if 'current_glue' not in st.session_state:
   st.session_state.current_glue = "0,1,full\n90,100,core_spring\n345,346,core_hypo"

# ==================== 侧边栏 ====================
with st.sidebar:
   st.header("当前版本编辑")
   st.session_state.current_name = st.text_input("版本名称", value=st.session_state.current_name)

   st.subheader("材料参数")
   E_core = st.number_input("芯丝杨氏模量 (MPa)", value=200000, step=1000, key="edit_E_core")
   G_core = st.number_input("芯丝剪切模量 (MPa)", value=77000, step=1000, key="edit_G_core")
   E_hypo = st.number_input("海波管杨氏模量 (MPa)", value=50000, step=1000, key="edit_E_hypo")
   G_hypo = st.number_input("海波管剪切模量 (MPa)", value=19231, step=1000, key="edit_G_hypo")

   st.subheader("几何参数")
   D_o = st.number_input("海波管外径 (mm)", value=0.33, step=0.01, key="edit_D_o")
   D_i = st.number_input("海波管内径 (mm)", value=0.23, step=0.01, key="edit_D_i")
   w_s = st.number_input("槽宽 (mm)", value=0.03, step=0.01, key="edit_w_s")
   L_total = st.number_input("导丝总长 (mm)", value=350, step=10, key="edit_L_total")

   st.subheader("载荷参数")
   F = st.number_input("远端横向力 F (N)", value=0.001, step=0.001, format="%.4f", key="edit_F")
   T0 = st.number_input("近端扭矩 T0 (N·mm)", value=1.0, step=0.1, key="edit_T0")

   st.subheader("弹簧圈范围")
   spring_start = st.number_input("弹簧圈起始位置 (mm)", value=st.session_state.current_spring[0], step=5, key="edit_spring_start")
   spring_end = st.number_input("弹簧圈结束位置 (mm)", value=st.session_state.current_spring[1], step=5, key="edit_spring_end")
   st.session_state.current_spring = (spring_start, spring_end)

   st.subheader("点胶区间")
   glue_text = st.text_area("格式: start,end,type (每行一个)", value=st.session_state.current_glue, key="edit_glue")
   st.session_state.current_glue = glue_text

   st.subheader("芯丝直径分段表")
   edited_core = st.data_editor(st.session_state.current_core, num_rows="dynamic", key="edit_core_editor")
   st.session_state.current_core = edited_core

   st.subheader("海波管开槽函数 (b(x), Z(x))")
   st.markdown("每行一个区间：`start,end,b_expr,Z_expr`\n\n"
               "可以使用 `x` 作为变量，支持 `+ - * / **` 和括号。\n\n"
               "例如：`10,90,-0.000000154786*x**3+0.000017054434*x**2+0.0011531092*x+0.0229182506,"
               "-0.000008789096*x**2+0.0018164096*x+0.0407148136`")
   hypo_text = st.text_area("海波管函数", value=st.session_state.current_hypo_text, height=200, key="edit_hypo_text")
   st.session_state.current_hypo_text = hypo_text

   st.subheader("传递系数")
   with st.expander("完全点胶区 (full)"):
       full_b = st.number_input("弯曲", value=1.0, step=0.05, key="full_b")
       full_t = st.number_input("扭转", value=1.0, step=0.05, key="full_t")
       full_a = st.number_input("轴向", value=1.0, step=0.05, key="full_a")
   with st.expander("芯丝+弹簧圈点胶区 (core_spring)"):
       cs_b = st.number_input("弯曲", value=0.9, step=0.05, key="cs_b")
       cs_t = st.number_input("扭转", value=0.9, step=0.05, key="cs_t")
       cs_a = st.number_input("轴向", value=0.0, step=0.05, key="cs_a")
   with st.expander("芯丝+海波管点胶区 (core_hypo)"):
       ch_b = st.number_input("弯曲", value=1.0, step=0.05, key="ch_b")
       ch_t = st.number_input("扭转", value=1.0, step=0.05, key="ch_t")
       ch_a = st.number_input("轴向", value=1.0, step=0.05, key="ch_a")
   with st.expander("有弹簧圈无点胶区"):
       sp_b = st.number_input("弯曲", value=0.9, step=0.05, key="sp_b")
       sp_t = st.number_input("扭转", value=0.6, step=0.05, key="sp_t")
       sp_a = st.number_input("轴向", value=0.0, step=0.05, key="sp_a")
   with st.expander("无弹簧圈无点胶区"):
       ns_b = st.number_input("弯曲", value=0.85, step=0.05, key="ns_b")
       ns_t = st.number_input("扭转", value=0.35, step=0.05, key="ns_t")
       ns_a = st.number_input("轴向", value=0.0, step=0.05, key="ns_a")

   # 保存当前版本
   if st.button("保存当前版本", type="primary"):
       # 解析点胶区间
       glue_intervals = []
       if glue_text.strip():
           for line in glue_text.strip().splitlines():
               parts = [p.strip() for p in line.split(',')]
               if len(parts) == 3:
                   try:
                       glue_intervals.append((float(parts[0]), float(parts[1]), parts[2]))
                   except:
                       pass

       # 解析海波管函数
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

   # 预设版本加载
   st.subheader("加载预设版本")
   col1, col2, col3 = st.columns(3)
   if col1.button("版本一"):
       st.session_state.current_name = "版本一（阶梯）"
       st.session_state.current_core = pd.DataFrame([
           {"start": 0, "end": 15, "d_start": 0.0508, "d_end": 0.0508},
           {"start": 15, "end": 25, "d_start": 0.0508, "d_end": 0.0762},
           {"start": 25, "end": 100, "d_start": 0.0762, "d_end": 0.0762},
           {"start": 100, "end": 160, "d_start": 0.0762, "d_end": 0.127},
           {"start": 160, "end": 350, "d_start": 0.127, "d_end": 0.127},
       ])
       st.session_state.current_hypo_text = (
           "0,10,0.036,0.058\n"
           "10,20,0.049,0.0663\n"
           "20,30,0.058,0.0747\n"
           "30,40,0.071,0.083\n"
           "40,50,0.086,0.0913\n"
           "50,60,0.096,0.0997\n"
           "60,70,0.102,0.108\n"
           "70,80,0.112,0.1163\n"
           "80,90,0.126,0.1246\n"
           "90,350,0.152,0.133"
       )
       st.session_state.current_spring = (0, 150)
       st.session_state.current_glue = "0,1,full\n90,100,core_spring\n345,346,core_hypo"
       st.rerun()
   if col2.button("版本二"):
       st.session_state.current_name = "版本二（连续函数）"
       st.session_state.current_core = pd.DataFrame([
           {"start": 0, "end": 15, "d_start": 0.0508, "d_end": 0.0508},
           {"start": 15, "end": 25, "d_start": 0.0508, "d_end": 0.0762},
           {"start": 25, "end": 100, "d_start": 0.0762, "d_end": 0.0762},
           {"start": 100, "end": 160, "d_start": 0.0762, "d_end": 0.127},
           {"start": 160, "end": 350, "d_start": 0.127, "d_end": 0.127},
       ])
       st.session_state.current_hypo_text = (
           "0,10,0.036,0.058\n"
           "10,90,-0.000000154786*x**3+0.000017054434*x**2+0.0011531092*x+0.0229182506,"
           "-0.000008789096*x**2+0.0018164096*x+0.0407148136\n"
           "90,350,0.152,0.133"
       )
       st.session_state.current_spring = (0, 150)
       st.session_state.current_glue = "0,1,full\n90,100,core_spring\n345,346,core_hypo"
       st.rerun()
   if col3.button("版本三"):
       st.session_state.current_name = "版本三（新设计）"
       st.session_state.current_core = pd.DataFrame([
           {"start": 0, "end": 15, "d_start": 0.0508, "d_end": 0.0508},
           {"start": 15, "end": 25, "d_start": 0.0508, "d_end": 0.0889},
           {"start": 25, "end": 70, "d_start": 0.0889, "d_end": 0.0889},
           {"start": 70, "end": 126, "d_start": 0.0889, "d_end": 0.14224},
           {"start": 126, "end": 175, "d_start": 0.14224, "d_end": 0.14224},
           {"start": 175, "end": 350, "d_start": 0.14224, "d_end": 0.22098},
       ])
       st.session_state.current_hypo_text = (
           "0,10,0.036,0.058\n"
           "10,70,-0.000000154786*x**3+0.00001307278*x**2+0.0011531092*x+0.023316416,"
           "-37/3600000*x**2+0.0014388889*x+0.0446388888\n"
           "70,145,0.115,0.095\n"
           "145,180,-0.000030204*(x-145)**2+(0.037-1225*(-0.000030204))/35*(x-145)+0.115,"
           "0.095+0.038*(x-145)/35\n"
           "180,350,0.152,0.133"
       )
       st.session_state.current_spring = (0, 120)
       st.session_state.current_glue = "0,1,full\n90,100,core_spring\n345,346,core_hypo"
       st.rerun()

# ==================== 主区域 ====================
st.header("已保存版本")
if not st.session_state.saved_versions:
   st.info("请在左侧编辑参数并点击“保存当前版本”。")
else:
   # 显示版本列表
   for idx, ver in enumerate(st.session_state.saved_versions):
       col1, col2, col3 = st.columns([3,1,1])
       col1.write(f"**{ver['name']}**")
       if col2.button("加载到左侧", key=f"load_{idx}"):
           st.session_state.current_name = ver['name']
           st.session_state.current_core = ver['core_df'].copy()
           st.session_state.current_spring = (ver['spring_start'], ver['spring_end'])
           st.session_state.current_glue = "\n".join([f"{s},{e},{t}" for s,e,t in ver['glue_intervals']])
           # 将海波管函数反写为文本
           hypo_lines = []
           for seg in ver['hypo_segments']:
               hypo_lines.append(f"{seg[0]},{seg[1]},{seg[2]},{seg[3]}")
           st.session_state.current_hypo_text = "\n".join(hypo_lines)
           # 传递系数
           st.session_state.full_b = ver['eta']['full_b']
           st.session_state.full_t = ver['eta']['full_t']
           st.session_state.full_a = ver['eta']['full_a']
           st.session_state.cs_b = ver['eta']['core_spring_b']
           st.session_state.cs_t = ver['eta']['core_spring_t']
           st.session_state.cs_a = ver['eta']['core_spring_a']
           st.session_state.ch_b = ver['eta']['core_hypo_b']
           st.session_state.ch_t = ver['eta']['core_hypo_t']
           st.session_state.ch_a = ver['eta']['core_hypo_a']
           st.session_state.sp_b = ver['eta']['spring_b']
           st.session_state.sp_t = ver['eta']['spring_t']
           st.session_state.sp_a = ver['eta']['spring_a']
           st.session_state.ns_b = ver['eta']['no_spring_b']
           st.session_state.ns_t = ver['eta']['no_spring_t']
           st.session_state.ns_a = ver['eta']['no_spring_a']
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
           # 使用当前侧边栏的 L_total 作为采样长度，也可以取所有版本的最大长度
           x = np.linspace(0, L_total, 500)

           fig1, axes1 = plt.subplots(3, 1, figsize=(10, 12))
           fig1.suptitle("刚度对比")
           axes1[0].set_ylabel('弯曲刚度 EI (N·mm²)')
           axes1[1].set_ylabel('扭转刚度 GJ (N·mm²)')
           axes1[2].set_xlabel('距远端距离 (mm)')
           axes1[2].set_ylabel('轴向刚度 EA (N)')
           for ax in axes1:
               ax.grid(True)

           fig2, axes2 = plt.subplots(3, 1, figsize=(10, 12))
           fig2.suptitle("海波管连接筋应力对比")
           axes2[0].set_ylabel('弯曲正应力 (MPa)')
           axes2[1].set_ylabel('扭转剪切应力 (MPa)')
           axes2[2].set_xlabel('距远端距离 (mm)')
           axes2[2].set_ylabel('等效应力 (MPa)')
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
