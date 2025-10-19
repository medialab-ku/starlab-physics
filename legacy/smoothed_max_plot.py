import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    'text.usetex': True,
    'font.size': 14,
    'font.family': 'Times New Roman',
    "text.latex.preamble": r"\usepackage{amsmath}",
})

def f_prime(t, a):
    return -a + np.sqrt(np.maximum(t, 0)**2 + a**2)

def f_double_prime(t, a):
    result = np.zeros_like(t)
    positive_t_indices = t > 0
    result[positive_t_indices] = t[positive_t_indices] / np.sqrt(t[positive_t_indices]**2 + a**2)
    return result

# --- Plotting Setup ---
a_values = [0.01, 0.05, 0.1]
t_values = np.linspace(0, 1.0, 1000)

# 색상 팔레트 (진한 → 연한)
colors_fprime = ["#1f77b4", "#4c9be8", "#a6cee3"]   # 블루 계열 (F')
colors_fdouble = ["#1b7837", "#33a02c", "#66c2a5"]  # 그린 계열 (F'')

plt.figure(figsize=(6, 3.5))

# 먼저 모든 곡선 그림
for i, a_value in enumerate(a_values):
    y_prime = f_prime(t_values, a_value)
    y_double_prime = f_double_prime(t_values, a_value)

    plt.plot(t_values, y_prime,
             color=colors_fprime[i],
             linestyle="-",
             linewidth=1.5)

    plt.plot(t_values, y_double_prime,
             color=colors_fdouble[i],
             linestyle="--",
             linewidth=1.5)

# 그룹 대표선만 legend에 추가
line_fprime, = plt.plot([], [], color="black", linestyle="-", linewidth=2, label=r"$F'(\phi)$")
line_fdouble, = plt.plot([], [], color="black", linestyle="--", linewidth=2, label=r"$F''(\phi)$")

# 보조선
plt.axhline(1, color='gray', linestyle='--', linewidth=1.0)
plt.axvline(0, color='k', linestyle=':', linewidth=1.0)

# 축 라벨
plt.xlabel(r'$\phi$', fontsize=20)
plt.xlim(0, 1.0)
plt.ylim(0, 1.0)

# Formatter 설정 (0.0 겹침 방지)
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

def y_fmt(val, pos):
    if np.isclose(val, 0.0):
        return ""           
    return f"{val:g}"

ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_fmt))

# Layout & 저장
plt.tight_layout()
plt.subplots_adjust(left=0.06, right=0.98, top=0.98, bottom=0.14)
plt.legend(fontsize=16)
plt.savefig("smoothed_max_grouped.png", bbox_inches='tight', pad_inches=0, dpi=300)

plt.show()