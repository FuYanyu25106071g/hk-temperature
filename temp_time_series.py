import matplotlib
matplotlib.use('TkAgg')
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
import io

# -----------------------------
# 1. 获取香港天文台主站每日温度数据
# -----------------------------
def fetch_temperature_data():
    url = "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php"
    params = {
        "dataType": "CLMTEMP",  # Daily Mean Temperature
        "rformat": "csv",
        "station": "HKO"
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    # 跳过前两行说明，指定分隔符为逗号
    df = pd.read_csv(io.StringIO(response.text), skiprows=2)
    # 自动检测日期列名
    date_col = None
    for col in df.columns:
        if "date" in col.lower() or "年月日" in col:
            date_col = col
            break
    if date_col is None:
        date_col = df.columns[0]  # 默认取第一列
    # 过滤掉无效数据行
    # 先尝试将日期列解析为日期，无法解析的行会变为NaT
    df["Date"] = pd.to_datetime(df[date_col], errors='coerce')
    df = df[df["Date"].notna()].copy()
    # 自动检测温度列名
    temp_cols = [c for c in df.columns if "mean" in c.lower() or "平均" in c]
    min_cols = [c for c in df.columns if "min" in c.lower() or "最低" in c]
    max_cols = [c for c in df.columns if "max" in c.lower() or "最高" in c]
    df["MeanTemperature"] = df[temp_cols[0]].astype(float) if temp_cols else np.nan
    df["MinTemperature"] = df[min_cols[0]].astype(float) if min_cols else np.nan
    df["MaxTemperature"] = df[max_cols[0]].astype(float) if max_cols else np.nan
    # 过滤掉温度为 NaN 的行
    df = df[df[["MeanTemperature", "MinTemperature", "MaxTemperature"]].notna().all(axis=1)].copy()
    return df

# -----------------------------
# 2. 设置渐变色
# -----------------------------
rain_cmap = LinearSegmentedColormap.from_list("rain_blue", ["#a0c4ff", "#4361ee", "#03045e"])

# -----------------------------
# 3. 创建动态动画图
# -----------------------------
def create_animation(df):

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(df['Date'].min(), df['Date'].max())
    ax.set_ylim(df['MinTemperature'].min() - 2, df['MaxTemperature'].max() + 2)
    ax.set_xlabel("日期")
    ax.set_ylabel("温度 (°C)")
    ax.set_title("动态雨水艺术温度可视化", fontsize=16)
    ax.grid(True, alpha=0.2)

    # 多条温度曲线（雨水流动）
    lines = [ax.plot([], [], lw=2, color=rain_cmap(i / 5), alpha=0.5)[0] for i in range(5)]

    # 粒子参数
    n_drops = 20
    drop_x = np.random.choice(df['Date'], n_drops)
    # 时间类型转换为 int64 以便插值
    date_ints = df['Date'].astype('int64')
    drop_x_ints = pd.Series(drop_x).astype('int64')
    drop_y = np.interp(drop_x_ints, date_ints, df['MeanTemperature'])
    drop_offsets = np.random.uniform(0, 2*np.pi, n_drops)
    drop_speeds = np.random.uniform(0.5, 1.5, n_drops)
    drop_colors = [rain_cmap(np.random.rand()) for _ in range(n_drops)]
    drops = [ax.scatter([], [], s=60, color=drop_colors[i], alpha=0.7, edgecolors='white', linewidths=1.5) for i in range(n_drops)]

    # 波纹参数
    ripples = [ax.scatter([], [], s=[], color=drop_colors[i], alpha=0.3) for i in range(n_drops)]

    # 数据说明文本（左上角）
    info_text = ax.text(0.01, 0.98, "香港天文台每日平均温度", transform=ax.transAxes, fontsize=13, color="#4361ee", va='top', ha='left', alpha=0.8, bbox=dict(facecolor='white', alpha=0.3, edgecolor='none'))

    # 当前帧数据文本（右上角）
    frame_text = ax.text(0.99, 0.98, "", transform=ax.transAxes, fontsize=13, color="#03045e", va='top', ha='right', alpha=0.9, bbox=dict(facecolor='white', alpha=0.3, edgecolor='none'))

    def init():
        for line in lines:
            line.set_data([], [])
        for drop in drops:
            drop.set_offsets(np.empty((0, 2)))
        for ripple in ripples:
            ripple.set_offsets(np.empty((0, 2)))
            ripple.set_sizes([])
        frame_text.set_text("")
        return lines + drops + ripples + [frame_text, info_text]

    def update(frame):
        # 动态温度曲线
        for i, line in enumerate(lines):
            offset = i * 0.5
            y = df['MeanTemperature'].values + np.sin(np.linspace(0, 2*np.pi, len(df)) + frame/5 + i) * offset
            line.set_data(df['Date'], y)

        # 动态雨滴粒子
        for i, drop in enumerate(drops):
            # 雨滴沿温度曲线下落
            x = drop_x[i]
            x_int = x.astype('int64')
            base_y = np.interp(x_int, date_ints, df['MeanTemperature'])
            y = base_y + np.sin(frame/10 + drop_offsets[i]) * 2 - frame * drop_speeds[i] * 0.05
            drop.set_offsets([[x, y]])

            # 波纹扩散
            ripple_size = max(0, (frame % 40) * 2)
            ripple_y = y - 0.5
            ripples[i].set_offsets([[x, ripple_y]])
            ripples[i].set_sizes([ripple_size])

        # 当前帧的日期和温度数值（取温度曲线第一个点）
        idx = frame % len(df)
        date_str = df['Date'].iloc[idx].strftime('%Y-%m-%d')
        temp_val = df['MeanTemperature'].iloc[idx]
        frame_text.set_text(f"日期: {date_str}\n平均温度: {temp_val:.1f}°C")

        return lines + drops + ripples + [frame_text, info_text]

    ani = FuncAnimation(fig, update, frames=len(df)*2, init_func=init, blit=True, interval=50)
    # Save animation as GIF
    try:
        ani.save('temperature_animation.gif', writer='pillow', fps=20)
        print("Animation saved as temperature_animation.gif")
    except Exception as e:
        print(f"Failed to save animation: {e}\nMake sure the pillow package is installed.")
    plt.show()

# -----------------------------
# 4. 主程序
# -----------------------------
if __name__ == "__main__":
    # 恢复联网获取香港天文台温度数据
    try:
        df = fetch_temperature_data()
        # 只保留最近60天数据
        df = df.sort_values('Date')
        df = df.iloc[-60:]
        if df.empty:
            raise ValueError("No valid data")
        print("已成功获取香港天文台数据，正在可视化...")
        create_animation(df)
    except Exception as e:
        print("无法获取官网数据，自动切换为本地模拟数据。")
        # 本地模拟数据，保证可视化效果出色
        dates = pd.date_range(start='2025-01-01', periods=60, freq='D')
        temps = 18 + 8 * np.sin(np.linspace(0, 2*np.pi, len(dates))) + np.random.normal(0, 1, len(dates))
        df = pd.DataFrame({
            'Date': dates,
            'MeanTemperature': temps,
            'MinTemperature': temps - np.random.uniform(1, 3, len(dates)),
            'MaxTemperature': temps + np.random.uniform(1, 3, len(dates))
        })
        print("正在展示本地模拟温度数据的艺术可视化效果...")
        create_animation(df)
