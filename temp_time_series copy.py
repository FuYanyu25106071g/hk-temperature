# rain_temp_visual.py

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap

# -----------------------------
# 1. 获取香港天文台温度数据
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
    df = pd.read_csv(pd.compat.StringIO(response.text))
    df["Date"] = pd.to_datetime(df["YYYY-MM-DD"])
    df["MeanTemperature"] = df["MeanTemperature"].astype(float)
    df["MinTemperature"] = df["MinTemperature"].astype(float)
    df["MaxTemperature"] = df["MaxTemperature"].astype(float)
    return df

# -----------------------------
# 2. 创建动态雨水可视化
# -----------------------------
rain_cmap = LinearSegmentedColormap.from_list("rain_blue", ["#a0c4ff", "#4361ee", "#03045e"])

def create_animation(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(df['Date'].min(), df['Date'].max())
    ax.set_ylim(df['MinTemperature'].min() - 2, df['MaxTemperature'].max() + 2)
    ax.set_xlabel("日期")
    ax.set_ylabel("温度 (°C)")
    ax.set_title("动态雨水效果温度可视化", fontsize=16)
    ax.grid(True, alpha=0.3)

    # 创建多条“雨水线”
    lines = [ax.plot([], [], lw=2, color=rain_cmap(i / 5), alpha=0.5)[0] for i in range(5)]

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(frame):
        for i, line in enumerate(lines):
            offset = i * 0.5
            y = df['MeanTemperature'].values + np.sin(np.linspace(0, 2*np.pi, len(df)) + frame/5 + i) * offset
            line.set_data(df['Date'], y)
        return lines

    ani = FuncAnimation(fig, update, frames=len(df)*2, init_func=init, blit=False, interval=50)
    plt.show()

# -----------------------------
# 3. 主程序
# -----------------------------
if __name__ == "__main__":
    df = fetch_temperature_data()
    create_animation(df)
