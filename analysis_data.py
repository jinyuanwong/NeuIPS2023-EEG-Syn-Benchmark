import mne
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import signal
import time
from functools import wraps

def timeout(seconds):
    """超时装饰器"""
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")

        @wraps(func)
        def wrapper(*args, **kwargs):
            # 设置信号处理器
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)  # 设置闹钟
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)  # 关闭闹钟
            return result
        return wrapper
    return decorator

@timeout(30)  # 设置30秒超时
def analyze_edf_file(file_path):
    """分析单个EDF文件，特别关注EEG通道"""
    print(f"\nAnalyzing file: {file_path}")
    
    # 读取EDF文件
    raw = mne.io.read_raw_edf(file_path, preload=True)
    
    # 打印所有通道名称
    print("\nAll channel names:")
    print(raw.ch_names)
    
    # 获取EEG通道的数据
    eeg_channels = ['EEG Fpz-Cz', 'EEG Pz-Oz']
    eeg_data = raw.get_data(picks=eeg_channels)
    
    # 打印EEG数据的详细信息
    print("\nEEG data information:")
    print(f"Type: {type(eeg_data)}")
    print(f"Shape: {eeg_data.shape}")
    print("\nFirst channel (Fpz-Cz) first 10 values:")
    print(eeg_data[0, :10])
    print("\nSecond channel (Pz-Oz) first 10 values:")
    print(eeg_data[1, :10])
    
    return raw, eeg_data

@timeout(30)  # 设置30秒超时
def visualize_channels(raw, save_dir, segment_duration=300, downsample_factor=10):
    """
    可视化所有通道的数据
    segment_duration: 每段显示的秒数
    downsample_factor: 降采样因子
    """
    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # 获取数据
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    print(f"Sampling frequency: {sfreq}")
    
    # 计算一个片段的点数
    points_per_segment = int(segment_duration * sfreq)
    
    # 选择前30秒的数据并降采样
    data_segment = data[:, :points_per_segment:downsample_factor]
    time_points = np.arange(data_segment.shape[1]) * downsample_factor / sfreq
    
    # 绘制所有通道
    plt.figure(figsize=(20, 15))
    n_channels = len(raw.ch_names)
    
    for i in range(n_channels):
        plt.subplot(n_channels, 1, i+1)
        plt.plot(time_points, data_segment[i, :])
        plt.title(f'Channel: {raw.ch_names[i]}')
        plt.ylabel('Amplitude')
        if i == n_channels-1:  # 只在最后一个子图显示x轴标签
            plt.xlabel('Time (seconds)')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'all_channels_30s.png')
    plt.close()
    
    # 单独保存每个通道的图
    for i in range(n_channels):
        plt.figure(figsize=(15, 5))
        plt.plot(time_points, data_segment[i, :])
        plt.title(f'Channel: {raw.ch_names[i]}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.savefig(save_dir / f'channel_{raw.ch_names[i]}_30s.png')
        plt.close()

def main():
    start_time = time.time()
    try:
        # 设置数据路径
        base_path = "./physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"
        save_dir = "./data/visualisation"
        
        # 分析第一个文件
        first_file = "SC4001E0-PSG.edf"
        file_path = os.path.join(base_path, first_file)
        
        if os.path.exists(file_path):
            try:
                raw, _ = analyze_edf_file(file_path)
                visualize_channels(raw, save_dir)
                print(f"Visualizations saved to {save_dir}")
            except TimeoutError as e:
                print(f"Operation timed out: {e}")
        else:
            print(f"File not found: {file_path}")
            
        # 列出目录中的所有EDF文件
        print("\nAll EDF files in directory:")
        for file in os.listdir(base_path):
            if file.endswith('.edf'):
                print(file)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        end_time = time.time()
        print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
