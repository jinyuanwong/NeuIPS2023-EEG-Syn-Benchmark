from copy import deepcopy
import mne
import numpy as np
from pathlib import Path
from mne.datasets.sleep_physionet.age import fetch_data

# 设置输出目录
output_dir = Path("./data/physionet-sleep-data-npy")
output_dir.mkdir(parents=True, exist_ok=True)

# 原始代码的数据获取设置
subject_ids = None
recording_ids = None
load_eeg_only = True
crop_wake_mins = 30

if subject_ids is None:
    subject_ids = range(83)
if recording_ids is None:
    recording_ids = [1, 2]

# 原始代码的数据获取方法
paths = fetch_data(
    subjects=subject_ids,
    recording=recording_ids, on_missing='warn')

# 原始代码的通道映射
ch_mapping = {
    'EOG horizontal': 'eog',
    'Resp oro-nasal': 'misc',
    'EMG submental': 'misc',
    'Temp rectal': 'misc',
    'Event marker': 'misc'
}
exclude = list(ch_mapping.keys()) if load_eeg_only else ()

# 原始代码的数据处理循环
for p in paths:
    raw = mne.io.read_raw_edf(p[0], preload=True, exclude=exclude)
    annots = mne.read_annotations(p[1])
    raw = raw.set_annotations(annots, emit_warning=False)
    # Rename EEG channels
    ch_names = {i: i.replace('EEG ', '') for i in raw.ch_names if 'EEG' in i}
    raw = raw.rename_channels(ch_names)
    mask = [
        x[-1] in ['1', '2', '3', '4', 'R'] for x in annots.description]
    sleep_event_inds = np.where(mask)[0]
    # Crop raw
    tmin = annots[int(sleep_event_inds[0])]['onset'] - crop_wake_mins * 60
    tmax = annots[int(sleep_event_inds[-1])]['onset'] + crop_wake_mins * 60
    raw.crop(tmin=max(tmin, raw.times[0]), tmax=min(tmax, raw.times[-1]))
    raw = raw.filter(l_freq=None, h_freq=18)

    # 原始的输出路径代码（注释掉）
    '''
    path_annots = p[1].replace(".edf", "-annotation.npy").replace("physionet-sleep-data",
                                                                  "physionet-sleep-data-npy")
    print(f"Saving the annotation into {path_annots}")
    Path(path_annots).parent.mkdir(parents=True, exist_ok=True)
    annots = raw.annotations
    np.save(file=path_annots, arr=annots)
    '''

    # 新的输出路径代码
    file_name = Path(p[1]).name
    output_annot = output_dir / file_name.replace(".edf", "-annotation.npy")
    print(f"Saving the annotation into {output_annot}")
    np.save(file=output_annot, arr=annots)

    for cha in raw.ch_names:
        file_name = Path(p[0]).name
        output_file = output_dir / file_name.replace(".edf", f"-{cha}.npy")
        print(f"Saving channel {cha} to {output_file}")
        raw_pick = deepcopy(raw).pick_channels(ch_names=[cha])
        data = raw_pick.get_data()
        np.save(file=output_file, arr=data)

print("Done")

# 以下是单文件处理的代码，暂时注释掉
'''
# 设置输入文件和输出目录
input_file = Path("./physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4001E0-PSG.edf")
output_dir = Path("./data/physionet-sleep-data-npy")
output_dir.mkdir(parents=True, exist_ok=True)

# 读取EDF文件
print(f"Processing file: {input_file}")
raw = mne.io.read_raw_edf(input_file, preload=True)

# 保存每个通道的数据
for cha in raw.ch_names:
    output_file = output_dir / f"{input_file.stem}-{cha}.npy"
    print(f"Saving channel {cha} to {output_file}")
    raw_pick = deepcopy(raw).pick_channels(ch_names=[cha])
    data = raw_pick.get_data()
    np.save(file=output_file, arr=data)
'''
