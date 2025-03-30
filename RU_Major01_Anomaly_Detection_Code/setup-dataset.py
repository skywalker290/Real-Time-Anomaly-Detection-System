import os
import shutil
import random
from tqdm import tqdm  # For displaying progress bars

# Set paths
base_dir = 'Anomoly-Dataset'
anomaly_dir = os.path.join(base_dir, 'Anomoly-Videos-Clipped')
normal_dir = os.path.join(base_dir, 'Normal-Videos-Clipped')

# Destination folders
output_base = 'Anomaly-detection-clipped-Dataset'
splits = ['train', 'test', 'val']

# Split ratios
split_ratios = {'train': 0.8, 'test': 0.1, 'val': 0.1}

# Function to split and copy videos without category subfolders
def split_and_copy_files(video_files, split_ratios, category_name, pbar):
    # Shuffle files and calculate split sizes
    random.shuffle(video_files)
    train_size = int(split_ratios['train'] * len(video_files))
    test_size = int(split_ratios['test'] * len(video_files))
    
    # Create splits
    train_files = video_files[:train_size]
    test_files = video_files[train_size:train_size + test_size]
    val_files = video_files[train_size + test_size:]

    # Copy files to respective directories with progress bar
    for split, files in zip(splits, [train_files, test_files, val_files]):
        split_dir = os.path.join(output_base, split, category_name)
        os.makedirs(split_dir, exist_ok=True)
        for file in files:
            # Construct the correct source path
            src = file if category_name == 'Anomaly' else os.path.join(normal_dir, file)
            dest = os.path.join(split_dir, os.path.basename(file))
            shutil.copy(src, dest)
            pbar.update(1)

# List of all anomaly and normal videos
# Full paths are gathered for anomaly videos
anomaly_videos = [
    os.path.join(root, f)
    for root, _, files in os.walk(anomaly_dir)
    for f in files if f.endswith('.mp4')
]
normal_videos = [f for f in os.listdir(normal_dir) if f.endswith('.mp4')]

# Total videos for progress bar
total_videos = len(anomaly_videos) + len(normal_videos)

# Display progress bar for entire dataset
with tqdm(total=total_videos, desc="Processing dataset", unit="video") as pbar:
    # Copy Anomaly videos
    split_and_copy_files(anomaly_videos, split_ratios, 'Anomaly',pbar)

    # Copy Normal videos
    split_and_copy_files(normal_videos, split_ratios, 'Normal',pbar)
