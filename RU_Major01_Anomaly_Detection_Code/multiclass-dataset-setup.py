import os
import shutil
import random
from tqdm import tqdm  # For displaying progress bars

# Set paths
base_dir = 'Anomoly-Dataset'
anomaly_dir = os.path.join(base_dir, 'Anomoly-Videos')

# Destination folders
output_base = 'Anomaly-Multiclass-Dataset'
splits = ['train', 'test', 'val']

# Split ratios
split_ratios = {'train': 0.7, 'test': 0.2, 'val': 0.1}

# Function to split and copy anomaly videos into specific folder structures
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
        # Create category subfolder in each split directory
        split_dir = os.path.join(output_base, split, category_name)
        os.makedirs(split_dir, exist_ok=True)
        for file in files:
            # Construct the correct source path
            src = file
            dest = os.path.join(split_dir, os.path.basename(file))
            
            # Copy only if file doesn't already exist
            if not os.path.exists(dest):
                shutil.copy(src, dest)
                pbar.update(1)

# Process each category in Anomaly videos for all splits
all_anomaly_videos = {
    category: [
        os.path.join(root, f)
        for root, _, files in os.walk(os.path.join(anomaly_dir, category))
        for f in files if f.endswith('.mp4')
    ]
    for category in os.listdir(anomaly_dir) if os.path.isdir(os.path.join(anomaly_dir, category))
}

# Total videos for progress bar
total_videos = sum(len(files) for files in all_anomaly_videos.values())

# Display progress bar for entire dataset
with tqdm(total=total_videos, desc="Processing dataset", unit="video") as pbar:
    # Copy Anomaly videos for each category into each split folder
    for category, videos in all_anomaly_videos.items():
        split_and_copy_files(videos, split_ratios, category, pbar)
