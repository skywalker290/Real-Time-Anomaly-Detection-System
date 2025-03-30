model_dataset = "vivit-timesformer-d2"
dataset_root_path = "d2/Anomaly-Multiclass-Dataset"
batch_size = 1

# prompt: define all_video_file_paths which contain file path of all the videos in the directory /content/UCF101_subset

import os
all_video_file_paths = []
for root, _, files in os.walk(dataset_root_path):
    for file in files:
        if file.endswith((".mp4", ".avi")):  # Add other video extensions if needed
            all_video_file_paths.append(os.path.join(root, file))
            
class_labels = sorted({str(path).split("/")[3] for path in all_video_file_paths})
label2id = {label: i for i, label in enumerate(class_labels)}
id2label = {i: label for label, i in label2id.items()}

print(f"Unique classes: {list(label2id.keys())}.")



from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from transformers import  VivitConfig,VivitForVideoClassification, VivitImageProcessor

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#image_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")

#model = VideoMAEForVideoClassification.from_pretrained(
#    "MCG-NJU/videomae-base",
#    label2id=label2id,
#    id2label=id2label,
#    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
#)

#image_processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400",
                                                          label2id=label2id,
                                                          id2label=id2label,
                                                          ignore_mismatched_sizes=True
                                                          )

image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
#model = VivitForVideoClassification.from_pretrained(
#    "google/vivit-b-16x2-kinetics400",
#    label2id=label2id,
#    id2label=id2label,
#    ignore_mismatched_sizes=True)
model.to(device)

import pytorchvideo.data
import torchvision

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)
mean = image_processor.image_mean
std = image_processor.image_std
if "shortest_edge" in image_processor.size:
    height = width = image_processor.size["shortest_edge"]
else:
    height = image_processor.size["height"]
    width = image_processor.size["width"]
resize_to = (height, width)

num_frames_to_sample = model.config.num_frames
sample_rate = 4
fps = 30
clip_duration = num_frames_to_sample * sample_rate / fps

train_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(resize_to),
                    RandomHorizontalFlip(p=0.5),
                ]
            ),
        ),
    ]
)

train_dataset = pytorchvideo.data.Ucf101(
    data_path=os.path.join(dataset_root_path, "train"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
    decode_audio=False,
    transform=train_transform,
)
val_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    Resize(resize_to),
                ]
            ),
        ),
    ]
)

val_dataset = pytorchvideo.data.Ucf101(
    data_path=os.path.join(dataset_root_path, "val"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
    decode_audio=False,
    transform=val_transform,
)

test_dataset = pytorchvideo.data.Ucf101(
    data_path=os.path.join(dataset_root_path, "test"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
    decode_audio=False,
    transform=val_transform,
)
print(train_dataset.num_videos, val_dataset.num_videos, test_dataset.num_videos)

import imageio
import numpy as np
from IPython.display import Image

def unnormalize_img(img):
    """Un-normalizes the image pixels."""
    img = (img * std) + mean
    img = (img * 255).astype("uint8")
    return img.clip(0, 255)

def create_gif(video_tensor, filename="sample.gif"):
    """Prepares a GIF from a video tensor.

    The video tensor is expected to have the following shape:
    (num_frames, num_channels, height, width).
    """
    frames = []
    for video_frame in video_tensor:
        frame_unnormalized = unnormalize_img(video_frame.permute(1, 2, 0).numpy())
        frames.append(frame_unnormalized)
    kargs = {"duration": 0.25}
    imageio.mimsave(filename, frames, "GIF", **kargs)
    return filename

def display_gif(video_tensor, gif_name="sample.gif"):
    """Prepares and displays a GIF from a video tensor."""
    video_tensor = video_tensor.permute(1, 0, 2, 3)
    gif_filename = create_gif(video_tensor, gif_name)
    return Image(filename=gif_filename)

sample_video = next(iter(train_dataset))
video_tensor = sample_video["video"]


import wandb

# Replace with your actual W&B API key
API_KEY = "f7b65d8399dd6262084e166f128e83f97b568e6e"

# Log in to W&B with the API key
wandb.login(key=API_KEY,relogin=True)

# Project details
PROJECT = model_dataset
MODEL_NAME = model_dataset
DATASET = "UCF Anomaly Multiclass Classification"

# Initialize W&B with an increased timeout
wandb.init(
    project=PROJECT,
    tags=[MODEL_NAME, DATASET],
    notes="model training"  # Increase timeout to 300 seconds
)




from transformers import TrainingArguments, Trainer

model_name = model_dataset.split("-")[-2]
new_model_name = model_dataset
num_epochs = 10

args = TrainingArguments(
    new_model_name,
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_ratio=0.1,
    logging_steps=10,
    eval_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
    max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
    fp16=True,
    report_to="wandb"
)

import evaluate

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def collate_fn(examples):
    # permute to (num_frames, num_channels, height, width)
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)
import torch

train_results = trainer.train()
trainer.push_to_hub()
