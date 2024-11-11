from transformers import Trainer, TrainingArguments, AdamW
from model_configuration import *
from preprocessing import create_dataset
from data_handling import frames_convert_and_create_dataset_dictionary
from model_configuration import initialise_model
import wandb
from dotenv import load_dotenv
import os
env_path =  ".env"
load_dotenv(env_path)

import model_configuration
from model_configuration import compute_metrics
import cv2
import av
from data_handling import sample_frame_indices, read_video_pyav

Project_Name = "ViVit For Anomaly Detection"

def load_data(dataset_path):
    '''
    dataset_path : path to the dataset folder whose inner structure 
    is of type dataset_path/train, dataset_path/test, dataset_path/val
    '''
    path_files = dataset_path
    video_dict, class_labels = frames_convert_and_create_dataset_dictionary(path_files)

    print("length of Video Dictionary:",len(video_dict))

    class_labels = sorted(class_labels)
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}

    print(f"Unique classes: {list(label2id.keys())}.")

    print("Creating Dataset...")
    shuffled_dataset = create_dataset(video_dict)

    return shuffled_dataset


def Training_configuration(video_dataset):
    '''
    video_dataset: dataset created in the load_data() and returned
    after shuffling.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shuffled_dataset = video_dataset
    model = model_configuration.initialise_model(shuffled_dataset, device)

    training_output_dir = "/tmp/results"
    
    training_args = TrainingArguments(
        output_dir=training_output_dir,         
        num_train_epochs=3,             
        per_device_train_batch_size=2,   
        per_device_eval_batch_size=2,    
        learning_rate=5e-05,            
        weight_decay=0.01,              
        logging_dir="./logs",           
        logging_steps=10,                
        seed=42,                       
        eval_strategy="steps",    
        eval_steps=10,                   
        warmup_steps=int(0.1 * 20),      
        optim="adamw_torch",          
        lr_scheduler_type="linear",      
        fp16=True,  
        report_to="wandb"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-05, betas=(0.9, 0.999), eps=1e-08)
    
    trainer = Trainer(
        model=model,                      
        args=training_args,              
        train_dataset=shuffled_dataset["train"],      
        eval_dataset=shuffled_dataset["test"],       
        optimizers=(optimizer, None),  
        compute_metrics = compute_metrics
    )

    return trainer


def training(trainer):

    wandb_key =  os.getenv("WANDB_API_KEY")
    wandb.login(key=wandb_key)

    PROJECT = Project_Name
    MODEL_NAME = "google/vivit-b-16x2-kinetics400"
    DATASET = "UCF_Video_dataset"

    wandb.init(project=PROJECT, 
           tags=[MODEL_NAME, DATASET],
           notes ="Fine tuning ViViT with ucf101-subset")
    
    with wandb.init(project=PROJECT, job_type="train", 
           tags=[MODEL_NAME, DATASET],
           notes =f"Fine tuning {MODEL_NAME} with {DATASET}."):
           train_results = trainer.train()

    trainer.save_model("model")
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    custom_path = "./model"

    with wandb.init(project=PROJECT, job_type="models"):
        artifact = wandb.Artifact("ViViT-Fine-tuned", type="model")
        artifact.add_dir(custom_path)
        wandb.save(custom_path)
        wandb.log_artifact(artifact) 

