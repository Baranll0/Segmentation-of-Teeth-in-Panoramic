import os
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation, TrainingArguments, Trainer
from transformers import default_data_collator
from torchvision.transforms import ColorJitter
from datasets import DatasetDict
import evaluate
import torch
import numpy as np

# Veri yolları
IMG_DIR = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/data/processed/img"
MASK_DIR = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/data/processed/mask"

# Dataset yükleme
from src.preprocessing.dataset import load_datasets
dataset = load_datasets(IMG_DIR, MASK_DIR)

# Model ve Feature Extractor
model_checkpoint = "nvidia/mit-b0"
feature_extractor = AutoImageProcessor.from_pretrained(
    model_checkpoint,
    reduce_labels=True,
    size={"height": 256, "width": 256},  # Resize işlemi burada yapılıyor
)

model = AutoModelForSemanticSegmentation.from_pretrained(
    model_checkpoint,
    ignore_mismatched_sizes=True,
)

# Augmentation
jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)

def convert_to_rgb(image):
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image

def train_transforms(example_batch):
    images = [convert_to_rgb(jitter(x)) for x in example_batch["pixel_values"]]
    labels = [x for x in example_batch["label"]]
    inputs = feature_extractor(images, labels, return_tensors="pt")  # Resize burada otomatik uygulanıyor
    return inputs

def val_transforms(example_batch):
    images = [convert_to_rgb(x) for x in example_batch["pixel_values"]]
    labels = [x for x in example_batch["label"]]
    inputs = feature_extractor(images, labels, return_tensors="pt")  # Resize burada otomatik uygulanıyor
    return inputs

dataset["train"].set_transform(train_transforms)
dataset["validation"].set_transform(val_transforms)

# Training Arguments
training_args = TrainingArguments(
    output_dir="/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/outputs/segmentation_model",
    learning_rate=0.00006,
    num_train_epochs=10,
    per_device_train_batch_size=2,  # Batch size 2 olarak ayarlandı
    per_device_eval_batch_size=2,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=50,
    eval_steps=50,
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    fp16=True,  # Mixed Precision Training GPU bellek tüketimini azaltır
)

# Evaluation Metric
metric = evaluate.load("mean_iou")
def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        logits_tensor = torch.nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)
        pred_labels = logits_tensor.detach().cpu().numpy()

        return metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=len(feature_extractor.id2label),
            ignore_index=0,
            reduce_labels=True,
        )

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
    data_collator=default_data_collator,
)

# Eğitim
trainer.train()