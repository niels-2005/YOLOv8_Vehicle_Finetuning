# Import necessary libraries
import os
import random
import shutil
import subprocess

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from IPython.display import Video
from PIL import Image
from ultralytics import YOLO


def load_model(preset_name: str):
    model = YOLO(preset_name)
    return model


def load_model_predict_sample(
    img_path: str,
    preset_name: str = "yolov8n.pt",
    imgsz: int = 640,
    conf: float = 0.5,
    line_width: int = 2,
    plot_image: bool = True,
    plot_title: str = "Detected Objects in Sample Image by the Pre-trained YOLOv8 Model on COCO Dataset",
    figsize: tuple[int, int] = (20, 15),
    fontsize: int = 20,
    return_model_and_img: bool = False,
):

    model = YOLO(preset_name)
    results = model.predict(source=img_path, imgsz=imgsz, conf=conf)
    sample_image = results[0].plot(line_width=line_width)
    sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

    if plot_image:
        plt.figure(figsize=figsize)
        plt.imshow(sample_image)
        plt.title(plot_title, fontsize=fontsize)
        plt.axis("off")
        plt.show()

    if return_model_and_img:
        return model, sample_image

    return model


def load_yaml_content(yaml_file_path, print_content: bool = True):
    with open(yaml_file_path, "r") as file:
        yaml_content = yaml.load(file, Loader=yaml.FullLoader)
        if print_content:
            print(yaml.dump(yaml_content, default_flow_style=False))
    return yaml_content


def check_img_dataset_values(
    img_train_path: str, img_valid_path: str, format: str = ".jpg"
):
    num_train_images = 0
    num_valid_images = 0

    train_image_sizes = set()
    valid_image_sizes = set()

    # Check train images sizes and count
    for filename in os.listdir(img_train_path):
        if filename.endswith(format):
            num_train_images += 1
            image_path = os.path.join(img_train_path, filename)
            with Image.open(image_path) as img:
                train_image_sizes.add(img.size)

    # Check validation images sizes and count
    for filename in os.listdir(img_valid_path):
        if filename.endswith(format):
            num_valid_images += 1
            image_path = os.path.join(img_valid_path, filename)
            with Image.open(image_path) as img:
                valid_image_sizes.add(img.size)

    # Print the results
    print(f"Number of training images: {num_train_images}")
    print(f"Number of validation images: {num_valid_images}")

    # Check if all images in training set have the same size
    if len(train_image_sizes) == 1:
        print(f"All training images have the same size: {train_image_sizes.pop()}")
    else:
        print("Training images have varying sizes.")

    # Check if all images in validation set have the same size
    if len(valid_image_sizes) == 1:
        print(f"All validation images have the same size: {valid_image_sizes.pop()}")
    else:
        print("Validation images have varying sizes.")


def plot_random_samples(
    path: str,
    format: str = ".jpg",
    n_images: int = 8,
    nrows: int = 2,
    ncols: int = 4,
    figsize: tuple[int, int] = (20, 10),
    plot_title: str = "Sample Images from Path",
    fontsize: int = 20,
):

    image_files = [file for file in os.listdir(path) if file.endswith(format)]

    selected_images = random.sample(image_files, n_images)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for ax, img_file in zip(ax.ravel(), selected_images):
        img_path = os.path.join(path, img_file)
        image = Image.open(img_path)
        ax.imshow(image)
        ax.axis("off")

    plt.suptitle(plot_title, fontsize=fontsize)
    plt.tight_layout()
    plt.show()


def train_yolo_model(
    model,
    yaml_file_path: str,
    epochs: int = 100,
    imgsz: int = 640,
    device: int = 0,
    patience: int = 50,
    batch: int = 32,
    optimizer: str = "auto",
    lr0: float = 0.0001,
    lrf: float = 0.1,
    dropout: float = 0.1,
    seed: int = 0,
):

    results = model.train(
        data=yaml_file_path,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        patience=patience,
        batch=batch,
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        dropout=dropout,
        seed=seed,
    )
    return results


def plot_loss_learning_curve(
    df: pd.DataFrame,
    train_loss_col: str,
    val_loss_col: str,
    plot_title: str,
    train_color: str = "#141140",
    train_linestyle: str = "-",
    valid_color: str = "orangered",
    valid_linestyle: str = "--",
    linewidth: int = 2,
):

    plt.figure(figsize=(12, 5))
    sns.lineplot(
        data=df,
        x="epoch",
        y=train_loss_col,
        label="Train Loss",
        color=train_color,
        linestyle=train_linestyle,
        linewidth=linewidth,
    )
    sns.lineplot(
        data=df,
        x="epoch",
        y=val_loss_col,
        label="Validation Loss",
        color=valid_color,
        linestyle=valid_linestyle,
        linewidth=linewidth,
    )
    plt.title(plot_title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_metric_learning_curve(
    df: pd.DataFrame,
    metric_col: str,
    plot_title: str,
    color: str = "#141140",
    linestyle: str = "-",
    linewidth: int = 2,
):

    plt.figure(figsize=(12, 5))
    sns.lineplot(
        data=df,
        x="epoch",
        y=metric_col,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
    )
    plt.title(plot_title)
    plt.show()


def plot_norm_confusion_matrix(cm_path: str):
    cm_img = cv2.imread(cm_path)
    cm_img = cv2.cvtColor(cm_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 8))
    plt.imshow(cm_img)
    plt.axis("off")
    plt.show()


def get_results_df(
    post_training_path: str = "./runs/detect/train",
    plot_learning_curves: bool = True,
    plot_confusion_matrix: bool = True,
):
    results_csv_path = os.path.join(post_training_path, "results.csv")

    df = pd.read_csv(results_csv_path)
    df.columns = df.columns.str.strip()

    if plot_learning_curves:
        plot_loss_learning_curve(
            df=df,
            train_loss_col="train/box_loss",
            val_loss_col="val/box_loss",
            plot_title="Box Loss Learning Curve",
        )
        plot_loss_learning_curve(
            df=df,
            train_loss_col="train/cls_loss",
            val_loss_col="val/cls_loss",
            plot_title="Classification Loss Learning Curve",
        )
        plot_loss_learning_curve(
            df=df,
            train_loss_col="train/dfl_loss",
            val_loss_col="val/dfl_loss",
            plot_title="Distribution Focal Loss Learning Curve",
        )
        plot_metric_learning_curve(
            df=df, metric_col="metrics/precision(B)", plot_title="Metrics Precision (B)"
        )
        plot_metric_learning_curve(
            df=df, metric_col="metrics/recall(B)", plot_title="Metrics Recall (B)"
        )
        plot_metric_learning_curve(
            df=df, metric_col="metrics/mAP50(B)", plot_title="Metrics mAP50 (B)"
        )
        plot_metric_learning_curve(
            df=df, metric_col="metrics/mAP50-95(B)", plot_title="Metrics mAP50-95 (B)"
        )

    if plot_confusion_matrix:
        cm_path = os.path.join(post_training_path, "confusion_matrix_normalized.png")
        plot_norm_confusion_matrix(cm_path=cm_path)

    return df


def get_best_model(post_training_path: str = "./runs/detect/train"):
    best_model_path = os.path.join(post_training_path, "weights/best.pt")
    best_model = YOLO(best_model_path)
    return best_model


def get_best_model_metrics(best_model):
    metrics = best_model.val(split="val")

    metrics_df = pd.DataFrame.from_dict(
        metrics.results_dict, orient="index", columns=["Metric Value"]
    )

    print("Best Model Metric Values: \n")
    return metrics_df.round(3)


def predict_random_samples(
    model,
    img_path: str,
    conf: float = 0.5,
    imgsz: int = 640,
    format: str = ".jpg",
    n_images: int = 9,
    nrows: int = 3,
    ncols: int = 3,
    plot_title: str = "Random Image Predictions",
    figsize: tuple[int, int] = (20, 21),
    fontsize: int = 24,
):

    image_files = [file for file in os.listdir(img_path) if file.endswith(format)]

    selected_images = random.sample(image_files, n_images)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.suptitle(plot_title, fontsize=fontsize)

    for i, ax in enumerate(ax.flatten()):
        image_path = os.path.join(img_path, selected_images[i])
        results = model.predict(source=image_path, imgsz=imgsz, conf=conf)
        annotated_image = results[0].plot(line_width=1)
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        ax.imshow(annotated_image_rgb)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def predict_samples(
    model,
    img_path: str,
    conf: float = 0.5,
    imgsz: int = 640,
    format: str = ".jpg",
    n_images: int = 9,
    nrows: int = 3,
    ncols: int = 3,
    plot_title: str = "Image Predictions",
    figsize: tuple[int, int] = (20, 21),
    fontsize: int = 24,
):

    image_files = [file for file in os.listdir(img_path) if file.endswith(format)]

    n_files = len(image_files)
    selected_images = [image_files[i] for i in range(0, n_files, n_files // n_images)]

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.suptitle(plot_title, fontsize=fontsize)

    for i, ax in enumerate(ax.flatten()):
        image_path = os.path.join(img_path, selected_images[i])
        results = model.predict(source=image_path, imgsz=imgsz, conf=conf)
        annotated_image = results[0].plot(line_width=1)
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        ax.imshow(annotated_image_rgb)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def predict_video(
    model,
    video_path: str,
    show_video_notebook: bool = True,
    video_avi_path: str = "./runs/detect/predict/sample_video.avi",
    final_video_name: str = "predicted_video.mp4",
    embed: bool = True,
    video_width: int = 960,
):

    model.predict(source=video_path, save=True)

    if show_video_notebook:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "panic",
                "-i",
                f"{video_avi_path}",
                f"{final_video_name}",
            ]
        )
        Video(final_video_name, embed=embed, width=video_width)


