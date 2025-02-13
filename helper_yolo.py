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


def load_model(preset_name: str) -> YOLO:
    """
    Initializes and loads a YOLO model based on a given preset name.

    Args:
        preset_name (str): The name of the pre-trained model to be loaded.

    Returns:
        YOLO: An instance of the YOLO model.
    """
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
    """
    Loads a YOLO model, performs prediction on a specified image, and optionally plots the image with detected objects.

    Args:
        img_path (str): Path to the image file on which detection is to be performed.
        preset_name (str, optional): Name of the pre-trained model file. Defaults to "yolov8n.pt".
        imgsz (int, optional): Size to which the image is resized before prediction. Defaults to 640.
        conf (float, optional): Confidence threshold for the predictions. Defaults to 0.5.
        line_width (int, optional): Line width of the bounding boxes drawn on the detected objects. Defaults to 2.
        plot_image (bool, optional): Whether to plot the image with detected objects. Defaults to True.
        plot_title (str, optional): Title for the plot. Defaults to "Detected Objects in Sample Image by the Pre-trained YOLOv8 Model on COCO Dataset".
        figsize (tuple[int, int], optional): Figure size of the plot. Defaults to (20, 15).
        fontsize (int, optional): Font size of the plot title. Defaults to 20.
        return_model_and_img (bool, optional): Whether to return the model and the image array along with the detections. Defaults to False.

    Returns:
        Union[YOLO, Tuple[YOLO, np.ndarray]]: The YOLO model, and optionally the image array if `return_model_and_img` is True.
    """
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


def load_yaml_content(yaml_file_path, print_content: bool = True) -> dict:
    """
    Loads and optionally prints the content of a YAML file.

    Args:
        yaml_file_path (str): The file path to the YAML file to be loaded.
        print_content (bool, optional): Whether to print the content of the YAML file. Defaults to True.

    Returns:
        dict: The content of the YAML file.
    """
    with open(yaml_file_path, "r") as file:
        yaml_content = yaml.load(file, Loader=yaml.FullLoader)
        if print_content:
            print(yaml.dump(yaml_content, default_flow_style=False))
    return yaml_content


def check_img_dataset_values(
    img_train_path: str, img_valid_path: str, format: str = ".jpg"
):
    """
    Checks and prints the number of images and unique image sizes in training and validation datasets.

    Args:
        img_train_path (str): The file path to the training images.
        img_valid_path (str): The file path to the validation images.
        format (str, optional): The image file format to be considered. Defaults to ".jpg".

    Returns:
        None
    """
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
    """
    Plots random sample images from a specified directory.

    Args:
        path (str): Directory path containing images.
        format (str, optional): Image file format to look for. Defaults to ".jpg".
        n_images (int, optional): Number of images to sample and plot. Defaults to 8.
        nrows (int, optional): Number of rows in the subplot grid. Defaults to 2.
        ncols (int, optional): Number of columns in the subplot grid. Defaults to 4.
        figsize (tuple[int, int], optional): Figure size of the plot. Defaults to (20, 10).
        plot_title (str, optional): Title of the plot. Defaults to "Sample Images from Path".
        fontsize (int, optional): Font size for the plot title. Defaults to 20.

    Returns:
        None
    """
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
) -> dict:
    """
    Trains a YOLO model with specified parameters.

    Args:
        model (YOLO): The YOLO model to be trained.
        yaml_file_path (str): The file path to the dataset configuration YAML file.
        epochs (int, optional): Number of epochs to train for. Defaults to 100.
        imgsz (int, optional): Input image size. Defaults to 640.
        device (int, optional): Device to run the training on. Defaults to 0 (for CUDA device).
        patience (int, optional): Patience for early stopping. Defaults to 50.
        batch (int, optional): Batch size. Defaults to 32.
        optimizer (str, optional): Type of optimizer to use. Defaults to "auto".
        lr0 (float, optional): Initial learning rate. Defaults to 0.0001.
        lrf (float, optional): Final learning rate factor. Defaults to 0.1.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        seed (int, optional): Seed for random number generators. Defaults to 0.

    Returns:
        dict: Training results.
    """

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
) -> None:
    """
    Plots a learning curve for training and validation losses over epochs.

    Args:
        df (pd.DataFrame): DataFrame containing the loss data across epochs.
        train_loss_col (str): Column name for the training loss.
        val_loss_col (str): Column name for the validation loss.
        plot_title (str): Title of the plot.
        train_color (str, optional): Color for the training loss line. Defaults to "#141140".
        train_linestyle (str, optional): Line style for the training loss line. Defaults to "-".
        valid_color (str, optional): Color for the validation loss line. Defaults to "orangered".
        valid_linestyle (str, optional): Line style for the validation loss line. Defaults to "--".
        linewidth (int, optional): Width of the line. Defaults to 2.

    Returns:
        None
    """
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
) -> None:
    """
    Plots a learning curve for a specific metric over epochs.

    Args:
        df (pd.DataFrame): DataFrame containing the metric data across epochs.
        metric_col (str): Column name for the metric to plot.
        plot_title (str): Title of the plot.
        color (str, optional): Color of the metric line. Defaults to "#141140".
        linestyle (str, optional): Line style of the metric line. Defaults to "-".
        linewidth (int, optional): Width of the line. Defaults to 2.

    Returns:
        None
    """
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


def plot_norm_confusion_matrix(cm_path: str) -> None:
    """
    Displays a normalized confusion matrix from a specified file path.

    Args:
        cm_path (str): Path to the image file of the normalized confusion matrix.

    Returns:
        None
    """
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
) -> pd.DataFrame:
    """
    Loads training results from a CSV file, optionally plots learning curves and confusion matrix, and returns the results as a DataFrame.

    Args:
        post_training_path (str, optional): Path to the directory containing training results. Defaults to "./runs/detect/train".
        plot_learning_curves (bool, optional): Whether to plot learning curves for losses and metrics. Defaults to True.
        plot_confusion_matrix (bool, optional): Whether to display the normalized confusion matrix. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame containing the training results.
    """
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


def get_best_model(post_training_path: str = "./runs/detect/train") -> YOLO:
    """
    Loads the best YOLO model based on the weights saved during training.

    Args:
        post_training_path (str, optional): Path to the directory containing the best model's weights. Defaults to "./runs/detect/train".

    Returns:
        YOLO: The best YOLO model.
    """
    best_model_path = os.path.join(post_training_path, "weights/best.pt")
    best_model = YOLO(best_model_path)
    return best_model


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
) -> None:
    """
    Predicts and plots random sample images with detected objects using the specified YOLO model.

    Args:
        model: The YOLO model to use for predictions.
        img_path (str): Path to the directory containing images for prediction.
        conf (float, optional): Confidence threshold for the predictions. Defaults to 0.5.
        imgsz (int, optional): Size to which the images are resized before prediction. Defaults to 640.
        format (str, optional): Image file format to consider for predictions. Defaults to ".jpg".
        n_images (int, optional): Number of images to sample and predict. Defaults to 9.
        nrows (int, optional): Number of rows in the subplot grid. Defaults to 3.
        ncols (int, optional): Number of columns in the subplot grid. Defaults to 3.
        plot_title (str, optional): Title for the plot of predictions. Defaults to "Random Image Predictions".
        figsize (tuple[int, int], optional): Figure size for the plot. Defaults to (20, 21).
        fontsize (int, optional): Font size for the plot title. Defaults to 24.

    Returns:
        None
    """

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
) -> None:
    """
    Performs predictions on a selected number of sample images from a specified directory, using the given YOLO model, and plots the results.

    This function selects a subset of images, performs object detection on them, and displays the annotated images in a grid. The images are selected based on evenly spaced intervals within the directory.

    Args:
        model: The YOLO model used for prediction.
        img_path (str): The path to the directory containing the images for prediction.
        conf (float, optional): The confidence threshold for the predictions. Defaults to 0.5.
        imgsz (int, optional): The size to which to resize the images before performing predictions. Defaults to 640.
        format (str, optional): The file format of the images to consider for prediction. Defaults to ".jpg".
        n_images (int, optional): The number of images to predict and display. Defaults to 9.
        nrows (int, optional): The number of rows in the grid to display the images. Defaults to 3.
        ncols (int, optional): The number of columns in the grid to display the images. Defaults to 3.
        plot_title (str, optional): The title displayed above the grid of images. Defaults to "Image Predictions".
        figsize (tuple[int, int], optional): The size of the figure to display the grid of images. Defaults to (20, 21).
        fontsize (int, optional): The font size of the plot title. Defaults to 24.

    Returns:
        None
    """
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
    final_video_name: str = "./src/predicted_video.mp4",
) -> None:
    """
    Predicts objects in a video using the specified YOLO model, converts the processed video to mp4 format.

    Args:
        model: The YOLO model to use for video prediction.
        video_path (str): Path to the input video file.
        show_video_notebook (bool, optional): Whether to display the processed video in a Jupyter Notebook. Defaults to True.
        video_avi_path (str, optional): Path where the processed video in AVI format is saved. Defaults to "./runs/detect/predict/sample_video.avi".
        final_video_name (str, optional): Name of the final video file in mp4 format. Defaults to "predicted_video.mp4".

    Returns:
        None
    """
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


def export_model(best_model, format: str = "onnx"):
    best_model.export(format=format)
