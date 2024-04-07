from helper_yolo import *
from cfg import CFG


def finetuning_pipeline():
    print("1. Checking Dataset Balance and Shape...\n")
    check_img_dataset_values(
        img_train_path=CFG.img_train_path, img_valid_path=CFG.img_valid_path
    )

    print("\n\n2. Load yolov8n.pt Model and predict sample...\n")
    model = load_model_predict_sample(img_path=CFG.sample_img_path)

    print("\n\n3. Predict random samples from Training Set...\n")
    predict_random_samples(model=model, img_path=CFG.img_train_path)

    print("\n\n4. Predict random samples from Validation Set...\n")
    predict_random_samples(model=model, img_path=CFG.img_valid_path)

    print("\n\n5. Finetuning yolov8n.pt Model...\n")
    results = train_yolo_model(
        model=model,
        yaml_file_path=CFG.yaml_file_path,
        epochs=CFG.epochs,
        imgsz=CFG.imgsz,
        device=CFG.device,
        patience=CFG.patience,
        batch=CFG.batch,
        optimizer=CFG.optimizer,
        lr0=CFG.lr0,
        lrf=CFG.lrf,
        dropout=CFG.dropout,
        seed=CFG.seed,
    )


def evaluation_pipeline():
    print("6. Plotting Evaluation Metrics...\n")
    df = get_results_df()
    return df


def compare_pipeline():
    # load base model
    base_model = load_model(preset_name=CFG.yolo_preset_name)

    # load best model from Training
    best_model = get_best_model()

    print("8. Base Model predicts Validation Samples...\n")
    predict_samples(
        model=base_model,
        img_path=CFG.img_valid_path,
        plot_title="Base Model Vehicle Predictions",
    )

    print("\n\n9. Best Model predicts Validation Samples...\n")
    predict_samples(
        model=best_model,
        img_path=CFG.img_valid_path,
        plot_title="Finetuned Best Model Vehicle Predictions",
    )

    print("\n\n10. Base Model predicts a video...\n")
    predict_video(
        model=base_model,
        video_path=CFG.sample_video_path,
        final_video_name="base_model_predictions.mp4",
    )

    print("\n\n11. Best Model predicts a video...\n")
    predict_video(
        model=best_model,
        video_path=CFG.sample_video_path,
        final_video_name="best_model_predictions.mp4",
    )
