import subprocess
import os
import shutil
import yaml


def train_yolo(data_yaml, weights, epochs, batch_size, img_size, exp_name):
    """
    Invokes the YOLOv5 training process.
    """
    cmd = [
        "python3", "yolov5/train.py",
        "--data", data_yaml,
        "--weights", weights,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--img", str(img_size),
        "--name", exp_name
    ]
    print("Running training with command:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():

    labeled_images_dir = "./labeled/images"
    labeled_labels_dir = "./labeled/labels"

    
    labeled_data_yaml = "./data.yaml"
    
    initial_epochs = 10
    retrain_epochs = 10
    batch_size = 16
    img_size = 640
    confidence_threshold = 0.9
    num_classes = 2  
    names = ["Animal", "Human"] 
    
    initial_exp_name = "initial_train"
    detect_exp_name = "detect_pseudo"
    combined_exp_name = "retrain_combined"
    
    base_weights = "yolov5s.pt"
    

    train_yolo(labeled_data_yaml, base_weights, initial_epochs, batch_size, img_size, initial_exp_name)
    
    weights_trained = f"runs/train/{initial_exp_name}/weights/best.pt"
    

if __name__ == '__main__':
    main()
