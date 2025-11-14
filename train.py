import os
import wandb
from ultralytics import YOLO

def main():
    
    wandb.init(project="enph353-fizz-yolo", name="yolo_fizz_obstacles_v1")

    dataset_yaml = os.path.join(os.path.dirname(__file__),
                                "fizz_yolo_dataset",
                                "fizz_dataset.yaml")

    model = YOLO("yolo12l.pt")
    
    results = model.train(
        data=dataset_yaml,
        epochs=50,              # adjust as needed
        imgsz=640,
        batch=128,
        project="enph353-fizz-yolo",
        name="yolo_fizz_obstacles_v1",
        pretrained=True,
        optimizer="AdamW",
        lr0=1e-3,
        patience=20,
        verbose=True,
        device=[0,1,2,3],
        exist_ok=True,
        # W&B integration
        val=True
    )

    wandb.finish()

    # The best model weights will be something like:
    # runs/detect/enph353-fizz-yolo/yolo_fizz_obstacles_v1/weights/best.pt

if __name__ == "__main__":
    main()