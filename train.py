from argparse import ArgumentParser

from ultralytics import YOLO

from utils import parse_unknown_arguments

CFG = {
    "model" : "yolo11s.pt",
    "epochs" : 5,
    "imgsz" : 640, 
    "workers" : 16, 
    "batch" : 16,
    "single_cls" : True
}

INVALID_ARGS = ["data"]

def overrides():
    parser = ArgumentParser(prog = "train_yolov11", description="Train a YOLOv11 model", epilog="See https://docs.ultralytics.com/modes/train/#train-settings for options. `model` and `data` cannot be specified.")
    args, extra = parser.parse_known_args()
    try:
        extra = parse_unknown_arguments(extra)
    except ValueError as e:
        raise ValueError(
                f"Error parsing extra arguments: `{' '.join(extra)}`. {e}\n\n"
                f"{parser.format_help()}"
            )
    return {**vars(args), **extra}

if __name__ == "__main__":
    CFG = {**CFG, **overrides()}
    if any([iarg in CFG for iarg in INVALID_ARGS]):
        raise ValueError(f'An invalid argument (one of: {INVALID_ARGS}) found in the config:\n{CFG}')
    
    model = YOLO(CFG.pop("model"))  # Load a pretrained model
    results = model.train(data="data.yaml", **CFG)