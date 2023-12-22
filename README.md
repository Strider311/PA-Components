# Setup

## Installation

The first step is to create a python environment and install dependencies. Use python V3.11.6 and create a virtual environment:

```bash
python -m venv .venv
pip install -r requirements.txt
```

### Yolov8 setup

Depending on your hardware if you have gpu or not, install the appropriate pytorch v2.1.2 (stable) from the following link: https://pytorch.org/get-started/locally/

To install yolov8:
```bash
pip install ultralytics
```

To prepare datasets for yolo training, there is a script `Utilities/create_yolo_labels.py` which can help with preparing the labels directory for yolo v8 training. Refer to https://docs.ultralytics.com/datasets/detect/ for more information on dataset structure.
Helpful tools:
- https://roboflow.com/how-to-label/yolov8

To use yolov8, follow this guide as it is very thorough and explains how to use it: https://learnopencv.com/train-yolov8-on-custom-dataset/

