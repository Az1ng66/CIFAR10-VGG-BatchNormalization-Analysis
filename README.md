# CIFAR-10 Classification with VGG-A and Batch Normalization

This project implements a VGG-A based neural network for CIFAR-10 classification, featuring an in-depth analysis of Batch Normalization's impact on the loss landscape.

## 1. Project Overview
- **Architecture**: VGG-A (with and without Batch Normalization)
- **Dataset**: CIFAR-10
- **Total Parameters**: 9,756,426
- **Optimization**: Adam Optimizer, StepLR Scheduler

## 2. Main Results
- **Best Test Accuracy**: [85.04% ]
- **Loss Landscape Analysis**: Demonstrated that Batch Normalization significantly smoothens the optimization landscape, allowing for more stable training.
- **Insights**: Visualization of first-layer filters shows the model's ability to capture basic edge and color features.

## 3. Repository Structure
- `models/vgg.py`: Definition of VGG-A and VGG-A-BatchNorm models.
- `data/loaders.py`: Data loading and preprocessing scripts.
- `task1_best_model.py`: Script for training the optimized model.
- `task2_landscape.py`: Comparative experiment for loss landscape visualization.
- `viz_filters.py`: Tool for filter visualization.
- `get_stats.py`: Tool for counting model parameters.

## 4. Resource Links
- **Trained Weights**: [此处粘贴你的网盘链接]
- **Dataset**: Official CIFAR-10 dataset (downloaded via torchvision)
