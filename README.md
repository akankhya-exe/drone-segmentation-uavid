# Semantic Segmentation of UAV Imagery: Atrous-ECA U-Net vs ResNet-18

This repository contains the codebase for training, evaluating, profiling, and visualizing a custom **Atrous-ECA U-Net** against a standard **ResNet-18 U-Net** baseline on the UAVid dataset.

The core objective of this project is to demonstrate architectural efficiency. By integrating Atrous (Dilated) convolutions for multi-scale context and Efficient Channel Attention (ECA) for cross-channel feature weighting, the custom architecture is designed to capture complex, high-resolution urban geometries from drone footage without relying on heavy pre-trained weights.

Trained on: x2 T4 GPU

## 📊 Head-to-Head Results (Ablation Study)

To prove the sample efficiency of the custom architecture, I conducted an ablation study forcing both the ResNet-18 baseline and my Custom model to learn from absolute scratch, alongside an "Industry Standard" ImageNet-pretrained baseline.

| Architecture | Initialization | mIoU | F1 Score (Dice) | Inference Speed |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet-18 Baseline** | ImageNet (Pre-trained) | 0.8067 | 0.8930 | 47.70 FPS |
| **Custom Atrous-ECA** | **Scratch (None)** | **0.7891** | **0.8821** | **46.08 FPS** |
| **ResNet-18 Baseline** | Scratch (None) | 0.7563 | 0.8612 | 47.67 FPS |

**Conclusion:** When evaluated purely on architectural efficiency (learning from scratch), the Custom Atrous-ECA U-Net significantly outperforms the standard ResNet-18 baseline by **+3.28% mIoU**, proving its native capability to handle complex urban segmentation tasks at real-time speeds.

## ⚡ Architectural Benchmarks
*Evaluated on 512x512 RGB inputs.*
| Model | Parameters | Computational Complexity (MACs) |
| :--- | :--- | :--- |
| ResNet-18 Baseline | 14.33 M | 21.85 GMac |
| Custom Atrous-ECA | 9.28 M | 18.13 GMac |

*A few remarks:*
Here, I only used Atrous-ECA in the bottleneck and decoder parts of the network, the encoder is still a resnet-18.
Perhaps performance could be increased if Atrous-ECA blocks were involved in the encoder as well.

## 👁️ Visual Comparison

*(Observe how the Atrous-ECA model captures sharper building footprints and thinner road geometries compared to the smoothed-out Baseline models).*

[Access comparison images from the assets folder]

## 📁 Repository Structure

* `models/custom_unet.py`: Core architecture definitions for the Atrous-ECA network.
* `utils/`: Helper functions for metrics, dataset relabeling, and seeding.
* `chop_data.py`: Pre-processing script to tile high-resolution UAVid images into 512x512 patches safely.
* `train.py`: Main training loop with built-in validation checkpointing and dynamic routing.
* `eval.py`: Evaluation script for calculating precise mIoU, F1, and GPU latencies.
* `benchmark.py`: Profiler using `ptflops` to calculate MACs and Parameter counts.
* `visualize.py`: Generates the multi-column side-by-side comparison grids for visual inspection.

## Future scope and ideas
* Include AtrousECA blocks into the encoder as well
* Try a self-attention block in the bottleneck
* Figure out how to improve latency issues caused by ECA calculates that aren't GPU friendly

## 🚀 Quick Start

**1. Clone and Install**
```bash
git clone [https://github.com/akankhya-exe/drone-segmentation-uavid.git](https://github.com/akankhya-exe/drone-segmentation-uavid.git)
cd drone-segmentation-uavid
pip install -r requirements.txt