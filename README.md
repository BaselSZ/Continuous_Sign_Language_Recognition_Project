Continuous Sign Language Recognition using Deep Learning
Overview

This project implements and compares two Continuous Sign Language Recognition (CSLR) models on the Isharah dataset using pose keypoints extracted from videos.

Dataset
9,500 training samples
949 development samples
684 gloss vocabulary
Input shape: (T, 86, 2) pose keypoints
Models
1. BiGRU + CTC (Baseline)
3-layer Bidirectional GRU
Hidden size: 512
CTC Loss for sequence alignment

Result: Dev WER = 0.9014

2. Transformer + CTC
6 Transformer encoder layers
Hidden size: 384
6 attention heads
Sinusoidal positional encoding
CTC Loss for alignment

Result: Dev WER = 0.4473

Results
Model	Dev WER
BiGRU + CTC	0.9014
Transformer + CTC	0.4473

The Transformer achieved approximately 50% lower WER than the BiGRU baseline, demonstrating the effectiveness of self-attention for modeling long-range temporal dependencies in sign language sequences.

Technologies
Python
PyTorch
NumPy
Pandas
CUDA
