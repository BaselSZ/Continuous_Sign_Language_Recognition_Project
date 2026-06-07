# Continuous Sign Language Recognition (ICS471)

A deep learning project that recognizes continuous sign language sequences using human pose keypoints from the Isharah dataset.

## Dataset

* **Training Samples:** 9,500
* **Development Samples:** 949
* **Vocabulary Size:** 684 glosses
* **Input:** Pose keypoints `(T, 86, 2)`

## Models

### BiGRU + CTC

* 3-layer Bidirectional GRU
* Hidden Size: 512
* CTC Loss

**Dev WER:** `0.9014`

### Transformer + CTC

* 6 Transformer Encoder Layers
* 6 Attention Heads
* Hidden Size: 384
* Positional Encoding
* CTC Loss

**Dev WER:** `0.4473`

## Results

| Model             |    Dev WER |
| ----------------- | ---------: |
| BiGRU + CTC       |     0.9014 |
| Transformer + CTC | **0.4473** |

The Transformer model achieved a **50% reduction in Word Error Rate (WER)** compared to the BiGRU baseline, demonstrating superior performance for long-range temporal modeling in sign language recognition.

## Tech Stack

* Python
* PyTorch
* NumPy
* Pandas
* CUDA
