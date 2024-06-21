# Rational Kolmogorov-Arnold Network (rKAN)

Rational Kolmogorov-Arnold Network (rKAN) is a novel neural network that incorporates the distinctive attributes of Kolmogorov-Arnold Networks (KANs) with a trainable adaptive rational-orthogonal Jacobi function as its basis function. This method offers several advantages, including non-polynomial behavior, activity for both positive and negative input values, faster execution, and better accuracy.

## Installation

To install rKAN, use the following command:

```bash
$ pip install rkan
```

## Example Usage

The current implementation of rKAN works with both the TensorFlow and PyTorch APIs.

### TensorFlow

```python
from tensorflow import keras
from tensorflow.keras import layers
from rkan.tensorflow import JacobiRKAN, PadeRKAN

model = keras.Sequential(
    [
        layers.InputLayer(input_shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3)),
        JacobiRKAN(3),      # Jacobi polynomial of degree 3
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(16),
        PadeRKAN(2, 6),     # Pade [2/6]
        layers.Dense(num_classes, activation="softmax"),
    ]
)

```

### PyTorch

```python
import torch.nn as nn
from rkan.torch import JacobiRKAN, PadeRKAN

model = nn.Sequential(
    nn.Linear(1, 16),
    JacobiRKAN(3),      # Jacobi polynomial of degree 3
    nn.Linear(16, 32),
    PadeRKAN(2, 6),     # Pade [2/6]
    nn.Linear(32, 1),
)

```

## Experiments

The `example` folder contains the implementation of the experiments from the paper using rKAN. These experiments include:

### Deep Learning Tasks

- **Synthetic Regression**
- **MNIST Classification**

### Physics Informed Deep Learning

- **Lane Emden Ordinary Differential Equation**
- **Elliptic Partial Differential Equation**


## Current Limitations

- Maximum allowed Jacobi polynomial degree is set to six.
- The current library is not compatible with other deep learning frameworks, but it can be converted easily.

## Contribution

We encourage the community to contribute by opening issues and submitting pull requests to help address these limitations and improve the overall functionality of rKAN.

## Contact

If you have any questions or encounter any issues, please open an issue in this repository (preferred) or reach out to the author directly.

## Citation

If you use rKAN in your research, please cite our paper:

```
@article{aghaei2024rkan,
  title={rKAN: Rational Kolmogorov-Arnold Networks},
  author={Aghaei, Alireza Afzal},
  journal={arXiv preprint arXiv:2406.14495},
  year={2024}
}
```
