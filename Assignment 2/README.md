# Project #2 — Perceptron Neural Network Handwriting Recognition

**Due:** September 24, 2025
**Author:** James Levi

---

## 1. Data Preparation (a–c)

The dataset **`digits.mat`** contains four variables:

| Name          | Size       | Class |
| ------------- | ---------- | ----- |
| `train`       | 784 × 5000 | uint8 |
| `trainlabels` | 5000 × 1   | uint8 |
| `test`        | 784 × 1000 | uint8 |
| `testlabels`  | 1000 × 1   | uint8 |

Each column of `train` represents a **flattened 28×28 image** of a handwritten digit.
To restore the image format, it is reshaped to `(28, 28, 5000)` using MATLAB-compatible order (`order='F'` in Python):

```python
x = train.reshape(28, 28, 5000, order='F')
```

This allows displaying the handwritten digits as grayscale images.

---

## 1(d) Detecting Digit ‘0’

A **single-layer 2D perceptron** was implemented to detect the digit **0**.
The perceptron outputs **+1** when the input digit is 0 and **−1** otherwise.

### Training Setup

* **Input:** 784 features per image, normalized to [0, 1].
* **Bias:** Added as a row of ones.
* **Weights:** Initialized to zeros.
* **Update Rule:**
  [
  w \leftarrow w + 0.5 \cdot \eta \cdot (d - y) \cdot x
  ]
* **Learning Rate:** Starts at 0.1 and decays each iteration as ( \eta = 0.999 \times \eta ).
* **Activation:** Sign function
  [
  y = \text{sgn}(w^T x)
  ]
* **Desired Output:**
  ( d = +1 ) for digit 0, otherwise ( -1 ).

### Monitoring

The model prints **accuracy, precision, and recall** after each epoch based on the **last 100 samples** of the training set.

### Results

* Achieved around **98% accuracy** for detecting 0.
* High **precision** and **recall** confirm it correctly identifies zeros instead of defaulting to predicting “not zero.”
* The reshaped weight vector ( w ) (28×28) visually resembles the outline of the digit “0,” reflecting where the model assigns high importance.

---

## 1(e) Detecting Digits ‘8’, ‘1’, and ‘2’

The same perceptron architecture was used to detect digits **8**, **1**, and **2**.

### Digit ‘8’

Performance decreased compared to digit 0.
Digits like 0, 3, 6, and 9 share similar rounded patterns, leading to **false positives** and lower precision.

### Digit ‘1’

Performed **very well**, achieving strong precision and recall.
Digit 1’s simple vertical structure makes it linearly separable and easy to detect.

### Digit ‘2’

Performance was **moderate**.
The perceptron occasionally confused 2’s with 3’s and 7’s due to similar curved edges.

---

## 2. Two-Layer Perceptron (Backpropagation)

### Objective

Extend the perceptron to a **two-layer neural network** with:

* **784 input neurons** (pixels)
* **25 hidden neurons**
* **10 output neurons** (for digits 0–9)

Each output neuron ( i ) should produce **1** when the digit is ( i ), and **0** otherwise.
Example target vector for digit 2:
[
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
]

### Algorithm

The network is trained using **backpropagation** with **sigmoid activations**.

#### 1. Forward Pass

[
v^{(1)} = W_1 x
]
[
y^{(1)} = \sigma(v^{(1)})
]
Add bias to the hidden output:
[
y^{(1)}*{\text{bias}} = [1; y^{(1)}]
]
[
v^{(2)} = W_2 y^{(1)}*{\text{bias}}
]
[
y^{(2)} = \sigma(v^{(2)})
]

#### 2. Error

[
e = d - y^{(2)}
]

#### 3. Deltas

[
\delta_2 = (d - y^{(2)}) \cdot y^{(2)}(1 - y^{(2)})
]
[
\delta_1 = (W_2^{T} \delta_2)_{\text{no bias}} \cdot y^{(1)}(1 - y^{(1)})
]

#### 4. Weight Updates

[
W_2 \leftarrow W_2 + \eta \cdot \delta_2 \cdot (y^{(1)}_{\text{bias}})^T
]
[
W_1 \leftarrow W_1 + \eta \cdot \delta_1 \cdot x^T
]

### Training

* **Hidden nodes:** 25
* **Output nodes:** 10
* **Epochs:** 10
* **Initial learning rate:** 0.2 (decayed gradually)
* **Shuffling:** Images are shuffled each epoch.

The model prints **epoch accuracy** after each training cycle.

### Testing

During testing, the forward pass is performed with the learned weights.
The predicted digit corresponds to the output neuron with the maximum activation.
Final accuracy is printed after evaluating all test samples.

---

## 3. Observations

* The **single-layer perceptron** works well for simple, linearly separable digits such as 0 and 1.
* The **two-layer network** can distinguish all digits by introducing nonlinearity through the hidden layer.
* Weight visualization helps reveal the areas of the image the network uses to identify each digit.
* Learning rate decay and shuffling improve stability and convergence during training.

---

**Authored by:**
**James Levi**
M.S. in Artificial Intelligence
Florida Atlantic University

---
