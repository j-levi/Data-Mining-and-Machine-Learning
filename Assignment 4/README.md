# Mini Project #4 — Fast Independent Component Analysis (Fast-ICA)

**Author:** James Levi  
**Date:** October 22, 2025  

---

## 1. Overview

This project applies the **Fast Independent Component Analysis (Fast-ICA)** algorithm to separate two mixed grayscale images of a biological specimen. Both images were distorted by the same random pattern, and the goal is to recover the original, independent source images (the specimen and the noise).

Independent Component Analysis (ICA) is a method used in signal processing to separate a multivariate signal into additive subcomponents that are statistically independent.

---

## 2. Methodology

### 2.1 Image Preparation

Each image (`Recording-1.png` and `Recording-2.png`) is converted to grayscale and flattened into a one-dimensional vector of pixel values. The two vectors are stacked into a **2×N matrix**:

\[
X = 
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
\]

Each row represents one observed mixed signal.  
Before processing, each row is **centered** by subtracting its mean:

\[
X_{center} = X - \text{mean}(X, \text{axis}=1)
\]

This ensures zero-mean data, which is a requirement for ICA.

---

### 2.2 Whitening (Decorrelation)

Whitening removes correlations between the two recordings, making the covariance matrix an identity matrix.  

1. Compute the covariance matrix:

\[
C = \frac{1}{N} X_{center} X_{center}^T
\]

2. Perform **eigenvalue decomposition**:

\[
C = Q \Lambda Q^T
\]

where:
- \( Q \) is the matrix of eigenvectors,
- \( \Lambda \) is the diagonal matrix of eigenvalues.

3. Apply whitening transformation:

\[
X_{whitened} = Q \Lambda^{-1/2} Q^T X_{center}
\]

After whitening, the two recordings are uncorrelated and have unit variance, which simplifies the ICA process.

---

### 2.3 Fast-ICA Algorithm

The Fast-ICA algorithm iteratively estimates the **unmixing matrix** \( W \), which transforms the whitened signals into independent components:

\[
S = W X_{whitened}
\]

where \( S \) contains the separated sources.

#### Steps:
1. Initialize \( W \) randomly and normalize its rows.
2. Update using:
   \[
   w_{new} = E\{x g(w^T x)\} - E\{g'(w^T x)\} w
   \]
   where:
   \[
   g(u) = \tanh(u), \quad g'(u) = 1 - \tanh^2(u)
   \]
3. Apply **symmetric decorrelation** to keep all components orthogonal:
   \[
   W = (V \Lambda^{-1/2} V^T) W
   \]
   where \( V, \Lambda \) are obtained from eigenvalue decomposition of \( W W^T \).
4. Normalize each row of \( W \).
5. Stop when the direction of \( W \) changes less than a given tolerance.

---

### 2.4 Reconstructing the Images

After obtaining separated components \( S_1 \) and \( S_2 \), they are rescaled to the 0–255 range and reshaped into the original image dimensions to produce the final separated images.

---

## 3. Mathematical Summary

| Symbol | Meaning |
|---------|----------|
| \( X \) | Observed mixed signals (2×N) |
| \( C \) | Covariance matrix |
| \( Q, \Lambda \) | Eigenvectors and eigenvalues of \( C \) |
| \( X_{whitened} \) | Decorrelated and normalized data |
| \( W \) | Unmixing matrix learned by Fast-ICA |
| \( S = W X_{whitened} \) | Estimated independent sources |

---

## 4. Code Structure

| Function | Description |
|-----------|--------------|
| `convert_image_to_vector()` | Converts grayscale image to 1D vector. |
| `convert_vector_to_image()` | Converts 1D vector back to grayscale image. |
| `Fast_ICA()` | Implements the Fast-ICA algorithm with tanh nonlinearity and symmetric decorrelation. |
| `to_uint8()` | Normalizes output to displayable image format. |

---

## 5. Dependencies

```bash
pip install numpy pillow
````

---

## 6. Results

Running the script produces two separated images:

* One corresponding to the **biological specimen**
* The other corresponding to the **noise/distortion**

These outputs visually demonstrate how ICA successfully isolates independent sources from mixed signals.


