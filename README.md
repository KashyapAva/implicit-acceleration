# 📈 Implicit Acceleration in Deep Networks via Overparameterization

This research investigates how increasing the **depth** of linear neural networks can **accelerate optimization** — not by making models more expressive, but by acting as a form of **implicit preconditioning** for gradient descent.

## 📌 Project Summary

- **Goal**: Analyze how depth alone (without changing representational power) improves convergence in deep networks.
- **Key Insight**: Overparameterization in linear networks transforms gradient updates in a way that mimics momentum and adaptive learning rate behavior.
- **Method**: Formal derivation of update rules and proof of implicit acceleration using differential equations, Kronecker products, and SVD.

## 🧪 Core Topics Explored

- Gradient dynamics in deep linear networks
- Overparameterization vs. expressiveness
- Implicit vs. explicit regularization
- Acceleration in \( \ell_p \) regression for \( p > 2 \)
- Preconditioning matrices derived from network singular values

## 🔬 Experiments

- Empirical validation using UCI Gas Sensor dataset (scalar regression)
- Compared optimization speed under:
  - \( \ell_2 \) and \( \ell_4 \) loss
  - Standard gradient descent vs. derived update rule
  - Overparameterization vs. adaptive optimizers (AdaGrad, AdaDelta, Adam)
- Extension to CNNs on MNIST: replacing dense layers with deeper matrix products accelerated convergence

## 📊 Key Findings

| Scenario | Depth Helps? | Notes |
|----------|--------------|-------|
| \( \ell_2 \) loss | ❌ | No significant acceleration |
| \( \ell_4 \) loss | ✅ | Strong acceleration effect |
| Wide vs. Narrow | ⚖️ | Width has negligible effect — depth is key |
| Identity Init | ✅ | Improved convergence with linear residuals |

## 📘 Theoretical Contributions

- Derived the modified gradient update as:

  \[
  \vec{W}_e^{(t+1)} = (1 - \eta \lambda N) \cdot \vec{W}_e^{(t)} - \eta \cdot P_{W_e^{(t)}} \cdot \nabla L_1(W_e^{(t)})
  \]

  where \( P_{W_e} \) acts as a preconditioner that aligns updates along directions with stronger singular values.

- Proved that **no regularizer gradient** can replicate this implicit acceleration — showing it is fundamentally distinct from traditional regularization.

## 🛠 Tools Used
- TensorFlow (experiments)
- Python + NumPy (symbolic gradient tracing)
- Mathematical analysis using differential equations and linear algebra

## 📁 Files
- `STAT525_Report_vava2.pdf`: Full report with theoretical derivations and experimental results

## 👤 Author
- Kashyap Ava

---

*Project completed for STAT 525: Optimization Methods in Machine Learning at UIUC.*
