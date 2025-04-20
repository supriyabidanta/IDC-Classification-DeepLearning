# Breast Cancer IDC Classification using Deep Learning

This project implements a hybrid deep learning pipeline for classifying Invasive Ductal Carcinoma (IDC) from histopathological breast cancer images. It integrates three powerful approaches:

- **Convolutional Neural Networks (CNNs)** for feature learning
- **Singular Value Decomposition (SVD)** for dimensionality reduction
- **Deep Convolutional GANs (DCGANs)** for synthetic image generation to address data imbalance

---

## 📁 Project Structure

```
├── BreastCancerDetection-64.ipynb         # Main notebook: full pipeline
├── BreastCancerDetection-Copy1.ipynb      # Secondary/variant version
├── Images/                                # Raw IDC histopathology images
```

---

## 📊 Methods Summary

- **Dataset**: ~277,000 64x64 image patches, binary-labeled (IDC+ / IDC−)
- **Image Compression**: Per-channel SVD with top-k singular values
- **GAN Augmentation**: DCGAN to generate synthetic IDC+ samples
- **Models**:
  - ResNet50 (transfer learning)
  - Custom lightweight CNN
  - Hybrid CNN (dual input: original + SVD images)
- **Metrics**: Accuracy, Precision, Recall, F1, AUC

---

## 🧪 Results Highlights

| Model           | AUC   | Notes                                  |
|----------------|-------|----------------------------------------|
| ResNet50        | 0.58  | Overfitted on imbalanced data          |
| Custom CNN      | 0.95  | Effective with GAN-augmented data      |
| Hybrid CNN      | 0.95  | Best performance: dual-input strategy  |

---

## 🔧 Requirements

- Python 3.8+
- TensorFlow or PyTorch
- NumPy, OpenCV, Matplotlib
- scikit-learn, seaborn
- tqdm

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/your-username/IDC-Classification-DeepLearning.git

# Install dependencies
pip install -r requirements.txt

# Run the Jupyter notebook
jupyter notebook BreastCancerDetection-64.ipynb
```

---

## 📌 Future Directions

- StyleGAN-based augmentation for higher-fidelity images
- Adaptive SVD compression
- Hyperparameter tuning & cross-validation
- Clinical data validation for real-world deployment

---

## 📬 Contact

**Supriya Bidanta**  
Ph.D. Student, Bioengineering, Indiana University  
Email: sbidanta@iu.edu  
LinkedIn: [linkedin.com/in/supriya-bidanta](https://www.linkedin.com/in/supriya-bidanta/)
