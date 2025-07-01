# 🚗 Car Damage Detection with Grad-CAM & Streamlit

An AI-powered deep learning project that detects car damage types using a fine-tuned ResNet18 model. Built with PyTorch and Streamlit, with visual explanation using Grad-CAM.

---

## 🧠 Model
- Architecture: **ResNet18**
- Classes Detected:
  - `F_Breakage`, `F_Crushed`, `F_Normal`
  - `R_Breakage`, `R_Crushed`, `R_Normal`
- Accuracy: ~73% on test data
- Explainability: **Grad-CAM** heatmaps

---

## 🖥️ Streamlit App
- Upload a car image
- Get prediction with **class label**
- Visualize heatmap highlighting damaged area
- Option to download report

Run with:
```bash
streamlit run app.py
