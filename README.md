# AnimalReID-CLEF2025

This repository contains the implementation of my baseline for the **AnimalCLEF 2025 – Multi-species Individual Animal Identification** challenge.  
The project introduces a **feature-fusion ensemble** combining **ResNet-18** and **EfficientNet-B0**, trained on the official dataset to perform robust, open-set animal re-identification across species.

🐾 **Summary:**  
- Built a modular PyTorch pipeline for feature extraction, fusion, and classification.  
- Handled long-tailed identity distributions through stratified splits and targeted augmentations.  
- Implemented confidence-based open-set recognition and stable training with AdamW, warmup scheduling, and early stopping.  
- Conducted qualitative retrieval analysis to evaluate embedding structure and robustness.  
- Developed as part of the *M.Sc. Artificial Intelligence* program at **FAU Erlangen-Nürnberg**, under **Prof. Vincent Christlein**.

🔗 **Code:** [https://github.com/usmanklodhi/AnimalReID-CLEF2025](https://github.com/usmanklodhi/AnimalReID-CLEF2025)
