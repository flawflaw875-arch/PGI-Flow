# PG-Flow Surrogates
Predicting Object Detection Transferability via Parameter-Gated Flow Surrogates

## 1. Research Background and Problem Definition

Object detection models in construction sites experience significant performance degradation when deployed in new sites (Target Sites) after being trained on a specific site (Source Site) due to differences in lighting, camera angles, equipment types, and worker attire. Re-training the model for each new site or testing all models individually incurs substantial time and cost.

This project aims to address this issue by developing a system that predicts the transfer performance of a trained model using only the parameter information of the model and the visual features of the target site, without actual deployment or re-training.

## 2. Key Ideas and Methodology

This research proposes the following key ideas:

1.  **'Parameter-Based Flow Surrogate' (`s`)**:
    *   **Overcoming Limitations of Existing Research:** Previous studies primarily analyzed the architecture's structure or used only the final output/single-layer features to predict transfer performance. This often failed to reflect how the model was actually trained (parameter information), leading to the underestimation of the potential of 'shallow expert' models.
    *   **Differentiation of This Study:** We integrate the statistical characteristics of the trained model's parameters (e.g., weight norms, variance) into the information flow simulation. This generates a unique **'fingerprint' (`s`)** formed by the results of training on a specific dataset, representing not only the model's structure but also the intrinsic characteristics and learning experiences of the model.

2.  **Target Dataset Features (`f`)**:
    *   Visual features are extracted from unlabeled images of the new target site to create a **feature vector (`f`)** that represents the dataset of that site. This process identifies the characteristics of the site without labeling costs.

3.  **Learning the 'Discriminator'(`P`)**:
    *   `P` is a regression model that takes the fingerprint of the source model `s` and the features of the target dataset `f` as input to predict the expected transfer performance of the source model in the target site.
    *   **Supporting Strategic Decision-Making:** `P` goes beyond merely predicting performance; it provides strategic insights into which type, single-site expert model or multi-site generalist model, is more suitable for the new site. of model
