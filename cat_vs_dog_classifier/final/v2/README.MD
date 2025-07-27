## 📁 File Structure

- 🔧 **Model Weights**: Stored in `./model_weights/{model_name}.pt`
- 🧠 **Training Scripts**: Available in notebooks like `./model{i}.ipynb`, which include class-wise accuracy, loss, and training/validation accuracy curves for each model.
  - Example: `model1_0` and `model1_1` share the same architecture but are trained independently (different batch size or augmentations)

---

## 🏋️ Training Details

- **Epochs**: 10 total
- **Optimizer**: AdamW (`Adam2`)
- **Learning Rate Schedule**:
  - Epochs 1–6: `0.001`
  - Epochs 7–9: `0.0001`
  - Remaining epochs: `0.00001`
- **Batch Size**: `40` (two models were trained with `16`)
- **Input Image Size**: `128 × 128`
- **Augmentation**: Horizontal flip — every image has a flipped version, doubling the training set

---

## 🛠️ Preprocessing & Augmentation

All preprocessing steps (resizing, flipping, dataset split, etc.) are handled in:  
`./preprocess.ipynb`

## 📦 Model Summary

| Model Name | Parameter count |  MMAC  | Accuracy on train data | Accuracy on test data | Loss on train data | Loss on test data |
|------------|-----------------|--------|------------------------|-----------------------|--------------------|-------------------|
|  model0_0  |      8.25k      | 30.52  |         87.34%         |        85.48%         |        0.2915      |       0.3345      |
|  model0_1  |      8.25k      | 30.52  |         88.67%         |        86.56%         |        0.2629      |       0.3100      |
|  model1_0  |      8.05k      | 11.71  |         82.67%         |        79.93%         |        0.3867      |       0.4280      |
|  model2_0  |      4.79k      |  9.95  |         83.42%         |        81.37%         |        0.3755      |       0.4020      |
|  model3_0  |     12.20k      | 14.24  |         85.60%         |        81.99%         |        0.3282      |       0.4027      |
|  model4_0  |      3.15k      |  9.25  |         78.70%         |        77.38%         |        0.4576      |       0.4767      |
|  model5_0  |      4.91k      | 26.66  |         84.87%         |        83.68%         |        0.3450      |       0.3695      |
|  model6_0  |     12.48k      | 35.30  |         87.68%         |        85.43%         |        0.2864      |       0.3294      |
|  model7_0  |      8.05k      | 29.25  |         85.72%         |        83.88%         |        0.3293      |       0.3593      |
|  model8_0  |      4.79k      | 25.62  |         84.17%         |        82.77%         |        0.3575      |       0.3842      |
|  model9_0  |     12.20k      | 33.81  |         88.55%         |        85.74%         |        0.2692      |       0.3251      |
|  model10_0 |      4.79k      | 25.62  |         82.11%         |        78.81%         |        0.3936      |       0.4331      |
|  model11_0 |      8.05k      | 29.25  |         85.17%         |        83.14%         |        0.3407      |       0.3803      |
|  model12_0 |     12.20k      | 33.81  |         87.82%         |        84.59%         |        0.2877      |       0.3498      |
|  model13_0 |      4.91k      | 26.66  |         83.02%         |        81.48%         |        0.3780      |       0.4072      |
|  model14_0 |      8.25k      | 30.52  |         87.46%         |        84.61%         |        0.2945      |       0.3543      |
|  model15_0 |     12.48k      | 35.30  |         89.65%         |        86.87%         |        0.2470      |       0.3078      |
|  model15_1 |     12.48k      | 35.30  |         89.69%         |        86.06%         |        0.2479      |       0.3213      |
|  model16_0 |      4.91k      | 25.83  |         84.19%         |        82.91%         |        0.3545      |       0.3810      |
|  model17_0 |      8.25k      | 29.51  |         86.28%         |        84.27%         |        0.3178      |       0.3582      |
|  model18_0 |     12.48k      | 34.11  |         88.46%         |        85.87%         |        0.2697      |       0.3230      |
|  model19_0 |      4.91k      | 25.83  |         84.40%         |        82.07%         |        0.3538      |       0.4041      |
|  model20_0 |      8.25k      | 29.51  |         85.55%         |        82.89%         |        0.3311      |       0.3841      |
|  model21_0 |     12.48k      | 34.11  |         88.53%         |        85.21%         |        0.2692      |       0.3397      |
|  model22_0 |      3.50k      | 25.07  |         84.54%         |        83.37%         |        0.3522      |       0.3772      |
|  model23_0 |      3.40k      | 24.14  |         83.20%         |        82.12%         |        0.3760      |       0.3941      |
|  model24_0 |      3.40k      | 24.14  |         82.99%         |        80.78%         |        0.3824      |       0.4108      |
|  model25_0 |      3.50k      | 25.07  |         82.71%         |        80.73%         |        0.3891      |       0.4189      |
|  model26_0 |      3.50k      | 24.32  |         83.77%         |        82.19%         |        0.3693      |       0.3931      |
|  model27_0 |      3.50k      | 24.32  |         82.62%         |        80.50%         |        0.3881      |       0.4258      |
