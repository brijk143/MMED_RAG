# Clinical Chest X-ray Report Dataset

## Overview

This dataset is a **clinical chest X-ray radiology report corpus** designed for **medical text classification**, **radiology report understanding**, and **clinical NLP research**.  
Each entry corresponds to a **single patient study**, containing both structured and unstructured clinical text extracted from chest X-ray reports.

The `label` column represents the **final disease class** assigned to each report and serves as the ground-truth target for supervised learning tasks.

---

## Dataset Characteristics

- **Modality:** Chest X-ray (PA / Lateral)
- **Data Type:** Clinical radiology text
- **Task Type:** Multi-class classification (single-label)
- **Total Classes:** 24
- **Domain:** Medical imaging, radiology, pulmonary and cardiovascular diseases

---

## Disease Class Labels (24 Classes)

1. Normal  
2. Hyperinflation  
3. Calcified Granuloma  
4. Cardiomegaly  
5. Consolidation  
6. Aortic Aneurysm  
7. Scoliosis  
8. Atherosclerosis  
9. Mass  
10. Fibrosis / Pulmonary Fibrosis  
11. Nodule  
12. Pleural Effusion  
13. Rib Fracture / Fractures  
14. Hernia  
15. Emphysema  
16. Pleural Thickening  
17. Pneumothorax  
18. Edema  
19. Pneumonia  
20. Bullous Disease  
21. Tuberculosis  
22. Subcutaneous Emphysema  
23. Bronchiectasis  
24. Bronchiolitis  

---

## Class-wise Distribution

The dataset shows a **class imbalance**, with a small number of frequent classes and many rare disease categories.

| Class                         | Count | Percentage |
| ----------------------------- | ----: | ---------: |
| Normal                        | 1808 | **46.95%** |
| Hyperinflation                |  398 | 10.33% |
| Calcified Granuloma           |  377 | 9.79% |
| Cardiomegaly                  |  321 | 8.34% |
| Consolidation                 |  312 | 8.10% |
| Aortic Aneurysm               |  133 | 3.45% |
| Scoliosis                     |   92 | 2.39% |
| Atherosclerosis               |   65 | 1.69% |
| Mass                          |   56 | 1.45% |
| Fibrosis / Pulmonary Fibrosis |   56 | 1.45% |
| Nodule                        |   47 | 1.22% |
| Pleural Effusion              |   45 | 1.17% |
| Rib Fracture / Fractures      |   33 | 0.86% |
| Hernia                        |   31 | 0.80% |
| Emphysema                     |   22 | 0.57% |
| Pleural Thickening            |   15 | 0.39% |
| Pneumothorax                  |   14 | 0.36% |
| Edema                         |   11 | 0.29% |
| Pneumonia                     |    4 | 0.10% |
| Bullous Disease               |    4 | 0.10% |
| Tuberculosis                  |    4 | 0.10% |
| Subcutaneous Emphysema        |    1 | 0.03% |
| Bronchiectasis                |    1 | 0.03% |
| Bronchiolitis                 |    1 | 0.03% |

<p align="center">
  <img src="class.png" width="500"/>
</p>

**Observations:**
- Clear **long-tail distribution**
- Several clinically significant conditions have **<1% representation**
- Suitable for research on **imbalanced learning**, **few-shot learning**, and **robust medical NLP models**

---

## Data Schema

Each row represents **one patient study** and contains the following fields:

| Column | Description |
|------|-------------|
| `uid` | Unique identifier for each radiology report |
| `image` | Type of imaging study (e.g., Chest X-ray PA / Lateral) |
| `indication` | Clinical reason for the scan |
| `comparison` | Reference to prior imaging studies (if available) |
| `findings` | Detailed radiologist observations |
| `impression` | Final diagnostic summary |
| `MeSH` | Structured medical terms (disease, location, severity) |
| `label` | Final disease class used for model training |

---

## Intended Use Cases

- Medical text classification  
- Radiology report understanding and summarization  
- Clinical NLP and language modeling  
- Disease prediction from radiology narratives  
- Medical concept extraction using MeSH terminology   

---

## Ethical Considerations

This dataset is intended **strictly for research and educational purposes**.  
It must **not** be used for clinical decision-making or diagnostic deployment without appropriate validation and regulatory approval.
