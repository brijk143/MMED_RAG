**About Dataset**
This is a clinical chest X-ray report dataset designed for medical text classification and radiology understanding tasks.
Label column represents the final disease class assigned to each chest X-ray report.

**Complete List of 24 Classes :-**
1-normal
2-Hyperinflation
3-Calcified Granuloma
4-Cardiomegaly
5-Consolidation
6-Aortic Aneurysm
7-Scoliosis
8-Atherosclerosis
9-Mass
10-Fibrosis / Pulmonary Fibrosis
11-Nodule
12-Pleural Effusion
13-Rib Fracture / Fractures
14-Hernia
15-Emphysema
16-Pleural Thickening
17-Pneumothorax
18-Edema
19-Pneumonia
20-Bullous Disease
21-Tuberculosis
22-Subcutaneous Emphysema
23-Bronchiectasis
24-Bronchiolitis



Class-wise Percentage Distribution:-

| Class                         | Count | Percentage |
| ----------------------------- | ----: | ---------: |
| normal                    |  1808 | **46.95%** |
| Hyperinflation                |   398 |     10.33% |
| Calcified Granuloma           |   377 |      9.79% |
| Cardiomegaly                  |   321 |      8.34% |
| Consolidation                 |   312 |      8.10% |
| Aortic Aneurysm               |   133 |      3.45% |
| Scoliosis                     |    92 |      2.39% |
| Atherosclerosis               |    65 |      1.69% |
| Mass                          |    56 |      1.45% |
| Fibrosis / Pulmonary Fibrosis |    56 |      1.45% |
| Nodule                        |    47 |      1.22% |
| Pleural Effusion              |    45 |      1.17% |
| Rib Fracture / Fractures      |    33 |      0.86% |
| Hernia                        |    31 |      0.80% |
| Emphysema                     |    22 |      0.57% |
| Pleural Thickening            |    15 |      0.39% |
| Pneumothorax                  |    14 |      0.36% |
| Edema                         |    11 |      0.29% |
| Pneumonia                     |     4 |      0.10% |
| Bullous Disease               |     4 |      0.10% |
| Tuberculosis                  |     4 |      0.10% |
| Subcutaneous Emphysema        |     1 |      0.03% |
| Bronchiectasis                |     1 |      0.03% |
| Bronchiolitis                 |     1 |      0.03% |





<p align="center">
  <img src="class.png" width="500"/>
</p>


X-axis → Disease classes ,
Y-axis → Percentage of samples ,
Clear long-tail distribution ,
Many rare classes have <1% representation


**In this dataset Each row represents one patient study and contains:**

uid → Unique identifier for each report ,
image → Type of imaging study (e.g., Chest X-ray PA/Lateral) ,
indication → Clinical reason for the scan ,
comparison → Reference to prior scans (if available) ,
findings → Detailed radiologist observations ,
impression → Final diagnostic summary ,
MeSH → Structured medical terms (disease, location, severity) ,
label → Final disease class used for model training 

