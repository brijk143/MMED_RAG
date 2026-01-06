# Knowledge Graph–Driven Medical Reasoning for Chest X-ray Findings

## 📌 Problem Statement

Deep learning models for chest X-ray analysis can accurately detect **radiological findings** such as consolidation, pleural effusion, or hyperinflation.  
However, they fail at **medical reasoning**:

- A single finding may correspond to **multiple diseases**
- Models often **over-predict diagnoses**
- There is no explicit handling of **uncertainty, absence, or contradiction**
- Outputs lack **clinical explainability**

### Objective
To design a **Knowledge Graph (KG)–based reasoning layer** that converts **model-predicted findings** into **ranked disease hypotheses**, mimicking **clinical differential diagnosis**, without directly predicting diseases.

---

## 🧠 Core Principle

> **Findings activate edges.  
> Relations decide plausibility.  
> Diseases are ranked, not predicted.**

This principle ensures:
- Medical safety
- Interpretability
- Reduced hallucination
- Explicit uncertainty handling

---

## 🏗️ System Overview

graph LR
    A[📷 Chest X-ray] --> B[🔬 BioMedClip]
    B --> C[📋 Predicted Findings]
    C --> D[🧠 Knowledge Graph]
    D --> E[🏆 Ranked Diagnoses]



---

## 🧪 Step 1: Model Training (Perception Layer)

### Goal
Train a model to **detect radiological findings**, not diseases.

### Input
- Chest X-ray images

### Output
One or more predefined classes such as:
- Airspace Disease
- Bronchiectasis
- Bronchiolitis
- Bullous Disease
- Calcified Granuloma
- Calcinosis
- Calcinosis Cutis
- Cardiomegaly
- Cardiac Shadow (abnormal)
- Consolidation
- Degenerative Change
- Edema
- Effusion
- Emphysema
- Fibrosis
- Fractures, Bone
- Hernia
- Hyperinflation
- Increased Lung Markings
- Interstitial Lung Disease
- Kyphosis
- Lesion
- Mass
- Nodule
- Osteophyte
- Pleural Effusion
- Pleural Thickening
- Pneumothorax
- Pulmonary Artery Enlargement
- Pulmonary Fibrosis
- Rib Fracture
- Scoliosis
- Subcutaneous Emphysema
- Thickening
- Volume Loss
- normal


### Important Rule
⚠️ Model outputs are treated as **observations**, not diagnoses.

Here's the example output in Markdown format:

```json
{
  "predicted_class": "Consolidation",
  "confidence": 0.84
}
```


# Step 2: Knowledge Graph Design

The Knowledge Graph is **pre-constructed and static**.

## 2.1 Node Types

| Node Type | Description |
|-----------|-------------|
| **Finding** | Observable radiological patterns |
| **Disease** | Latent clinical conditions |
| **Normal** | Explicit absence of pathology |
| **Symptom** | Patient-reported evidence |
| **Anatomy** | Location context |

## Design Philosophy

The KG is **finding-centric**, reflecting how clinicians reason.

# Step 3: Core Relations (Reasoning Rules)

These relations encode medical logic and are used during inference.

| Relation            | Meaning                     |
| ------------------- | --------------------------- |
| `has_finding`       | Disease → Finding           |
| `has_symptom`       | Disease → Symptom           |
| `suggests`          | Finding/Symptom → Disease   |
| `strongly_suggests` | High diagnostic value       |
| `weakly_suggests`   | Needs more evidence         |
| `contradicts`       | Makes disease unlikely      |
| `requires`          | Disease → Mandatory finding |
| `confirmed_by`      | Disease → Test              |
| `located_in`        | Finding → Anatomy           |

# Step 4: Knowledge Graph Construction

Example edges:

- Consolidation ─strongly_suggests→ Pneumonia
- Consolidation ─weakly_suggests→ Tuberculosis
- Calcified Granuloma ─strongly_suggests→ Tuberculosis
- Hyperinflation ─strongly_suggests→ Bronchiolitis
- Hyperinflation ─suggests→ Emphysema
- Pneumothorax ─strongly_suggests→ Bullous Disease
- Normal ─contradicts→ Pneumonia
- Normal ─contradicts→ Tuberculosis

The KG itself never changes at inference time.

## 🔗 Core Relationships (Extracted Knowledge)

These are the **only relationships required** for finding-driven medical reasoning.

---

### 1. `strongly_suggests`
**Finding → Disease**

Meaning:
- High diagnostic specificity
- Strong medical evidence

Example:
```
Calcified Granuloma ─strongly_suggests→ Tuberculosis
Pneumothorax ─strongly_suggests→ Bullous Disease
```

---

### 2. `suggests`
**Finding → Disease**

Meaning:
- Moderate support
- Common but non-specific

Example:
```
Consolidation ─suggests→ Pneumonia
Pleural Effusion ─suggests→ Edema
```

---

### 3. `weakly_suggests`
**Finding → Disease**

Meaning:
- Insufficient alone
- Requires additional evidence

Example:
```
Mass ─weakly_suggests→ Tuberculosis
Fibrosis ─weakly_suggests→ Bronchiectasis
```

---

### 4. `contradicts`
**Finding / Normal → Disease**

Meaning:
- Makes disease unlikely
- Negative evidence

Example:
```
Normal ─contradicts→ Pneumonia
Normal ─contradicts→ Tuberculosis
```

---

### 5. `requires`
**Disease → Finding**

Meaning:
- Mandatory evidence for confident diagnosis

Example:
```
Tuberculosis ─requires→ Calcified Granuloma
Pulmonary Edema ─requires→ Cardiomegaly
```

---

### 6. `absence_weakens`
**Absent Finding → Disease**

Meaning:
- Absence lowers disease plausibility

Example:
```
No_Cavitation ─absence_weakens→ Tuberculosis
No_Effusion ─absence_weakens→ Edema
```

---

## 🧬 Relationship Representation

Relationships are stored as **directed edges**.

### JSON Representation

```json
{
  "Consolidation": {
    "strongly_suggests": ["Pneumonia"],
    "weakly_suggests": ["Tuberculosis"]
  },
  "Calcified Granuloma": {
    "strongly_suggests": ["Tuberculosis"]
  },
  "Normal": {
    "contradicts": ["Pneumonia", "Tuberculosis"]
  }
}
```
