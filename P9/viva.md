

# 🚕 **Uber Ride Fare Prediction using PCA & Linear Regression**

### 📘 **Objective**

To predict the **fare amount** for Uber rides using various trip and passenger details.
We compare two models:

1. **Linear Regression without PCA**
2. **Linear Regression with PCA (Principal Component Analysis)**

The goal is to understand the effect of **dimensionality reduction** (PCA) on model performance.

---

## 🧠 **Concepts Covered**

### 1️⃣ Linear Regression

Linear Regression is a **supervised machine learning algorithm** used for predicting a continuous target variable.

**Equation:**
[
y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \ldots + \beta_nx_n + \epsilon
]
Where:

* ( y ) = predicted fare amount
* ( x_1, x_2, ... x_n ) = input features (longitude, latitude, etc.)
* ( \beta_i ) = coefficients (weights learned by model)
* ( \epsilon ) = error term

**Goal:** Minimize the Mean Squared Error (MSE)
[
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2
]

---

### 2️⃣ PCA (Principal Component Analysis)

PCA is a **dimensionality reduction** technique that transforms features into a smaller set of **principal components** while retaining maximum variance.

**Steps in PCA:**

1. Standardize the data
2. Compute covariance matrix
3. Find eigenvalues & eigenvectors
4. Choose top k eigenvectors with highest eigenvalues
5. Form new reduced dataset

**Why PCA?**

* Removes correlated (redundant) features
* Reduces model complexity
* Improves computation efficiency

**Explained Variance Ratio:**
Tells how much information (variance) each principal component retains.

---

### 3️⃣ Feature Engineering – Haversine Distance

Since fare depends heavily on **trip distance**, we calculate the distance using latitude & longitude.

**Formula:**
[
d = 2r \times \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta lat}{2}\right) + \cos(lat1) \cos(lat2) \sin^2\left(\frac{\Delta lon}{2}\right)}\right)
]
where

* ( r = 6371 ) km (Earth’s radius)
* ( lat ) and ( lon ) are in radians

---

### 4️⃣ Data Preprocessing Steps

1. **Remove null & invalid entries**
2. **Convert datetime** → extract `hour`, `day`, `month`, `year`
3. **Feature Scaling (StandardScaler)**

   * Formula:
     [
     z = \frac{x - \mu}{\sigma}
     ]
4. **Feature Selection:**

   * pickup/dropoff coordinates
   * passenger count
   * hour, day, month, year
   * distance_km

---

### 5️⃣ Model Evaluation Metrics

1. **R² Score (Coefficient of Determination):**
   [
   R^2 = 1 - \frac{\sum(y_i - \hat{y_i})^2}{\sum(y_i - \bar{y})^2}
   ]
   Measures how well the model fits the data.
   Higher R² → better performance.

2. **MSE (Mean Squared Error):**
   Measures average squared difference between actual and predicted values.

3. **RMSE (Root Mean Squared Error):**
   [
   RMSE = \sqrt{MSE}
   ]
   Gives error in same units as target variable (fare).

---

## 🧩 **Workflow**

### Step 1: Load and Clean Data

* Removed missing and invalid rows
* Extracted date-time features
* Dropped unnecessary columns (`Unnamed: 0`, `key`)

### Step 2: Feature Engineering

* Calculated **Haversine distance**
* Added **hour**, **day**, **month**, **year**

### Step 3: Split Dataset

* 80% → Training
* 20% → Testing

### Step 4: Scale the Data

* Used **StandardScaler** to normalize features

### Step 5: Train Two Models

* **Model 1:** Linear Regression (no PCA)
* **Model 2:** Linear Regression (with PCA, 95% variance retained)

### Step 6: Evaluate and Compare

| Model       | R² Score | RMSE            | Comment            |
| ----------- | -------- | --------------- | ------------------ |
| Without PCA | ~0.70    | Lower           | More accurate      |
| With PCA    | ~0.68    | Slightly higher | Faster computation |

---

## 📊 **Visualization**

* **PCA Variance Plot:** Shows how many components explain 95% variance.
* **Feature vs Fare Correlation:** Shows `distance_km` has highest correlation with fare.

---

## 💬 **Common Viva Questions**

### 🔹 Basic Understanding

1. What is the main goal of this project?
2. Why did you choose Linear Regression?
3. What does PCA do?
4. What is the purpose of StandardScaler?

### 🔹 Mathematical / Technical

5. Write the formula for Linear Regression.
6. What is the difference between MSE and RMSE?
7. How do you calculate Haversine distance?
8. What does “explained variance ratio” mean in PCA?

### 🔹 Interpretation

9. Why did R² decrease slightly after PCA?
   → Because PCA compresses some information, slightly reducing accuracy.
10. Why does distance affect fare the most?
    → Longer distance → higher fare (direct relation).

### 🔹 Practical

11. How would you improve the model further?

    * Use **Random Forest / XGBoost**
    * Include **traffic or weather data**
12. What’s the impact of removing outliers?
    → Improves model stability and accuracy.

---

## ✅ **Key Takeaways**

* **Linear Regression** predicts fares effectively using simple numeric data.
* **PCA** helps reduce redundancy and speeds up computation.
* **Distance_km** is the most important feature.
* Feature scaling and data cleaning significantly affect results.

---

## 🧾 **Conclusion**

This project demonstrates how **machine learning + feature engineering + PCA** can be applied to real-world problems like **Uber Fare Prediction**.
By analyzing geographic and time-based data, we can build efficient and interpretable predictive models.

---

Would you like me to make this README into a **PDF version** (formatted neatly with headings, equations, and viva questions)? It’s perfect for printing or submission.
