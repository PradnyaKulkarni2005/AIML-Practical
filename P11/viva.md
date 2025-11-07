# 🏠 **Assignment 11 – Linear Regression Model for House Price Prediction**

### 🎯 **Objective:**
Implement a **Linear Regression model** to predict **house prices** using features like **area, number of bedrooms, and location**.  
Validate the model using **K-Fold Cross Validation**.

---

## 📘 **Concept Overview**

### 🔹 **Linear Regression:**

* Linear Regression is a **supervised learning algorithm** used for **predicting continuous values**.
* It finds a **linear relationship** between input features (X) and output variable (y).
* The model fits a line (or plane in higher dimensions) that best represents the data:

  ```
  y = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ
  ```

  where:
  - **y** = predicted value (house price)
  - **x₁, x₂, ...** = input features (area, bedrooms, location, etc.)
  - **b₀** = intercept
  - **bᵢ** = coefficients (weights)

---

## 💻 **Code Explanation**

### **1️⃣ Importing Libraries**

```python
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
```

👉 Libraries used:
* **pandas** – data handling
* **scikit-learn** – model creation and evaluation
* **numpy** – numerical operations

---

### **2️⃣ Loading and Understanding the Data**

```python
df = pd.read_csv("housing.csv")
print(df.head())
```
Displays the first few rows to understand the structure of the dataset.

---

### **3️⃣ Handling Missing Values**

```python
df = df.fillna(df.median(numeric_only=True))
```
Missing values are replaced by the **median** of the respective column — avoids bias caused by extreme values.

---

### **4️⃣ Encoding Categorical Variables**

```python
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)
```
* The column `ocean_proximity` contains **text data** (like "NEAR BAY", "INLAND").
* **One-Hot Encoding** converts categories into **numerical columns (0/1)**.
* `drop_first=True` avoids **dummy variable trap** (redundancy).

---

### **5️⃣ Splitting the Data**

```python
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
* **X** → features (area, bedrooms, location, etc.)
* **y** → target (house price)
* Data is split into:
  * **80% training** (for learning)
  * **20% testing** (for evaluation)

---

### **6️⃣ Building and Training the Model**

```python
model = LinearRegression()
model.fit(X_train, y_train)
```
Creates and trains a **Linear Regression** model on the training data.

---

### **7️⃣ Making Predictions**

```python
y_pred = model.predict(X_test)
```
The model predicts house prices for the test set.

---

### **8️⃣ Evaluating the Model**

```python
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
```

#### **Performance Metrics:**
* **RMSE (Root Mean Squared Error):**
  - Measures how far predictions are from actual values (lower is better).
  - Formula:
    ```
    RMSE = sqrt( (1/n) Σ(yᵢ - ŷᵢ)² )
    ```
* **R² Score (Coefficient of Determination):**
  - Indicates how well the model explains the variance in data (closer to 1 = better fit).

---

### **9️⃣ K-Fold Cross Validation**

```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf)
cv_rmse = np.sqrt(-cv_scores)
```

* Data is divided into **5 folds**.
* Model is trained and tested on each fold.
* The **average RMSE** across folds gives a more reliable estimate of performance.

---

### **🔟 Output Example**

```
Model Performance :
RMSE: 70060.52
R² Score: 0.6254

Performing 5-Fold Cross Validation..
Cross-Validation RMSE for each fold: [70060.52 69023.00 67725.64 65882.87 71714.85]
Average CV RMSE: 68881.38
```
👉 **Interpretation:**  
* Average RMSE (~68,881) shows how much predictions deviate from actual prices.
* R² Score of 0.6254 means ~62.5% of price variability is explained by the model.

---

### **🔢 Actual vs Predicted Example**

```
         Actual      Predicted
20046   47700.0    54055.44
3024    45800.0   124225.33
15663  500001.0   255489.37
20484  218600.0   268002.43
9814   278000.0   262769.43
```
* Shows how close model predictions are to real house prices.

---

## 🧠 **Concepts to Remember for Viva**

| Term                 | Meaning                                                     | Example                      |
| -------------------- | ----------------------------------------------------------- | ---------------------------- |
| **Feature**          | Independent variable (input)                                | Area, Bedrooms, Location     |
| **Target**           | Dependent variable (output)                                 | House Price                  |
| **One-Hot Encoding** | Converts text (categorical) data into numbers               | “Near Bay” → 1, “Inland” → 0 |
| **RMSE**             | Measures prediction error (lower is better)                 | 70060.52                     |
| **R² Score**         | Explains variance captured by the model                     | 0.6254                       |
| **Cross-Validation** | Technique to check model reliability                        | 5-Fold CV                    |
| **Overfitting**      | Model performs well on training data but poorly on new data | Prevented using CV           |

---

## 🕒 **Time and Space Complexity**

| Step                    | Operation                   | Complexity                            |
| ----------------------- | --------------------------- | ------------------------------------- |
| Data Cleaning           | Fill missing values         | O(n)                                  |
| Model Training          | Linear Regression fit       | O(n × p²) (n = samples, p = features) |
| Prediction              | Linear Equation Calculation | O(p)                                  |
| K-Fold Cross Validation | Repeated training (k times) | O(k × n × p²)                         |

---

## ⚙️ **Advantages**
- ✅ Easy to implement and interpret.
- ✅ Works well when the relationship between variables is linear.
- ✅ Fast to train and predict.
- ✅ Provides coefficients showing feature importance.

---

## ⚠️ **Limitations**
- ❌ Not suitable for non-linear relationships.
- ❌ Sensitive to **outliers** and **multicollinearity**.
- ❌ Requires **numerical** input (hence, needs preprocessing for categorical data).

---

## 🌍 **Real-Life Applications**
- 🏠 Predicting house prices
- 📈 Forecasting sales or stock prices
- 🚗 Predicting fuel efficiency based on engine specs
- 💼 Estimating salary based on experience and skills

---

## 🧾 **Summary Table**

| Evaluation Metric | Meaning                          | Ideal Value          |
| ----------------- | -------------------------------- | -------------------- |
| **RMSE**          | Average error in prediction      | Lower = Better       |
| **R² Score**      | Proportion of variance explained | Closer to 1 = Better |
| **CV RMSE**       | Average error across folds       | Consistent = Reliable model |

---

## ✅ **Conclusion**

* The **Linear Regression** model successfully predicts house prices using given features.
* The performance (RMSE ≈ 68,881 and R² ≈ 0.6254) shows a **moderately good fit**.
* **5-Fold Cross Validation** ensures the model is **generalized and not overfitted**.
* With more features and data normalization, performance can be improved.

---
