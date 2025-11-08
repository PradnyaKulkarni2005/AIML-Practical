

# 📧 Email Spam Detection using SVM (from Scratch)

### 🔍 Objective

To implement a **binary classification model** that classifies emails as **Normal (Not Spam)** or **Abnormal (Spam)** using the **Support Vector Machine (SVM)** algorithm — implemented **from scratch (no sklearn SVM)**.

---

## 🧠 1. Introduction

* **Spam Detection** is a classic binary classification problem.
* Goal: Predict whether an email is **Spam (1)** or **Not Spam (0)** based on word-frequency features extracted from the email content.
* **Algorithm used:** Support Vector Machine (SVM)
* **Why SVM?**

  * It finds the **optimal separating hyperplane** with **maximum margin**.
  * Works well with **high-dimensional data** like text.

---

## 📊 2. Dataset Description

* The dataset (`emails.csv`) contains:

  * Word frequency features such as `the`, `to`, `and`, `for`, `you`, etc.
  * The last column `Prediction`:

    * `0` → Not Spam (Normal)
    * `1` → Spam (Abnormal)
* The column `Email No.` is removed (it’s just an identifier).

**Example:**

| the | to | ect | and | for | of | you | ... | Prediction |
| --- | -- | --- | --- | --- | -- | --- | --- | ---------- |
| 0   | 0  | 1   | 0   | 0   | 0  | 0   | ... | 0          |
| 8   | 13 | 24  | 6   | 6   | 2  | 1   | ... | 1          |

---

## ⚙️ 3. Project Flow

| Step | Task             | Description                                            |
| ---- | ---------------- | ------------------------------------------------------ |
| 1    | Import libraries | numpy, pandas, sklearn                                 |
| 2    | Load dataset     | Read CSV file                                          |
| 3    | Preprocess data  | Drop “Email No.”, separate features (X) and labels (y) |
| 4    | Encode labels    | Convert 0 → -1, 1 → +1 for SVM math                    |
| 5    | Scale features   | Use StandardScaler to normalize data                   |
| 6    | Train-test split | Split data into 80% training, 20% testing              |
| 7    | Implement SVM    | Create custom SVM class using gradient descent         |
| 8    | Train model      | Update weights and bias over iterations                |
| 9    | Predict          | Classify emails as spam/not spam                       |
| 10   | Evaluate         | Measure accuracy, precision, recall, F1 score          |

---

## 🧩 4. SVM Algorithm – Theory

### 🎯 Goal

Find the **best hyperplane** that separates the data into two classes with the **maximum margin**.

For each training sample ( (x_i, y_i) ):

[
y_i \in {-1, +1}
]

The decision boundary (hyperplane) is:

[
w \cdot x - b = 0
]

Where:

* ( w ) = weight vector
* ( b ) = bias term

### ✅ Classification Rule

[
\text{Predict } y =
\begin{cases}
+1 & \text{if } (w \cdot x - b) \ge 0 \
-1 & \text{otherwise}
\end{cases}
]

---

## 🧮 5. SVM Objective Function (Loss Function)

We minimize the following function:

[
L(w, b) = \frac{1}{2} ||w||^2 + C \sum_{i=1}^{n} \max(0, 1 - y_i (w \cdot x_i - b))
]

Where:

* ( \frac{1}{2}||w||^2 ) → Regularization term (keeps margin large)
* ( \max(0, 1 - y_i (w \cdot x_i - b)) ) → Hinge loss (penalizes misclassification)
* ( C = \frac{1}{2\lambda} ) → Regularization constant (controls trade-off)

---

## 🔁 6. Optimization – Gradient Descent Updates

### Case 1: Correct Classification

If ( y_i(w \cdot x_i - b) \ge 1 ):

[
w = w - \eta (2\lambda w)
]
[
b = b
]

### Case 2: Misclassification

If ( y_i(w \cdot x_i - b) < 1 ):

[
w = w - \eta (2\lambda w - y_i x_i)
]
[
b = b - \eta y_i
]

Where:

* ( \eta ) = learning rate
* ( \lambda ) = regularization parameter

---

## 🧠 7. Implementation (Core Code)

```python
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.001, n_iters=2000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)
```

---

## 📈 8. Evaluation Metrics

We use the following to evaluate the model:

| Metric                   | Formula                                                         | Interpretation                                     |
| ------------------------ | --------------------------------------------------------------- | -------------------------------------------------- |
| **Accuracy**             | ( \frac{TP + TN}{TP + TN + FP + FN} )                           | Overall correctness                                |
| **Precision**            | ( \frac{TP}{TP + FP} )                                          | Fraction of predicted spams that are actually spam |
| **Recall (Sensitivity)** | ( \frac{TP}{TP + FN} )                                          | Fraction of actual spams detected                  |
| **F1 Score**             | ( 2 \times \frac{Precision \times Recall}{Precision + Recall} ) | Balance between precision and recall               |

Where:

* TP = True Positive
* TN = True Negative
* FP = False Positive
* FN = False Negative

---

## 🧮 9. Results (Sample Output)

After running and tuning:

```
Unique predictions: [0 1]
Accuracy: 0.85
Precision: 0.79
Recall: 0.82
F1 Score: 0.80
```

*(Values depend on dataset and tuning)*

---

## ⚖️ 10. Advantages of SVM

* Works well in **high-dimensional** spaces (like word frequencies).
* **Margin maximization** reduces overfitting.
* Performs well when classes are **linearly separable**.

---

## 🚫 11. Limitations of SVM

* **Not good for very large datasets** (computationally heavy).
* **Sensitive to feature scaling** (must normalize data).
* **Hard to tune** regularization and learning rate manually.
* **Does not work well** when classes overlap too much without using kernels.

---

## 🧩 12. Improvements / Future Work

* Use **kernel trick (RBF or polynomial)** for nonlinear separation.
* Implement **class weighting** for imbalanced data.
* Try **TF-IDF features** instead of raw word counts.
* Compare with **Logistic Regression** or **Naive Bayes**.

---

## 💬 13. Viva Preparation Questions

Here are the **expected viva questions** with **answers** 👇

| **Question**                                                | **Answer**                                                                                                                     |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| What is the main goal of SVM?                               | To find a hyperplane that separates data points of different classes with maximum margin.                                      |
| What is the equation of a hyperplane?                       | ( w \cdot x - b = 0 )                                                                                                          |
| What is margin in SVM?                                      | The distance between the hyperplane and the nearest data points (support vectors).                                             |
| What is a support vector?                                   | Data points that lie closest to the decision boundary and influence its position.                                              |
| Why convert labels to -1 and +1?                            | SVM uses sign of ( y_i(w \cdot x_i - b) ) for classification and optimization.                                                 |
| What is hinge loss?                                         | ( \max(0, 1 - y_i(w \cdot x_i - b)) ), which penalizes misclassifications.                                                     |
| What does the regularization term do?                       | Prevents overfitting by keeping weights small.                                                                                 |
| What is the role of learning rate?                          | Controls how fast weights are updated during training.                                                                         |
| What happens if features are not scaled?                    | SVM may give poor results because large features dominate updates.                                                             |
| What is the difference between SVM and Logistic Regression? | Logistic regression minimizes log loss, while SVM minimizes hinge loss and focuses on margin maximization.                     |
| Why is accuracy not enough for imbalanced data?             | Because the model can predict only the majority class and still get high accuracy. Precision and recall are better indicators. |

---

## 📘 14. References

* *“Pattern Recognition and Machine Learning” – Christopher Bishop*
* *Scikit-Learn documentation*
* *CS229 (Stanford) Notes on SVM*
* *Wikipedia: Support Vector Machine*

---

✅ **End of README**
This covers both **implementation** and **theoretical** concepts, making it perfect for assignment submission and viva preparation.

---

