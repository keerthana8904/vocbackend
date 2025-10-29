# ------------------------------------------------------
# train_rf_model_simple.py  ✅ (4-feature model for ESP32 + Flask backend)
# ------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import randint
import joblib
import os

# ------------------------------------------------------
# 1. Synthetic Dataset Creation
# ------------------------------------------------------
np.random.seed(42)

n = 5000
labels = ['healthy', 'asthma', 'copd', 'covid19', 'pneumonia', 'lung_cancer']
samples_per_class = n // len(labels)
data = []

for label in labels:
    if label == 'healthy':
        mq2 = np.random.normal(900, 100, samples_per_class)
        mq3 = np.random.normal(2900, 150, samples_per_class)
        mq135 = np.random.normal(950, 80, samples_per_class)
        temp = np.random.normal(28, 1.2, samples_per_class)
    elif label == 'asthma':
        mq2 = np.random.normal(1200, 150, samples_per_class)
        mq3 = np.random.normal(3100, 120, samples_per_class)
        mq135 = np.random.normal(1600, 150, samples_per_class)
        temp = np.random.normal(29, 1.0, samples_per_class)
    elif label == 'copd':
        mq2 = np.random.normal(1400, 150, samples_per_class)
        mq3 = np.random.normal(3200, 130, samples_per_class)
        mq135 = np.random.normal(1800, 180, samples_per_class)
        temp = np.random.normal(29.5, 1.2, samples_per_class)
    elif label == 'covid19':
        mq2 = np.random.normal(1000, 120, samples_per_class)
        mq3 = np.random.normal(2800, 150, samples_per_class)
        mq135 = np.random.normal(1500, 100, samples_per_class)
        temp = np.random.normal(30, 1.0, samples_per_class)
    elif label == 'pneumonia':
        mq2 = np.random.normal(1600, 150, samples_per_class)
        mq3 = np.random.normal(3400, 100, samples_per_class)
        mq135 = np.random.normal(2000, 200, samples_per_class)
        temp = np.random.normal(30.5, 1.0, samples_per_class)
    elif label == 'lung_cancer':
        mq2 = np.random.normal(1800, 200, samples_per_class)
        mq3 = np.random.normal(3500, 100, samples_per_class)
        mq135 = np.random.normal(2200, 150, samples_per_class)
        temp = np.random.normal(31, 1.0, samples_per_class)

    for i in range(samples_per_class):
        data.append([mq2[i], mq3[i], mq135[i], temp[i], label])

df = pd.DataFrame(data, columns=["mq2_adc", "mq3_adc", "mq135_adc", "temp_c", "label"])
os.makedirs("data", exist_ok=True)
df.to_csv("data/synthetic_voc_dataset.csv", index=False)
print("✅ Dataset saved as data/synthetic_voc_dataset.csv")

# ------------------------------------------------------
# 2. Train-Test Split
# ------------------------------------------------------
X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------
# 3. Model Training (RandomizedSearchCV)
# ------------------------------------------------------
param_dist = {
    "n_estimators": randint(100, 400),
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": randint(2, 10),
    "min_samples_leaf": randint(1, 5),
    "bootstrap": [True, False]
}

rf = RandomForestClassifier(random_state=42)
search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
    random_state=42
)

search.fit(X_train, y_train)
best_model = search.best_estimator_
print("\n🌟 Best Params:", search.best_params_)

# ------------------------------------------------------
# 4. Evaluate Model
# ------------------------------------------------------
y_pred = best_model.predict(X_test)
print("\n✅ Test Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# ------------------------------------------------------
# 5. Save Model
# ------------------------------------------------------
os.makedirs("model", exist_ok=True)
joblib.dump(best_model, "model/voc_random_forest_model.pkl")
print("\n💾 Model saved as model/voc_random_forest_model.pkl")
