import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# Load data
x_train = np.load("data/x_train.npy")
y_train = np.load("data/y_train.npy")
x_test = np.load("data/x_test.npy")
y_test = np.load("data/y_test.npy")

print("Training set shape:")
print("x_train:", x_train.shape)
print("y_train:", y_train.shape)

print("\nTest set shape:")
print("x_test:", x_test.shape)
print("y_test:", y_test.shape)

print("\nActivity labels:", np.unique(y_train))


# Try different k values using 5-fold cross-validation
k_values = range(1, 21)
cv_accuracy = []

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for k in k_values:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])

    scores = cross_val_score(
        model,
        x_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    cv_accuracy.append(scores.mean())


# Find the best k
best_k = k_values[np.argmax(cv_accuracy)]
best_cv_accuracy = max(cv_accuracy)

print("\nBest k:", best_k)
print("Best cross-validation accuracy:", round(best_cv_accuracy, 4))


# Plot cross-validation accuracy
plt.figure(figsize=(8, 5))
plt.plot(k_values, cv_accuracy, marker="o")

plt.title("KNN Cross-Validation Accuracy")
plt.xlabel("K Value")
plt.ylabel("Mean Accuracy")
plt.xticks(k_values)
plt.grid(True)

plt.tight_layout()
plt.show()


# Train final model using the selected k
final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=best_k))
])

final_model.fit(x_train, y_train)

y_prediction = final_model.predict(x_test)


# Final test evaluation
test_accuracy = accuracy_score(y_test, y_prediction)

print("\nFinal Test Accuracy:", round(test_accuracy, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_prediction))


# Confusion matrix
cm = confusion_matrix(y_test, y_prediction)

activity_names = [
    "Walking",
    "Upstairs",
    "Downstairs",
    "Sitting",
    "Standing",
    "Lying"
]

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=activity_names
).plot()

plt.title("KNN Confusion Matrix")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.show()
