# CIE 2 – Activity 2: Logistic Regression for Binary Classification

## Activity
This activity uses Logistic Regression to classify a used car's transmission as **Manual** or **Automatic**. The model produces a probability for the Automatic class, converts it into a class label using a decision threshold and is evaluated using classification metrics.

## Dataset and Source
- **Dataset:** Car Details from Car Dekho
- **Source:** [Kaggle – Vehicle Dataset from CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
- **Original size:** 4,340 rows and 8 columns
- **Input features used:** year, selling_price, km_driven, fuel, seller_type and owner
- **Target:** transmission
- **Negative class (0):** Manual
- **Positive class (1):** Automatic

Duplicate records were removed, leaving 3,577 rows. The cleaned data contains 3,265 Manual cars and 312 Automatic cars, so the classes are imbalanced.

## Work Completed
1. Loaded and inspected the dataset, including sample records, data types, summary statistics, missing values and duplicate records.
2. Checked the transmission class distribution.
3. Encoded the target as Manual = 0 and Automatic = 1.
4. Applied StandardScaler to numerical features and One-Hot Encoding to categorical features through a preprocessing pipeline.
5. Split the data into 80% training and 20% testing sets using stratification.
6. Trained a Logistic Regression model and generated test probabilities and predicted labels.
7. Plotted the sigmoid function and created a confusion matrix.
8. Calculated accuracy, precision, recall, specificity, F1-score and ROC-AUC, and displayed the classification report.
9. Compared the model with a majority-class baseline and tested thresholds 0.3, 0.5 and 0.7.
10. Compared training and testing accuracy and predicted the transmission class for a new car example.

## Key Results
The test set contains 716 records: 654 Manual and 62 Automatic.

At the default threshold of 0.5, the confusion matrix is:

| | Predicted Manual | Predicted Automatic |
|---|---:|---:|
| Actual Manual | 651 | 3 |
| Actual Automatic | 40 | 22 |

- **True Negative (TN): 651** – Manual cars correctly predicted as Manual.
- **False Positive (FP): 3** – Manual cars incorrectly predicted as Automatic.
- **False Negative (FN): 40** – Automatic cars incorrectly predicted as Manual.
- **True Positive (TP): 22** – Automatic cars correctly predicted as Automatic.

| Test metric | Result |
|---|---:|
| Accuracy | 93.99% |
| Precision | 88.00% |
| Recall | 35.48% |
| Specificity | 99.54% |
| F1-score | 50.57% |
| ROC-AUC | 0.8385 |

The majority-class baseline accuracy is **91.34%**. Logistic Regression reaches **93.99%** test accuracy. However, the dataset is imbalanced and the model detects only 22 of the 62 Automatic cars at the 0.5 threshold. Therefore, accuracy alone does not describe its performance.

### Threshold Comparison

| Threshold | Predicted Automatic | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| 0.3 | 36 | 93.02% | 66.67% | 38.71% | 48.98% |
| 0.5 | 25 | 93.99% | 88.00% | 35.48% | 50.57% |
| 0.7 | 18 | 93.85% | 100.00% | 29.03% | 45.00% |

Among the three tested thresholds, 0.5 gives the highest F1-score and a strong precision value. It is a reasonable default for this comparison, but its recall is low. If missing an Automatic car is more costly, a lower threshold should be considered and validated.

The training accuracy is **93.29%** and the test accuracy is **93.99%**. A new example with year = 2018, selling_price = ₹500,000, km_driven = 30,000, Petrol fuel, Individual seller and First Owner produced an Automatic probability of approximately **5.55%** and was classified as **Manual** at threshold 0.5.

## Limitations
The classes are imbalanced, and the model misses a substantial number of Automatic cars at the default threshold. The prediction is a probability-based estimate, not a guaranteed result. The model should not be used as the only basis for a real-world decision without further validation.

## Files
- `RCA_2(2).ipynb` – Activity 2 Jupyter Notebook
- `CAR DETAILS FROM CAR DEKHO.csv` – Dataset used by the notebook

Keep the notebook and dataset together when running the project, or update the CSV path in the notebook if the file is stored in another folder.

## Tools and Libraries
Python, Jupyter Notebook, Pandas, NumPy, Matplotlib, Seaborn and Scikit-learn.

## Faculty GitHub
[https://github.com/complexiota](https://github.com/complexiota)

