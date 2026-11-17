# Insurance Claim Prediction using Machine Learning

> A binary classification project to identify vehicle insurance policies that are more likely to result in a claim.

## 1. Business Story: Why does this matter?

Insurance companies need to understand claim risk so they can plan operations, review portfolios and focus attention where it may be needed. Reviewing every policy in the same way can be inefficient. A machine learning model can help identify patterns in historical policy and vehicle information and estimate whether a policy is associated with a claim.

This project explores vehicle insurance data and builds classification models to predict `claim_status`.

**Important:** The model is a learning and analytical prototype. It should not be used to approve, reject, price or otherwise make decisions about real customers without further validation, fairness assessment and appropriate business/legal review.

## 2. Project Objective

- Explore customer, policy, regional and vehicle-related information.
- Understand the distribution of the target variable, `claim_status`.
- Prepare selected features for machine learning.
- Train and compare Decision Tree, Random Forest and XGBoost classifiers.
- Evaluate models using class-level precision, recall, F1-score and ROC-AUC.
- Select the model with the highest ROC-AUC in the notebook's test evaluation.

## 3. Dataset Overview

The notebook loads a CSV file named `Insurance claims data.csv`.

- **Rows:** 58,592
- **Original columns:** 42
- **Target:** `claim_status` (0 or 1)
- **Selected modelling table:** 15 columns, including the target

The notebook's test-set classification reports show 10,969 records in class 0 and 750 records in class 1. This indicates a substantial class imbalance: claim records are much less common than non-claim records.

> Add the dataset source, data dictionary and usage/licensing details here before publishing. Do not upload private or restricted data. If redistribution is not permitted, explain how readers can obtain the dataset.

## 4. Workflow

1. **Load and inspect data** – use Pandas to read the CSV and inspect structure, data types and missing values.
2. **Explore the target** – examine claim counts and proportions.
3. **Explore features** – review numeric and categorical variables, group-level claim rates and visual patterns.
4. **Prepare data** – remove selected columns, convert selected Yes/No fields to numeric values and encode categorical variables with `OrdinalEncoder`.
5. **Split data** – use an 80/20 train-test split with stratification to preserve the target-class proportions.
6. **Address imbalance** – use `class_weight="balanced"` for Decision Tree and Random Forest, and calculate `scale_pos_weight` for XGBoost.
7. **Train models** – Decision Tree, Random Forest and XGBoost.
8. **Evaluate and compare** – print classification reports and calculate ROC-AUC.
9. **Choose a candidate** – select the model with the highest ROC-AUC among the three models evaluated.

## 5. Features Used in the Notebook

The selected modelling table contains these input fields:

- `subscription_length`
- `vehicle_age`
- `customer_age`
- `region_density`
- `model`
- `is_parking_sensors`
- `transmission_type`
- `steering_type`
- `is_front_fog_lights`
- `is_brake_assist`
- `is_power_steering`
- `is_driver_seat_height_adjustable`
- `is_day_night_rear_view_mirror`
- `is_speed_alert`

Target: `claim_status`

The notebook drops a number of original columns before modelling. Review the feature-selection rationale and confirm that no useful or potentially sensitive variables are being removed or retained without justification.

## 6. Models and Test Results

The following results are copied from the notebook's saved test-set output.

| Model | Accuracy | Class 1 Precision | Class 1 Recall | Class 1 F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 0.62 | 0.10 | 0.59 | 0.17 | 0.6388 |
| Random Forest | 0.60 | 0.10 | 0.63 | 0.17 | 0.6545 |
| XGBoost | 0.54 | 0.09 | 0.70 | 0.16 | **0.6554** |

**Notebook's selected candidate: XGBoost**, because it has the highest ROC-AUC (0.6554) among the three evaluated models.

### How to interpret this

- XGBoost has the highest ROC-AUC in this comparison, but the score indicates only modest discrimination.
- XGBoost identifies 70% of the positive class in this test report (recall), but its positive-class precision is 9%. In other words, many records predicted as claims are false positives.
- Accuracy alone is not a sufficient measure here because the target classes are imbalanced.
- The results are an initial baseline, not evidence that the model is ready for operational use.

## 7. Technologies

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- Joblib
- Jupyter Notebook

## 8. Repository Structure

A suggested GitHub structure:

```text
Insurance-Claim-Prediction/
│
├── Insurance_claim_prediction.ipynb
├── README.md
├── Insurance claims data.csv   # Include only if sharing is permitted
├── requirements.txt
└── models/
    ├── claim_model.pkl
    ├── model_columns.pkl
    └── model_options.pkl
```

Only include saved model files after correcting and validating the model-saving step. Avoid committing large or restricted datasets and generated files unnecessarily.

## 9. How to Run

1. Clone or download this repository.
2. Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter
```

3. Place the dataset in the project directory, or update the CSV path in the notebook.
4. Open the notebook:

```bash
jupyter notebook Insurance_claim_prediction.ipynb
```

5. Run the cells from top to bottom.

## 10. Important Validation Before Publishing

A review of the uploaded notebook identified several items to fix before treating the saved model as reusable:

1. **Two columns become entirely missing:** after mapping `is_parking_sensors` and `is_front_fog_lights` with `{"Yes": 1, "No": 0}`, the notebook output shows 0 non-null values for both columns. Inspect their original unique values and map them correctly, or remove them from the feature list if appropriate. Do not train with these columns in their current all-missing state.
2. **XGBoost parameter spelling:** the notebook uses `n_estimator=300`. The usual XGBoost parameter is `n_estimators=300`. Correct it and rerun the model.
3. **Model persistence:** `joblib.dump(best_model, "claim_model.pkl")` saves the string name (`"xgboost"`), not the fitted estimator. Save the fitted object instead, for example `joblib.dump(xgb, "claim_model.pkl")` after confirming XGBoost is the selected model.
4. **Consistent preprocessing for future predictions:** save the fitted encoder and the exact feature order, and apply the same transformations to new records. Prefer a Scikit-learn `Pipeline`/`ColumnTransformer` so preprocessing and prediction remain consistent.
5. **Validation quality:** use cross-validation and, where appropriate, tune the classification threshold based on the business cost of false positives and false negatives. Assess calibration and fairness before any real-world use.

The metrics in Section 6 are the outputs currently stored in the uploaded notebook. They should be regenerated after these corrections; results may change.

## 11. Conclusion

This project demonstrates an end-to-end introductory classification workflow on vehicle insurance data: exploratory analysis, feature preparation, handling class imbalance, training multiple tree-based models and comparing their performance. XGBoost achieved the highest ROC-AUC in the notebook's current comparison (0.6554), while also showing the trade-off between positive-class recall and precision. The next step is to correct the preprocessing and model-saving issues, rerun evaluation and improve the model through systematic validation.

## 12. Future Improvements

- Investigate and correct missing or incorrectly mapped feature values.
- Use a reproducible preprocessing pipeline.
- Apply cross-validation and hyperparameter tuning.
- Compare threshold-specific business costs and precision-recall curves.
- Review feature importance or SHAP explanations.
- Evaluate model stability, calibration and fairness.
- Build a small demonstration app only after validation.

---
**Author:** Rajeswary L  
**Project:** Insurance Claim Prediction using Machine Learning
