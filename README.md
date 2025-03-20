# CO₂ Storage Efficiency Prediction Using Well Log Data

This project aims to predict CO₂ storage efficiency using well log data from the Smeaheia reservoir. By leveraging geological features such as porosity, density, and gamma ray, the model aims to estimate CO₂ storage potential in subsurface formations, an essential task for carbon capture and storage (CCS) projects.

## Project Overview

- **Goal:** Predict CO₂ storage efficiency based on well log features.
- **Dataset Used:** Smeaheia Norway well log dataset.
- **Model:** Linear Regression was employed for prediction. Future improvements may include more complex models (Random Forest, Gradient Boosting).
- **Metrics:** 
  - **R-squared**: 0.9923 (indicating that the model explains over 99% of the variance).
  - **Mean Absolute Error**: 0.0137.

### Data License
- The datasets used in this project are made available under **[Smeaheia Dataset License](https://co2datashare.org/smeaheia-dataset/static/SMEAHEIA%20DATASET%20LICENSE_Gassnova%20and%20Equinor.pdf)** by **Equinor** and **Gassnova**. You can use the data for research and non-commercial purposes while providing appropriate credit.

---

## Table of Contents

1. [Project Setup](#project-setup)
2. [Data Exploration](#data-exploration)
3. [Feature Engineering](#feature-engineering)
4. [Modeling and Evaluation](#modeling-and-evaluation)
5. [Conclusion](#conclusion)
6. [Next Steps](#next-steps)

## Project Setup

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lasio
```

### Loading the Data

```python
import pandas as pd
import lasio

# Load well log data
las = lasio.read('path_to_your_well_log_data')
df_well_log = las.df()
```

## Data Exploration

The data is explored by visualizing geological features such as:

- **Depth vs. Gamma Ray**: Shale vs Non-Shale.
- **Depth vs. Porosity**: To understand the porosity at different depths.
- **Correlation Analysis**: To examine relationships between various geological parameters.

## Feature Engineering

Key interaction features were created:

- **Effective Porosity x Depth**: To analyze how porosity at different depths affects CO₂ storage.
- **Density x Depth**: Understanding the relationship between rock density and depth.
- **Gamma Ray x Porosity**: Impact of shale content and porosity on CO₂ storage.

## Modeling and Evaluation

A **Linear Regression** model was built to predict CO₂ storage efficiency. Data was split into training and testing sets, and the model was evaluated using:

- **R-squared:** 0.9923
- **Mean Absolute Error (MAE):** 0.0137

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print(f"R-squared: {r2_score(y_test, y_pred)}")
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
```

## Conclusion

- The model demonstrates a strong performance in predicting CO₂ storage efficiency, explaining over 99% of the variance.
- Linear Regression was effective, but more advanced models could be explored for improved accuracy.
- The model has been evaluated visually using actual vs predicted plots and residuals.

## Next Steps

- Explore advanced models such as **Random Forest** or **Gradient Boosting**.
- Further feature engineering to improve prediction accuracy.
- Incorporate additional datasets (temperature, pressure, etc.) for more comprehensive modeling.
