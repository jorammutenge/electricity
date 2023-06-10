import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np


# Load the trained model
reg = xgb.XGBRegressor()
reg.load_model("model.json")  # Replace "trained_model.xgb" with the actual path to the trained model file

# Load the data
df_all = pd.read_csv("data/data_2.csv")  # Replace "data.csv" with the actual path to your data file

# Partition features and target
FEATURES = df_all.iloc[:, 1:].columns
TARGET = df_all.iloc[:, [0]].columns

X_all, y_all = df_all[FEATURES], df_all[TARGET]


def predict_electricity_price(start_date, end_date):
    # Filter data based on the given date range
    mask = (df_all['Time'] >= start_date) & (df_all['Time'] <= end_date)
    X_pred = X_all.loc[mask]

    # Make predictions
    y_pred = reg.predict(X_pred)

    # Multiply the predicted usage by 2.022 to determine the dollar value
    y_pred_dollar = y_pred * 2.022

    # Create a DataFrame with the predicted values
    pred_df = pd.DataFrame({'Date': X_pred.index, 'Predicted Price': y_pred_dollar})

    return pred_df


def main():
    st.title("Electricity Usage Price Prediction")

    # Date range input
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")

    if start_date and end_date:
        # Convert the dates to the required format (e.g., 'yyyy-mm-dd')
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        # Perform prediction
        prediction = predict_electricity_price(start_date_str, end_date_str)

        # Display the prediction results
        st.subheader("Prediction Results")
        st.dataframe(prediction)


if __name__ == '__main__':
    main()
