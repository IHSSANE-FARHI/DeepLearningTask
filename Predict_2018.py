# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import joblib
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.layouts import gridplot
import os
import webbrowser

# Configuration
MODEL_PATH = "weather_predictor.pkl"
START_DATE = "2018-01-01"  # First prediction date
N_DAYS = 365               # Number of days to predict

def preprocess_data(train_df):
    """Add lag, rolling mean, and temporal features."""
    if train_df.empty:
        raise ValueError("The training dataset is empty.")

    # Create lag features
    for i in range(1, 4):
        train_df[f'MinTemp_Lag{i}'] = train_df['MinTemp'].shift(i)
        train_df[f'MaxTemp_Lag{i}'] = train_df['MaxTemp'].shift(i)
        train_df[f'Rainfall_Lag{i}'] = train_df['Rainfall'].shift(i)

    # Create rolling mean features
    train_df['MinTemp_7DayMean'] = train_df['MinTemp'].shift(1).rolling(7).mean()
    train_df['MaxTemp_7DayMean'] = train_df['MaxTemp'].shift(1).rolling(7).mean()

    # Ensure the Date column is in datetime format
    train_df['Date'] = pd.to_datetime(train_df['Date'], errors='coerce')

    # Add temporal features (DayOfWeek, Month, DayOfYear)
    train_df['DayOfWeek'] = train_df['Date'].dt.dayofweek
    train_df['Month'] = train_df['Date'].dt.month
    train_df['DayOfYear'] = train_df['Date'].dt.dayofyear

    return train_df.dropna()

def predict_future():
    """Generate predictions for 2018 and save them to a CSV file."""
    # Load model
    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    # Load training data
    print("Loading training data...")
    train_df = pd.read_csv(r'C:\Master DSEF\Semester3\Deep_learning\DL_final_project\TemperatureRainFall.csv')

    if train_df.empty:
        raise ValueError("The training dataset is empty. Check the file path or content.")
    
    print("Preprocessing data...")
    train_df = preprocess_data(train_df)
    
    # Expected features
    expected_features = [
        'MinTemp_Lag1', 'MaxTemp_Lag1', 'Rainfall_Lag1',
        'MinTemp_Lag2', 'MaxTemp_Lag2', 'Rainfall_Lag2',
        'MinTemp_Lag3', 'MaxTemp_Lag3', 'Rainfall_Lag3',
        'MinTemp_7DayMean', 'MaxTemp_7DayMean',
        'DayOfWeek', 'Month', 'DayOfYear'
    ]
    
    # Add missing columns if necessary
    for col in expected_features:
        if col not in train_df.columns:
            print(f"Adding missing column: {col}")
            train_df[col] = 0

    # Create initial feature window
    window = train_df.tail(7).copy()
    predictions = []

    current_date = pd.to_datetime(START_DATE)
    for _ in range(N_DAYS):
        temp_df = window.tail(7).copy()
        for i in range(1, 4):
            temp_df[f'MinTemp_Lag{i}'] = temp_df['MinTemp'].shift(i).values[-1]
            temp_df[f'MaxTemp_Lag{i}'] = temp_df['MaxTemp'].shift(i).values[-1]
            temp_df[f'Rainfall_Lag{i}'] = temp_df['Rainfall'].shift(i).values[-1]

        temp_df['MinTemp_7DayMean'] = temp_df['MinTemp'].rolling(7).mean().values[-1]
        temp_df['MaxTemp_7DayMean'] = temp_df['MaxTemp'].rolling(7).mean().values[-1]
        temp_df['DayOfWeek'] = current_date.dayofweek
        temp_df['Month'] = current_date.month
        temp_df['DayOfYear'] = current_date.dayofyear

        # Use expected features for GradientBoostingRegressor
        X = temp_df[expected_features].values[-1].reshape(1, -1)

        # Predictions
        gb_pred = model.predict(X)[0]

        predictions.append({
            'Date': current_date.strftime('%Y-%m-%d'),
            'MinTemp': gb_pred[0],
            'MaxTemp': gb_pred[1],
            '9amTemp': gb_pred[2],
            '3pmTemp': gb_pred[3],
            'Rainfall': gb_pred[4]
        })

        new_row = {
            'Date': current_date.strftime('%Y-%m-%d'),
            'MinTemp': gb_pred[0],
            'MaxTemp': gb_pred[1],
            '9amTemp': gb_pred[2],
            '3pmTemp': gb_pred[3],
            'Rainfall': gb_pred[4]
        }
        window = pd.concat([window, pd.DataFrame([new_row])], ignore_index=True)

        current_date += pd.Timedelta(days=1)

    results = pd.DataFrame(predictions)
    results.to_csv("2018_predictions_gb.csv", index=False)
    print("2018 predictions saved!")
    print(results.head())

    return results

def create_visualizations(results):
    """Create visualizations for the predictions."""
    results['Date'] = pd.to_datetime(results['Date'])
    source = ColumnDataSource(results)

    output_path = os.path.abspath("2018_predictions_gb_viz.html")
    output_file(output_path)

    # Create plots
    plots = []
    for feature, color in zip(
        ['MinTemp', 'MaxTemp', '9amTemp', '3pmTemp', 'Rainfall'],
        ['blue', 'green', 'orange', 'red', 'purple']
    ):
        p = figure(
            x_axis_type="datetime",
            title=f"{feature} Predictions for 2018",
            width=800,
            height=300,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            toolbar_location="above"
        )
        p.line('Date', feature, source=source, legend_label=feature, color=color, line_width=2)
        p.add_tools(HoverTool(tooltips=[("Date", "@Date{%F}"), (feature, f"@{feature}{{0.2f}}")],
                              formatters={'@Date': 'datetime'}))
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        plots.append([p])

    # Combine plots into a grid
    grid = gridplot(plots)

    # Show and open in browser
    show(grid)
    webbrowser.open(f"file://{output_path}")

if __name__ == "__main__":
    results = predict_future()
    create_visualizations(results)
