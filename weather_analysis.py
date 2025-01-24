# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import joblib
import os
import webbrowser
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, RangeTool
from bokeh.layouts import gridplot, column

def load_and_preprocess(filepath):
    """Enhanced preprocessing with full error handling"""
    try:
        df = pd.read_csv(filepath)
        
        # Date handling
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']).sort_values('Date')
        
        # Feature engineering
        for i in range(1, 4):
            df[f'MinTemp_Lag{i}'] = df['MinTemp'].shift(i)
            df[f'MaxTemp_Lag{i}'] = df['MaxTemp'].shift(i)
            df[f'Rainfall_Lag{i}'] = df['Rainfall'].shift(i)
        
        # Rolling means
        df['MinTemp_7DayMean'] = df['MinTemp'].shift(1).rolling(7).mean()
        df['MaxTemp_7DayMean'] = df['MaxTemp'].shift(1).rolling(7).mean()
        
        # Temporal features
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['DayOfYear'] = df['Date'].dt.dayofyear
        
        return df.dropna()
    
    except Exception as e:
        print(f"Preprocessing failed: {str(e)}")
        return None

def create_visualization(df_results):
    """Guaranteed visualization output with fallbacks"""
    output_path = os.path.abspath("weather_predictions.html")
    print(f"\nSaving visualization to: {output_path}")
    
    try:
        # Verify data structure
        required_cols = ['Date'] + \
                       [f"{f}_real" for f in ['MinTemp', 'MaxTemp', '9amTemp', '3pmTemp', 'Rainfall']] + \
                       [f"{f}_pred" for f in ['MinTemp', 'MaxTemp', '9amTemp', '3pmTemp', 'Rainfall']]
        
        missing = [col for col in required_cols if col not in df_results.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        # Create visualization
        output_file(output_path)
        source = ColumnDataSource(df_results)
        plots = []
        
        for feature in ['MinTemp', 'MaxTemp', '9amTemp', '3pmTemp', 'Rainfall']:
            # Main plot
            p = figure(
                x_axis_type="datetime",
                title=f"{feature} Predictions",
                width=1200,
                height=400,
                tools="pan,wheel_zoom,box_zoom,reset,save"
            )
            p.line('Date', f'{feature}_real', source=source, 
                   legend_label='Actual', color='navy', line_width=2)
            p.line('Date', f'{feature}_pred', source=source,
                   legend_label='Predicted', color='firebrick', 
                   line_dash='dashed', line_width=2)
            
            # Range selector
            range_tool = figure(
                height=100,
                width=1200,
                x_axis_type="datetime",
                y_axis_type=None,
                tools="",
                toolbar_location=None
            )
            range_tool.line('Date', f'{feature}_real', source=source)
            rt = RangeTool(x_range=p.x_range)
            rt.overlay.fill_color = "green"
            rt.overlay.fill_alpha = 0.2
            range_tool.add_tools(rt)
            
            plots.append(column(p, range_tool))
        
        grid = gridplot(plots, ncols=1)
        show(grid)
        
    except Exception as e:
        print(f"Visualization error: {str(e)}")
        # Create simple fallback plot
        p = figure(title="Basic Preview", width=400, height=400)
        p.line(df_results.index, df_results['MinTemp_real'], legend_label='MinTemp')
        show(p)
    
    finally:
        webbrowser.open(f"file://{output_path}")

if __name__ == "__main__":
    # Configuration
    DATA_PATH = r'C:\Master DSEF\Semester3\Deep_learning\DL_final_project\TemperatureRainFall.csv'
    MODEL_PATH = "weather_predictor.pkl"
    
    try:
        # 1. Load and preprocess data
        print("Loading data...")
        df = load_and_preprocess(DATA_PATH)
        if df is None:
            raise SystemExit("Failed to load data")
        
        # 2. Prepare features/targets
        features = ["MinTemp_Lag1", "MaxTemp_Lag1", "Rainfall_Lag1",
                    "MinTemp_7DayMean", "MaxTemp_7DayMean",
                    "DayOfWeek", "Month", "DayOfYear"]
        targets = ["MinTemp", "MaxTemp", "9amTemp", "3pmTemp", "Rainfall"]
        
        X = df[features]
        y = df[targets]
        
        # 3. Train/test split (time-based)
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # 4. Load or train model
        if os.path.exists(MODEL_PATH):
            print("Loading existing model...")
            model = joblib.load(MODEL_PATH)
        else:
            print("Training new model...")
            model = MultiOutputRegressor(
                GradientBoostingRegressor(n_estimators=200, max_depth=5)
            )
            model.fit(X_train, y_train)
            joblib.dump(model, MODEL_PATH)
        
        # 5. Generate predictions
        print("Generating predictions...")

        # Align columns with model
        expected_features = model.estimators_[0].feature_names_in_
        for col in expected_features:
            if col not in X.columns:
                print(f"Ajout de la colonne manquante : {col}")
                X[col] = 0
        X = X[expected_features]

        predictions = model.predict(X)
        df_results = pd.DataFrame({
            'Date': df['Date'],
            'MinTemp_real': df['MinTemp'],
            'MinTemp_pred': predictions[:, 0],
            'MaxTemp_real': df['MaxTemp'],
            'MaxTemp_pred': predictions[:, 1],
            '9amTemp_real': df['9amTemp'],
            '9amTemp_pred': predictions[:, 2],
            '3pmTemp_real': df['3pmTemp'],
            '3pmTemp_pred': predictions[:, 3],
            'Rainfall_real': df['Rainfall'],
            'Rainfall_pred': predictions[:, 4]
        })
        
        # 6. Create visualization
        print("Creating visualization...")
        create_visualization(df_results)
        
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        if 'df' in locals():
            df.to_csv("debug_data.csv", index=False)
            print("Debug data saved to debug_data.csv")