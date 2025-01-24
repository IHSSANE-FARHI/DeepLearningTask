# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf 
import os
import webbrowser
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, RangeTool
from bokeh.layouts import gridplot, column
from bokeh.palettes import Category10

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

def create_lstm_model(input_shape):
    """Create and compile LSTM model"""
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=False),
        Dense(32, activation='relu'),
        Dense(5)  # 5 output nodes for all targets
    ])
    model.compile(
        loss='mse', 
        optimizer=Adam(learning_rate=0.001), 
        metrics=['mae']
    )
    return model

def create_visualization(df_results):
    """Visualization with both models comparison"""
    output_path = os.path.abspath("model_comparison.html")
    print(f"\nSaving visualization to: {output_path}")
    
    try:
        output_file(output_path)
        source = ColumnDataSource(df_results)
        plots = []
        
        features = ['MinTemp', 'MaxTemp', '9amTemp', '3pmTemp', 'Rainfall']
        
        for feature in features:
            # Main plot
            p = figure(
                x_axis_type="datetime",
                title=f"{feature} Predictions Comparison",
                width=1200,
                height=400,
                tools="pan,wheel_zoom,box_zoom,reset,save"
            )
            
            # Actual values
            p.line('Date', f'{feature}_real', source=source, 
                   legend_label='Actual', color=Category10[3][0], line_width=2)
            
            # Gradient Boosting predictions
            p.line('Date', f'{feature}_gb_pred', source=source,
                   legend_label='Gradient Boosting', color=Category10[3][1], 
                   line_dash='dashed', line_width=2)
            
            # LSTM predictions
            p.line('Date', f'{feature}_lstm_pred', source=source, 
                   legend_label='LSTM', color=Category10[3][2], 
                   line_dash='dotted', line_width=2)
            
            # Range tool
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
            rt.overlay.fill_color = "navy"
            rt.overlay.fill_alpha = 0.2
            range_tool.add_tools(rt)
            
            plots.append(column(p, range_tool))
        
        grid = gridplot(plots, ncols=1)
        show(grid)
        
    except Exception as e:
        print(f"Visualization error: {str(e)}")
        # Fallback plot
        p = figure(title="Basic Preview", width=400, height=400)
        p.line(df_results.index, df_results['MinTemp_real'], legend_label='MinTemp')
        show(p)
    
    finally:
        webbrowser.open(f"file://{output_path}")

if __name__ == "__main__":
    # Configuration
    DATA_PATH = r'C:\Master DSEF\Semester3\Deep_learning\DL_final_project\TemperatureRainFall.csv'
    MODEL_PATH = "weather_predictor.pkl"
    LSTMMODEL_PATH = "weather_predictor_lstm.h5"
    
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
        y = df[targets].values
        
        # 3. Train/test split
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 4. Gradient Boosting Model
        if os.path.exists(MODEL_PATH):
            print("Loading GB model...")
            gb_model = joblib.load(MODEL_PATH)
        else:
            print("Training GB model...")
            gb_model = MultiOutputRegressor(
                GradientBoostingRegressor(n_estimators=200, max_depth=5)
            )
            gb_model.fit(X_train, y_train)
            joblib.dump(gb_model, MODEL_PATH)
        
        # 5. LSTM Model
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Reshape for LSTM [samples, timesteps, features]
        X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
        
        if os.path.exists(LSTMMODEL_PATH):
            print("Loading LSTM model...")
            lstm_model = tf.keras.models.load_model(
            LSTMMODEL_PATH,  
            custom_objects={'mse': 'mse'}
        )
        else:
            print("Training LSTM model...")
            lstm_model = create_lstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]))
            history = lstm_model.fit(
                X_train_lstm, y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=1
            )
            lstm_model.save(LSTMMODEL_PATH)   
        # 6. Generate predictions
        print("Generating predictions...")
        
        # Alignement des colonnes avant la prédiction
        expected_features = gb_model.estimators_[0].feature_names_in_

        for col in expected_features:
            if col not in X.columns:
                print(f"Ajout de la colonne manquante : {col}")
                X[col] = 0

        X = X[expected_features]

        gb_pred = gb_model.predict(X)
        lstm_pred = lstm_model.predict(X.values.reshape(X.shape[0], 1, X.shape[1]))
        
        # Create results DataFrame
        df_results = pd.DataFrame({
            'Date': df['Date'],
            'MinTemp_real': df['MinTemp'],
            'MinTemp_gb_pred': gb_pred[:, 0],
            'MinTemp_lstm_pred': lstm_pred[:, 0],
            'MaxTemp_real': df['MaxTemp'],
            'MaxTemp_gb_pred': gb_pred[:, 1],
            'MaxTemp_lstm_pred': lstm_pred[:, 1],
            '9amTemp_real': df['9amTemp'],
            '9amTemp_gb_pred': gb_pred[:, 2],
            '9amTemp_lstm_pred': lstm_pred[:, 2],
            '3pmTemp_real': df['3pmTemp'],
            '3pmTemp_gb_pred': gb_pred[:, 3],
            '3pmTemp_lstm_pred': lstm_pred[:, 3],
            'Rainfall_real': df['Rainfall'],
            'Rainfall_gb_pred': gb_pred[:, 4],
            'Rainfall_lstm_pred': lstm_pred[:, 4]
        })
        
        # 7. Create comparison visualization
        print("Creating visualization...")
        create_visualization(df_results)
        
        # 8. Calculate and print metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        def calculate_metrics(y_true, y_pred, model_name):
            return {
                'MAE': mean_absolute_error(y_true, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred))
            }
        
        gb_metrics = calculate_metrics(y_test, gb_model.predict(X_test), "Gradient Boosting")
        lstm_metrics = calculate_metrics(y_test, lstm_model.predict(X_test_lstm), "LSTM")
        
        metrics_df = pd.DataFrame({
            'Gradient Boosting': gb_metrics,
            'LSTM': lstm_metrics
        }).T
        
        print("\nModel Performance Comparison:")
        print(metrics_df)
        
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        if 'df' in locals():
            df.to_csv("debug_data.csv", index=False)
            print("Debug data saved to debug_data.csv")