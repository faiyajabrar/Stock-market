import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
import json
import os
from datetime import datetime

def load_and_prepare_data(file_path):
    """Load and prepare data for LSTM model"""
    # Load data
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Handle missing and infinite values
    df = df.ffill().bfill()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.ffill().bfill()
    
    # Ensure date column is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
    
    # Identify numeric columns only
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if 'TARGET' in numeric_cols:
        numeric_cols.remove('TARGET')
    if 'date' in numeric_cols:
        numeric_cols.remove('date')
    
    # Normalize features
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)
    
    return df, df_scaled, numeric_cols

def create_sequences(data, target, sequence_length):
    """Create sequences for LSTM input"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data.iloc[i:i+sequence_length].values)
        y.append(target.iloc[i+sequence_length])
    return np.array(X), np.array(y)

def build_model(input_shape, lstm_units, dropout_rate):
    """Build LSTM model with specified parameters"""
    model = Sequential()
    model.add(LSTM(lstm_units[0], return_sequences=(len(lstm_units) > 1), input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    
    # Add additional LSTM layers if specified
    for i in range(1, len(lstm_units)):
        return_sequences = i < len(lstm_units) - 1
        model.add(LSTM(lstm_units[i], return_sequences=return_sequences))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def train_and_evaluate(X_train, y_train, X_test, y_test, params):
    """Train and evaluate model with given parameters"""
    # Build model
    model = build_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        lstm_units=params['lstm_units'],
        dropout_rate=params['dropout_rate']
    )
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate model
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get validation metrics
    val_accuracy = max(history.history['val_accuracy'])
    val_loss = min(history.history['val_loss'])
    
    return {
        'model': model,
        'accuracy': accuracy,
        'val_accuracy': val_accuracy,
        'val_loss': val_loss,
        'epochs_trained': len(history.history['loss'])
    }

def hyperparameter_tuning(file_path, param_grid):
    """Perform hyperparameter tuning"""
    # Load and prepare data
    print("Loading and preparing data...")
    df, df_scaled, features = load_and_prepare_data(file_path)
    
    # Get target variable
    target = 'TARGET'
    if target not in df.columns:
        print(f"ERROR: '{target}' column not found in dataset!")
        return
    
    # Create results directory
    results_dir = 'tuning_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate parameter combinations
    param_combinations = list(ParameterGrid(param_grid))
    print(f"Testing {len(param_combinations)} parameter combinations...")
    
    # Results storage
    results = []
    best_accuracy = 0
    best_model = None
    best_params = None
    
    # Loop through parameter combinations
    for i, params in enumerate(param_combinations):
        print(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
        
        # Create sequences
        X, y = create_sequences(df_scaled, df[target], params['sequence_length'])
        
        # Train-test split
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train and evaluate model
        result = train_and_evaluate(X_train, y_train, X_test, y_test, params)
        
        # Add parameters to result
        result_with_params = {**params, **{
            'accuracy': result['accuracy'],
            'val_accuracy': result['val_accuracy'],
            'val_loss': result['val_loss'],
            'epochs_trained': result['epochs_trained']
        }}
        
        # Save result
        results.append(result_with_params)
        
        # Check if best model
        if result['accuracy'] > best_accuracy:
            best_accuracy = result['accuracy']
            best_model = result['model']
            best_params = params
            print(f"New best model: accuracy = {best_accuracy:.4f}")
        
        # Save current results to file
        with open(f"{results_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    # Save best model
    if best_model is not None:
        best_model.save(f"{results_dir}/best_model.h5")
        with open(f"{results_dir}/best_params.json", 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"Best model saved with accuracy: {best_accuracy:.4f}")
    
    # Return results
    return results, best_model, best_params

if __name__ == "__main__":
    # Simplified parameter grid with fewer combinations
    param_grid = {
        'sequence_length': [20, 30],
        'lstm_units': [[50], [50, 50]],
        'dropout_rate': [0.2],
        'batch_size': [32],
        'epochs': [30]
    }
    
    # Run hyperparameter tuning
    results, best_model, best_params = hyperparameter_tuning(
        'infolimpioavanzadoTarget.csv', param_grid
    )
    
    # Print best parameters
    print("\nBest parameters:")
    for key, value in best_params.items():
        print(f"{key}: {value}") 