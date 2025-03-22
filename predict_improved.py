import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Improved Stock Market Predictor')
    parser.add_argument('--date', type=str, help='Target date for prediction (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, default=5, help='Number of days to predict')
    parser.add_argument('--data', type=str, default='infolimpioavanzadoTarget.csv', help='CSV data file')
    parser.add_argument('--model', type=str, default='improved_model/improved_lstm_model.h5', help='Model file')
    parser.add_argument('--scaler', type=str, default='feature_scaler.pkl', help='Feature scaler file')
    parser.add_argument('--threshold', type=float, default=None, 
                        help='Prediction threshold (defaults to optimal value from improved_model/optimal_threshold.txt)')
    parser.add_argument('--plot', action='store_true', help='Generate prediction plot')
    return parser.parse_args()

def engineer_features(df):
    """Engineer additional features for prediction"""
    # Create a copy to avoid SettingWithCopyWarning
    df_new = df.copy()
    
    # Technical indicators
    # Moving Averages
    for window in [5, 20]:
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df_new.columns:
                df_new[f'{col}_ma{window}'] = df_new[col].rolling(window=window).mean()
                df_new[f'{col}_ma{window}_diff'] = df_new[col] - df_new[f'{col}_ma{window}']
    
    # Momentum indicators
    if 'close' in df_new.columns:
        # RSI (Relative Strength Index)
        delta = df_new['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df_new['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Handle missing values
    df_new = df_new.ffill().bfill()
    df_new.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_new = df_new.ffill().bfill()
    
    return df_new

def load_and_preprocess_data(file_path, scaler_path):
    """Load and preprocess data with feature engineering"""
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
    
    # Engineer features
    df = engineer_features(df)
    
    # Load the scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if 'TARGET' in numeric_cols: numeric_cols.remove('TARGET')
    if 'date' in numeric_cols: numeric_cols.remove('date')
    
    # Check if we have all required features
    if os.path.exists('improved_model/feature_names.txt'):
        with open('improved_model/feature_names.txt', 'r') as f:
            required_features = f.read().splitlines()
        
        # Check for missing features
        missing_features = set(required_features) - set(numeric_cols)
        if missing_features:
            print(f"Warning: Missing {len(missing_features)} features used during training.")
            print(f"Missing features: {', '.join(missing_features)}")
    
    # Scale features
    df_scaled = pd.DataFrame(
        scaler.transform(df[numeric_cols]), 
        columns=numeric_cols,
        index=df.index
    )
    
    # Add date column back
    df_scaled['date'] = df['date'].values
    
    # Add price for visualization
    if 'close' in df.columns:
        df_scaled['price'] = df['close'].values
    
    return df, df_scaled, numeric_cols

def get_prediction_dates(df, start_date, days=5):
    """Get dates to predict for"""
    # Convert start_date to datetime if it's a string
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    
    # Find dates on or after start_date
    valid_dates = df[df['date'] >= start_date]['date'].unique()
    
    # Take up to 'days' dates
    return sorted(valid_dates)[:days]

def create_sequence(df_scaled, target_date, sequence_length=30, feature_cols=None):
    """Create a sequence for prediction"""
    # Filter columns if specified
    if feature_cols is not None:
        df_features = df_scaled[feature_cols + ['date']]
    else:
        df_features = df_scaled.drop(['price'], axis=1, errors='ignore')
    
    # Find the target date
    target_date_mask = df_features['date'] == target_date
    
    if not target_date_mask.any():
        print(f"Date {target_date} not found in data")
        return None
    
    # Get target date index
    target_idx = df_features[target_date_mask].index[0]
    
    # Check if we have enough history
    if target_idx < sequence_length:
        print(f"Not enough history before {target_date}")
        return None
    
    # Get sequence (excluding date column)
    start_idx = target_idx - sequence_length + 1
    end_idx = target_idx + 1
    sequence_data = df_features.iloc[start_idx:end_idx].drop('date', axis=1).values
    
    return sequence_data

def predict_for_date(df, df_scaled, model, target_date, threshold=0.5, feature_cols=None, sequence_length=30):
    """Make prediction for a specific date"""
    # Create sequence
    sequence = create_sequence(df_scaled, target_date, sequence_length, feature_cols)
    if sequence is None:
        return None
    
    # Reshape for prediction (add batch dimension)
    sequence = np.expand_dims(sequence, axis=0)
    
    # Make prediction
    prediction_prob = model.predict(sequence, verbose=0)[0][0]
    prediction = 1 if prediction_prob >= threshold else 0
    
    # Get price
    target_date_mask = df['date'] == target_date
    price = df.loc[target_date_mask, 'close'].values[0] if 'close' in df.columns else None
    
    return {
        'date': target_date,
        'probability': prediction_prob,
        'prediction': prediction,
        'signal': 'BUY' if prediction == 1 else 'SELL',
        'price': price
    }

def print_results(results):
    """Print prediction results as a table"""
    if not results:
        print("No predictions were made.")
        return
    
    # Print table
    print("\n" + "═" * 65)
    print(" " * 20 + "IMPROVED STOCK PREDICTION RESULTS")
    print("═" * 65)
    print(f"{'DATE':<12} {'SIGNAL':<6} {'PROBABILITY':<12} {'CONFIDENCE':<12} {'PRICE'}")
    print("─" * 65)
    
    for result in results:
        date_str = result['date'].strftime('%Y-%m-%d')
        signal = result['signal']
        prob = result['probability']
        prob_str = f"{prob:.4f}"
        
        # Calculate confidence (distance from threshold)
        threshold = 0.5  # Default
        confidence = abs(prob - threshold) * 2  # Scale to 0-1
        confidence_str = f"{confidence:.2f}"
        
        price = f"${result['price']:.2f}" if result['price'] is not None else "N/A"
        print(f"{date_str:<12} {signal:<6} {prob_str:<12} {confidence_str:<12} {price}")
    
    print("═" * 65)

def plot_predictions(results, threshold):
    """Plot predictions with improved visualization"""
    if not results:
        print("No data to plot")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Plot 1: Price with buy/sell signals
    ax1.plot(results_df['date'], results_df['price'], marker='o', linewidth=2, label='Stock Price')
    
    # Plot buy signals
    buy_points = results_df[results_df['prediction'] == 1]
    if not buy_points.empty:
        ax1.scatter(buy_points['date'], buy_points['price'], marker='^', s=180, 
                   color='green', label='BUY Signal')
    
    # Plot sell signals
    sell_points = results_df[results_df['prediction'] == 0]
    if not sell_points.empty:
        ax1.scatter(sell_points['date'], sell_points['price'], marker='v', s=180, 
                   color='red', label='SELL Signal')
    
    # Format price plot
    ax1.set_title('Stock Price Prediction', fontsize=16)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Plot 2: Probabilities
    ax2.bar(results_df['date'], results_df['probability'], color='blue', alpha=0.6, label='Probability')
    ax2.axhline(y=threshold, color='purple', linestyle='--', alpha=0.8, 
               label=f'Threshold ({threshold:.2f})')
    
    # Add text annotations for probabilities
    for i, row in results_df.iterrows():
        ax2.text(row['date'], row['probability'] + 0.05, f"{row['probability']:.2f}", 
                ha='center', va='bottom', fontsize=9)
    
    # Format probability plot
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('improved_prediction_plot.png')
    print("Prediction plot saved as 'improved_prediction_plot.png'")
    plt.close()

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Load threshold if not specified
    threshold = args.threshold
    if threshold is None:
        try:
            with open('improved_model/optimal_threshold.txt', 'r') as f:
                threshold = float(f.read().strip())
            print(f"Using optimal threshold: {threshold}")
        except (FileNotFoundError, ValueError):
            threshold = 0.5
            print(f"Using default threshold: {threshold}")
    
    # Load and preprocess data
    df, df_scaled, feature_cols = load_and_preprocess_data(args.data, args.scaler)
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model)
    
    # Define target date
    target_date = args.date
    if target_date is None:
        target_date = df['date'].max()
        print(f"No date specified, using latest date: {target_date}")
    else:
        target_date = pd.to_datetime(target_date)
    
    # Get dates to predict for
    prediction_dates = get_prediction_dates(df, target_date, args.days)
    
    if not len(prediction_dates):
        print(f"No valid dates found after {target_date}")
        return
    
    print(f"Making predictions for {len(prediction_dates)} dates starting from {prediction_dates[0].strftime('%Y-%m-%d')}")
    
    # Make predictions
    results = []
    for date in prediction_dates:
        result = predict_for_date(df, df_scaled, model, date, threshold, feature_cols)
        if result:
            results.append(result)
    
    # Print results
    print_results(results)
    
    # Plot if requested
    if args.plot and results:
        plot_predictions(results, threshold)
    
    print("\nPrediction complete!")

if __name__ == "__main__":
    main() 