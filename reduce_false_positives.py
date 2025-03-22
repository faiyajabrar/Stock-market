import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    precision_score, recall_score, f1_score, roc_curve, auc
)
from tensorflow.keras.models import load_model
import pickle
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Reduce False Positives in Stock Prediction Model')
    parser.add_argument('--data', type=str, default='infolimpioavanzadoTarget.csv', help='CSV data file')
    parser.add_argument('--model', type=str, default='improved_model/improved_lstm_model.h5', help='Model file')
    parser.add_argument('--scaler', type=str, default='feature_scaler.pkl', help='Feature scaler file')
    parser.add_argument('--sequence_length', type=int, default=30, help='Sequence length')
    parser.add_argument('--output_dir', type=str, default='optimized_model', help='Output directory')
    return parser.parse_args()

def engineer_features(df):
    """Engineer features consistently with training"""
    df_new = df.copy()
    
    # Moving Averages
    for window in [5, 20]:
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df_new.columns:
                df_new[f'{col}_ma{window}'] = df_new[col].rolling(window=window).mean()
                df_new[f'{col}_ma{window}_diff'] = df_new[col] - df_new[f'{col}_ma{window}']
    
    # RSI
    if 'close' in df_new.columns:
        delta = df_new['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df_new['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Handle missing values
    df_new = df_new.iloc[20:].reset_index(drop=True)
    df_new = df_new.ffill().bfill()
    df_new.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_new = df_new.ffill().bfill()
    
    return df_new

def load_test_data(file_path, scaler_path, sequence_length=30, test_portion=0.2):
    """Load and preprocess test data"""
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Handle missing values
    df = df.ffill().bfill()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.ffill().bfill()
    
    # Convert date to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
    
    # Engineer features
    df = engineer_features(df)
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if 'TARGET' in numeric_cols: numeric_cols.remove('TARGET')
    if 'date' in numeric_cols: numeric_cols.remove('date')
    
    # Scale features
    df_scaled = pd.DataFrame(
        scaler.transform(df[numeric_cols]), 
        columns=numeric_cols
    )
    
    # Create sequences
    X, y, dates = [], [], []
    target_col = df['TARGET'].values
    
    for i in range(len(df) - sequence_length):
        seq = df_scaled.iloc[i:i+sequence_length].values
        target = target_col[i+sequence_length]
        date = df['date'].iloc[i+sequence_length]
        X.append(seq)
        y.append(target)
        dates.append(date)
    
    X = np.array(X)
    y = np.array(y)
    dates = np.array(dates)
    
    # Get test data (last portion)
    test_size = int(len(X) * test_portion)
    X_test = X[-test_size:]
    y_test = y[-test_size:]
    dates_test = dates[-test_size:]
    
    return X_test, y_test, dates_test, df

def find_optimal_threshold_for_precision(model, X_test, y_test, min_precision=0.6):
    """Find threshold that gives at least the minimum precision"""
    print("Finding optimal threshold to reduce false positives...")
    
    # Get predictions
    y_pred_proba = model.predict(X_test).flatten()
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Find threshold that achieves minimum precision
    valid_indices = np.where(precision >= min_precision)[0]
    
    if len(valid_indices) == 0:
        print(f"No threshold achieves {min_precision} precision. Using maximum precision.")
        best_idx = np.argmax(precision)
    else:
        # Among thresholds that achieve min_precision, find the one with highest recall
        recall_at_valid = recall[valid_indices]
        best_valid_idx = np.argmax(recall_at_valid)
        best_idx = valid_indices[best_valid_idx]
    
    optimal_threshold = thresholds[best_idx - 1] if best_idx > 0 else 0.99
    achieved_precision = precision[best_idx]
    achieved_recall = recall[best_idx]
    
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"Achieved precision: {achieved_precision:.4f}")
    print(f"Corresponding recall: {achieved_recall:.4f}")
    
    return optimal_threshold, precision, recall, thresholds

def create_precision_focused_predictions(model, X_test, y_test, dates_test, threshold):
    """Create predictions with a precision-focused threshold"""
    # Get predictions
    y_pred_proba = model.predict(X_test).flatten()
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Create DataFrame with results
    results_df = pd.DataFrame({
        'date': dates_test,
        'true_label': y_test,
        'predicted_proba': y_pred_proba,
        'predicted_label': y_pred,
        'signal': ['BUY' if p == 1 else 'SELL' for p in y_pred]
    })
    
    return results_df

def analyze_class_distribution(y_test, y_pred):
    """Analyze the distribution of classes before and after prediction"""
    true_buy_pct = np.mean(y_test == 1) * 100
    pred_buy_pct = np.mean(y_pred == 1) * 100
    
    print(f"True BUY signals: {np.sum(y_test == 1)} ({true_buy_pct:.2f}%)")
    print(f"True SELL signals: {np.sum(y_test == 0)} ({100-true_buy_pct:.2f}%)")
    print(f"Predicted BUY signals: {np.sum(y_pred == 1)} ({pred_buy_pct:.2f}%)")
    print(f"Predicted SELL signals: {np.sum(y_pred == 0)} ({100-pred_buy_pct:.2f}%)")

def evaluate_predictions(y_test, y_pred, dates_test, output_dir):
    """Evaluate predictions with a focus on false positives"""
    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # False positive rate
    fpr = fp / (fp + tn)
    
    print("\nDetailed Evaluation Metrics:")
    print(f"True Positives (BUY correctly predicted): {tp}")
    print(f"False Positives (SELL incorrectly predicted as BUY): {fp}")
    print(f"True Negatives (SELL correctly predicted): {tn}")
    print(f"False Negatives (BUY incorrectly predicted as SELL): {fn}")
    print(f"Precision (% of predicted BUY that are correct): {precision:.4f}")
    print(f"Recall (% of actual BUY correctly identified): {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"False Positive Rate: {fpr:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['SELL', 'BUY'], 
                yticklabels=['SELL', 'BUY'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Precision-Focused)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/precision_confusion_matrix.png')
    
    # Save evaluation metrics
    with open(f'{output_dir}/precision_metrics.txt', 'w') as f:
        f.write("PRECISION-FOCUSED EVALUATION\n")
        f.write("============================\n\n")
        f.write(f"True Positives: {tp}\n")
        f.write(f"False Positives: {fp}\n")
        f.write(f"True Negatives: {tn}\n")
        f.write(f"False Negatives: {fn}\n\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"False Positive Rate: {fpr:.4f}\n")

def plot_precision_recall_curve(precision, recall, thresholds, optimal_threshold, output_dir):
    """Plot precision-recall curve with threshold visualization"""
    plt.figure(figsize=(12, 8))
    
    # Plot precision and recall curves against threshold
    plt.subplot(2, 1, 1)
    plt.plot(thresholds, precision[:-1], 'b-', label='Precision')
    plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', 
                label=f'Optimal Threshold ({optimal_threshold:.4f})')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision and Recall vs. Threshold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot precision-recall curve
    plt.subplot(2, 1, 2)
    plt.plot(recall, precision, 'b-')
    
    # Find point for optimal threshold
    idx = np.argmin(np.abs(thresholds - optimal_threshold))
    if idx < len(precision) - 1:
        opt_precision = precision[idx]
        opt_recall = recall[idx]
        plt.plot(opt_recall, opt_precision, 'ro', markersize=8,
                label=f'Optimal ({opt_recall:.2f}, {opt_precision:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/precision_recall_curve.png')

def analyze_false_positives(results_df, df, output_dir):
    """Analyze what causes false positives"""
    fp_cases = results_df[(results_df['true_label'] == 0) & (results_df['predicted_label'] == 1)]
    
    if len(fp_cases) == 0:
        print("No false positives to analyze.")
        return
    
    print(f"\nAnalyzing {len(fp_cases)} false positive cases...")
    
    # Merge with original features
    fp_dates = fp_cases['date'].tolist()
    original_data = df[df['date'].isin(fp_dates)].copy()
    fp_analysis = pd.merge(fp_cases, original_data, on='date')
    
    # Calculate average feature values for false positives
    numeric_cols = original_data.select_dtypes(include=['number']).columns
    fp_avg = fp_analysis[numeric_cols].mean()
    
    # Compare with true negatives (correctly predicted SELL)
    tn_cases = results_df[(results_df['true_label'] == 0) & (results_df['predicted_label'] == 0)]
    tn_dates = tn_cases['date'].tolist()
    tn_data = df[df['date'].isin(tn_dates)].copy()
    tn_avg = tn_data[numeric_cols].mean()
    
    # Calculate differences
    diff = ((fp_avg - tn_avg) / tn_avg * 100).dropna()
    
    # Find most distinctive features
    diff_abs = diff.abs()
    top_features = diff_abs.nlargest(10).index.tolist()
    
    # Plot feature comparison
    plt.figure(figsize=(12, 8))
    
    feature_comparison = pd.DataFrame({
        'False Positives': fp_avg[top_features],
        'True Negatives': tn_avg[top_features]
    })
    
    feature_comparison.plot(kind='bar', figsize=(12, 8))
    plt.title('Feature Comparison: False Positives vs True Negatives')
    plt.xlabel('Features')
    plt.ylabel('Average Value')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/false_positive_analysis.png')
    
    # Save analysis to file
    diff_sorted = diff.sort_values(ascending=False)
    with open(f'{output_dir}/false_positive_analysis.txt', 'w') as f:
        f.write("FALSE POSITIVE ANALYSIS\n")
        f.write("======================\n\n")
        f.write(f"Number of false positives: {len(fp_cases)}\n\n")
        f.write("Top distinctive features (% difference from true negatives):\n")
        for feature in diff_sorted.index:
            f.write(f"{feature}: {diff_sorted[feature]:.2f}%\n")
    
    # Create boxplots for key features
    for i, feature in enumerate(top_features[:5]):  # Top 5 features
        plt.figure(figsize=(10, 6))
        data = [fp_analysis[feature], tn_data[feature]]
        plt.boxplot(data, labels=['False Positives', 'True Negatives'])
        plt.title(f'Distribution of {feature}')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/feature_{i+1}_{feature}_boxplot.png')

def predict_with_precision_model(model, X_test, threshold, dates_test, df, output_dir):
    """Make predictions with the precision-focused model"""
    # Get predictions
    y_pred_proba = model.predict(X_test).flatten()
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Create results for visualization
    results = []
    for i, date in enumerate(dates_test):
        # Find price for the date
        df_date = df[df['date'] == date]
        price = df_date['close'].values[0] if 'close' in df_date.columns and len(df_date) > 0 else None
        
        results.append({
            'date': date,
            'probability': y_pred_proba[i],
            'prediction': y_pred[i],
            'signal': 'BUY' if y_pred[i] == 1 else 'SELL',
            'price': price
        })
    
    results_df = pd.DataFrame(results)
    
    # Plot predictions
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    
    # Plot price
    plt.plot(results_df['date'], results_df['price'], marker='o', label='Price')
    
    # Plot BUY signals
    buy_signals = results_df[results_df['prediction'] == 1]
    if len(buy_signals) > 0:
        plt.scatter(buy_signals['date'], buy_signals['price'], color='green', marker='^', s=100, label='BUY')
    
    plt.title('Precision-Focused Predictions')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot probabilities
    plt.subplot(2, 1, 2)
    plt.bar(results_df['date'], results_df['probability'], alpha=0.7)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.4f})')
    plt.ylabel('Probability')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/precision_predictions.png')
    
    return results_df

def create_precision_threshold_batch(threshold, output_dir):
    """Create batch file to use the precision-focused threshold"""
    batch_content = f"""@echo off
title Precision-Focused Stock Market Predictor

:: Check for Python installation
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

:: Get current date in YYYY-MM-DD format
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /format:list') do set datetime=%%I
set CURRENT_YEAR=%datetime:~0,4%
set CURRENT_MONTH=%datetime:~4,2%
set CURRENT_DAY=%datetime:~6,2%
set CURRENT_DATE=%CURRENT_YEAR%-%CURRENT_MONTH%-%CURRENT_DAY%

:: Default values
set DATE=%CURRENT_DATE%
set DAYS=5
set THRESHOLD={threshold:.4f}
set PLOT_FLAG=--plot

:: Parse command-line arguments
:parse_args
if "%~1"=="" goto :run_prediction
if /i "%~1"=="--date" (
    set DATE=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--days" (
    set DAYS=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--no-plot" (
    set PLOT_FLAG=
    shift
    goto :parse_args
)
shift
goto :parse_args

:run_prediction
echo.
echo ===================================================
echo      PRECISION-FOCUSED STOCK PREDICTION TOOL
echo ===================================================
echo.
echo Running prediction with the following parameters:
echo  - Date: %DATE%
echo  - Days: %DAYS%
echo  - Threshold: %THRESHOLD% (optimized for precision)
echo  - Plot: %PLOT_FLAG:--plot=yes%
echo.
echo ===================================================
echo.

:: Run the prediction
python predict_improved.py --date %DATE% --days %DAYS% --threshold %THRESHOLD% %PLOT_FLAG%

echo.
echo Prediction complete!
echo.

pause
"""
    
    # Write batch file
    with open(f'{output_dir}/predict_precision.bat', 'w') as f:
        f.write(batch_content)
    
    print(f"Created prediction batch file: {output_dir}/predict_precision.bat")

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load test data
    X_test, y_test, dates_test, df = load_test_data(
        args.data, 
        args.scaler, 
        sequence_length=args.sequence_length
    )
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model)
    
    # Find precision-focused threshold
    optimal_threshold, precision, recall, thresholds = find_optimal_threshold_for_precision(
        model, X_test, y_test, min_precision=0.7  # Adjust precision threshold as needed
    )
    
    # Plot precision-recall curve
    plot_precision_recall_curve(precision, recall, thresholds, optimal_threshold, args.output_dir)
    
    # Create precision-focused predictions
    results_df = create_precision_focused_predictions(
        model, X_test, y_test, dates_test, optimal_threshold
    )
    
    # Analyze class distribution
    analyze_class_distribution(y_test, results_df['predicted_label'].values)
    
    # Evaluate predictions
    evaluate_predictions(
        y_test, 
        results_df['predicted_label'].values, 
        dates_test, 
        args.output_dir
    )
    
    # Analyze false positives
    analyze_false_positives(results_df, df, args.output_dir)
    
    # Make sample predictions
    predict_with_precision_model(
        model, X_test, optimal_threshold, dates_test, df, args.output_dir
    )
    
    # Save optimal threshold
    with open(f'{args.output_dir}/precision_threshold.txt', 'w') as f:
        f.write(str(optimal_threshold))
    
    # Create batch file
    create_precision_threshold_batch(optimal_threshold, args.output_dir)
    
    print("\nPrecision optimization completed!")
    print(f"Results saved to {args.output_dir}/")
    print(f"Use {args.output_dir}/predict_precision.bat to make predictions with reduced false positives")

if __name__ == "__main__":
    main() 