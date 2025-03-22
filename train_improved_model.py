import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, matthews_corrcoef
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from imblearn.combine import SMOTEENN
import pickle
import os
import argparse

def load_and_preprocess_data(file_path='infolimpioavanzadoTarget.csv'):
    """Load and preprocess data"""
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
    
    # Print data statistics
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Check target distribution
    if 'TARGET' in df.columns:
        target_counts = df['TARGET'].value_counts()
        print("Target distribution:")
        for value, count in target_counts.items():
            print(f"  {value}: {count} ({count/len(df)*100:.2f}%)")
    
    return df

def engineer_features(df):
    """Engineer additional features to help with predictions"""
    print("Engineering additional features...")
    
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
    
    # Handle missing values created by rolling calculations
    df_new = df_new.iloc[20:].reset_index(drop=True)  # Remove rows with NaNs from rolling window calcs
    
    # Handle remaining NA values
    df_new = df_new.ffill().bfill()
    df_new.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_new = df_new.ffill().bfill()
    
    # List all the newly created features
    original_cols = set(df.columns)
    new_cols = set(df_new.columns) - original_cols
    print(f"Added {len(new_cols)} new features:")
    print(", ".join(sorted(new_cols)))
    
    return df_new

def prepare_sequences(df, sequence_length=30, test_size=0.2, balance=True):
    """Prepare sequences for training with balancing techniques"""
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if 'TARGET' in numeric_cols: numeric_cols.remove('TARGET')
    if 'date' in numeric_cols: numeric_cols.remove('date')
    
    print(f"Using {len(numeric_cols)} numeric features")
    
    # Normalize features
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[numeric_cols]), 
        columns=numeric_cols
    )
    
    # Save the scaler for later use in prediction
    with open('feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Create sequences and corresponding targets
    X, y = [], []
    target_col = df['TARGET'].values
    
    for i in range(len(df) - sequence_length):
        seq = df_scaled.iloc[i:i+sequence_length].values
        target = target_col[i+sequence_length]  # Target for the next day after sequence
        X.append(seq)
        y.append(target)
    
    X = np.array(X)
    y = np.array(y)
    
    # Print sequence shape information
    print(f"Sequence shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Positive examples (BUY): {sum(y == 1)} ({sum(y == 1)/len(y)*100:.2f}%)")
    print(f"Negative examples (SELL): {sum(y == 0)} ({sum(y == 0)/len(y)*100:.2f}%)")
    
    # Split into train and validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    # Balance the training data if requested
    if balance:
        print("Applying SMOTEENN to balance training data...")
        smoteenn = SMOTEENN(random_state=42)
        # Reshape to 2D for SMOTEENN
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        X_train_2d, y_train = smoteenn.fit_resample(X_train_2d, y_train)
        # Reshape back to 3D
        X_train = X_train_2d.reshape(X_train_2d.shape[0], sequence_length, len(numeric_cols))
    
    # Print rebalanced dataset information
    print("After balancing:")
    print(f"Training sequence shape: {X_train.shape}")
    print(f"Positive examples (BUY): {sum(y_train == 1)} ({sum(y_train == 1)/len(y_train)*100:.2f}%)")
    print(f"Negative examples (SELL): {sum(y_train == 0)} ({sum(y_train == 0)/len(y_train)*100:.2f}%)")
    
    return X_train, X_test, y_train, y_test, numeric_cols

def get_class_weights(y_train):
    """Calculate class weights for imbalanced dataset"""
    # Count samples in each class
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    total = neg_count + pos_count
    
    # Calculate weights inversely proportional to class frequencies
    weight_for_0 = (1 / neg_count) * (total / 2)
    weight_for_1 = (1 / pos_count) * (total / 2)
    
    return {0: weight_for_0, 1: weight_for_1}

def create_improved_model(input_shape, learning_rate=0.001):
    """Create an improved bidirectional LSTM model"""
    model = Sequential([
        # First Bidirectional LSTM layer
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.4),
        
        # Second Bidirectional LSTM layer
        Bidirectional(LSTM(32)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Dense output layers
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile with Adam optimizer
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def plot_training_history(history, output_dir='improved_model'):
    """Plot training history metrics"""
    plt.figure(figsize=(15, 6))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_history.png')
    print(f"Training history saved to {output_dir}/training_history.png")

def find_optimal_threshold(model, X_test, y_test):
    """Find the optimal threshold for predictions"""
    # Get predictions
    y_pred_proba = model.predict(X_test).flatten()
    
    # Try different thresholds to find optimal one
    thresholds = np.arange(0.1, 0.9, 0.05)
    metrics = []
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        
        # Calculate various metrics
        acc = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # Get confusion matrix values
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Store results
        metrics.append({
            'threshold': thresh,
            'accuracy': acc,
            'balanced_accuracy': balanced_acc,
            'mcc': mcc,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        })
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics)
    
    # Find threshold with highest MCC
    best_mcc_idx = metrics_df['mcc'].idxmax()
    best_mcc_threshold = metrics_df.loc[best_mcc_idx, 'threshold']
    
    # Find threshold with highest balanced accuracy
    best_ba_idx = metrics_df['balanced_accuracy'].idxmax()
    best_ba_threshold = metrics_df.loc[best_ba_idx, 'threshold']
    
    # Plot metrics vs threshold
    plt.figure(figsize=(12, 8))
    plt.plot(metrics_df['threshold'], metrics_df['accuracy'], 'o-', label='Accuracy')
    plt.plot(metrics_df['threshold'], metrics_df['balanced_accuracy'], 'o-', label='Balanced Accuracy')
    plt.plot(metrics_df['threshold'], metrics_df['mcc'], 'o-', label='Matthews Correlation Coefficient')
    plt.axvline(x=best_mcc_threshold, color='red', linestyle='--', 
                label=f'Best MCC Threshold ({best_mcc_threshold:.2f})')
    plt.axvline(x=best_ba_threshold, color='green', linestyle='--', 
                label=f'Best Balanced Acc Threshold ({best_ba_threshold:.2f})')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Performance Metrics vs. Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('improved_model/threshold_metrics.png')
    
    print(f"Best threshold by MCC: {best_mcc_threshold:.2f}")
    print(f"Best threshold by balanced accuracy: {best_ba_threshold:.2f}")
    
    # Use MCC as the main metric for threshold selection
    optimal_threshold = best_mcc_threshold
    
    # Save metrics to CSV
    metrics_df.to_csv('improved_model/threshold_metrics.csv', index=False)
    
    return optimal_threshold, y_pred_proba

def evaluate_model(model, X_test, y_test, output_dir='improved_model'):
    """Evaluate model with comprehensive metrics"""
    # Find optimal threshold
    optimal_threshold, y_pred_proba = find_optimal_threshold(model, X_test, y_test)
    
    # Generate predictions using optimal threshold
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    print("\nModel Evaluation with optimal threshold:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=['SELL', 'BUY'])
    print("\nClassification Report:")
    print(report)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"True Negatives (SELL correctly predicted): {tn}")
    print(f"False Positives (SELL incorrectly predicted as BUY): {fp}")
    print(f"False Negatives (BUY incorrectly predicted as SELL): {fn}")
    print(f"True Positives (BUY correctly predicted): {tp}")
    
    # Calculate additional metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['SELL', 'BUY'], 
                yticklabels=['SELL', 'BUY'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (Threshold: {optimal_threshold:.2f})')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    
    # Save evaluation results
    with open(f'{output_dir}/evaluation_results.txt', 'w') as f:
        f.write(f"Optimal threshold: {optimal_threshold:.2f}\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n")
        f.write(f"Matthews Correlation Coefficient: {mcc:.4f}\n\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    return optimal_threshold

def main():
    # Create output directory
    output_dir = 'improved_model'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set parameters
    sequence_length = 30
    test_size = 0.2
    learning_rate = 0.001
    epochs = 50
    batch_size = 32
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Engineer features
    df_engineered = engineer_features(df)
    
    # Prepare sequences with balancing
    X_train, X_test, y_train, y_test, features = prepare_sequences(
        df_engineered, 
        sequence_length=sequence_length,
        test_size=test_size,
        balance=True
    )
    
    # Save feature names
    with open(f'{output_dir}/feature_names.txt', 'w') as f:
        f.write('\n'.join(features))
    
    # Get input shape
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Create improved model
    model = create_improved_model(input_shape, learning_rate)
    
    # Print model summary
    model.summary()
    
    # Calculate class weights as additional balancing mechanism
    class_weights = get_class_weights(y_train)
    print(f"Class weights: {class_weights}")
    
    # Prepare callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=f'{output_dir}/improved_lstm_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train model
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history, output_dir)
    
    # Evaluate model and find optimal threshold
    optimal_threshold = evaluate_model(model, X_test, y_test, output_dir)
    
    # Save optimal threshold
    with open(f'{output_dir}/optimal_threshold.txt', 'w') as f:
        f.write(str(optimal_threshold))
    
    print("\nTraining and evaluation completed!")
    print(f"Improved model saved to {output_dir}/improved_lstm_model.h5")
    print(f"Optimal threshold: {optimal_threshold:.2f}")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    main() 