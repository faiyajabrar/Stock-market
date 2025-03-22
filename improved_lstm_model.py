import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    roc_curve, precision_recall_curve, auc, 
    balanced_accuracy_score, matthews_corrcoef, f1_score
)
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, LSTM, Dropout, BatchNormalization, Input,
    Bidirectional, Concatenate, GlobalAveragePooling1D, 
    Conv1D, MaxPooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard
)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l1_l2
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
import pickle
import os
import argparse
from datetime import datetime

# Focal Loss implementation for imbalanced classification
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, reduction=tf.keras.losses.Reduction.AUTO, name='focal_loss'):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = self.alpha * tf.math.pow(1 - y_pred, self.gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Improved LSTM Stock Market Predictor')
    parser.add_argument('--data', type=str, default='infolimpioavanzadoTarget.csv', help='CSV data file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--sequence_length', type=int, default=30, help='Sequence length')
    parser.add_argument('--output_dir', type=str, default='improved_model', help='Output directory')
    parser.add_argument('--model_type', type=str, default='ensemble', 
                       choices=['simple', 'bidirectional', 'cnn_lstm', 'ensemble'], 
                       help='Type of model to train')
    parser.add_argument('--balance_method', type=str, default='smoteenn', 
                       choices=['none', 'smote', 'smoteenn', 'class_weight', 'focal_loss'], 
                       help='Method to handle class imbalance')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size (ratio)')
    return parser.parse_args()

def load_and_preprocess_data(file_path):
    """Load and preprocess data with enhanced preprocessing"""
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
    for window in [5, 10, 20, 50]:
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df_new.columns:
                df_new[f'{col}_ma{window}'] = df_new[col].rolling(window=window).mean()
                df_new[f'{col}_ma{window}_diff'] = df_new[col] - df_new[f'{col}_ma{window}']
                df_new[f'{col}_ma{window}_pct'] = df_new[f'{col}_ma{window}_diff'] / df_new[col] * 100
    
    # Volatility indicators
    if all(col in df_new.columns for col in ['high', 'low', 'close']):
        # True Range
        df_new['tr1'] = abs(df_new['high'] - df_new['low'])
        df_new['tr2'] = abs(df_new['high'] - df_new['close'].shift(1))
        df_new['tr3'] = abs(df_new['low'] - df_new['close'].shift(1))
        df_new['true_range'] = df_new[['tr1', 'tr2', 'tr3']].max(axis=1)
        df_new.drop(['tr1', 'tr2', 'tr3'], axis=1, inplace=True)
        
        # Average True Range (ATR)
        df_new['atr_14'] = df_new['true_range'].rolling(window=14).mean()
        
        # Bollinger Bands
        for window in [20]:
            df_new[f'bb_middle_{window}'] = df_new['close'].rolling(window=window).mean()
            df_new[f'bb_std_{window}'] = df_new['close'].rolling(window=window).std()
            df_new[f'bb_upper_{window}'] = df_new[f'bb_middle_{window}'] + 2 * df_new[f'bb_std_{window}']
            df_new[f'bb_lower_{window}'] = df_new[f'bb_middle_{window}'] - 2 * df_new[f'bb_std_{window}']
            df_new[f'bb_width_{window}'] = (df_new[f'bb_upper_{window}'] - df_new[f'bb_lower_{window}']) / df_new[f'bb_middle_{window}']
            df_new[f'bb_pct_{window}'] = (df_new['close'] - df_new[f'bb_lower_{window}']) / (df_new[f'bb_upper_{window}'] - df_new[f'bb_lower_{window}'])
    
    # Momentum indicators
    if 'close' in df_new.columns:
        # RSI (Relative Strength Index)
        delta = df_new['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        for window in [14]:
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            
            rs = avg_gain / avg_loss
            df_new[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        # Price Rate of Change
        for window in [5, 10, 20]:
            df_new[f'roc_{window}'] = df_new['close'].pct_change(periods=window) * 100
    
    # Handle missing values created by rolling calculations
    df_new = df_new.iloc[50:].reset_index(drop=True)  # Remove rows with NaNs from rolling window calcs
    
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

def prepare_sequences(df, sequence_length=30, test_size=0.2, balance_method='smoteenn'):
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
    if balance_method == 'smote':
        print("Applying SMOTE to balance training data...")
        smote = SMOTE(random_state=42)
        # Reshape to 2D for SMOTE
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        X_train_2d, y_train = smote.fit_resample(X_train_2d, y_train)
        # Reshape back to 3D
        X_train = X_train_2d.reshape(X_train_2d.shape[0], sequence_length, len(numeric_cols))
        
    elif balance_method == 'smoteenn':
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
    
    return X_train, X_test, y_train, y_test

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

def create_simple_lstm_model(input_shape, learning_rate=0.001):
    """Create a simple LSTM model with regularization"""
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True, 
             kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        Dropout(0.4),
        LSTM(32, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def create_bidirectional_lstm_model(input_shape, learning_rate=0.001):
    """Create a bidirectional LSTM model for better sequence learning"""
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)), 
                     input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.4),
        Bidirectional(LSTM(32, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def create_cnn_lstm_model(input_shape, learning_rate=0.001):
    """Create a CNN-LSTM hybrid model for feature extraction and sequence learning"""
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', 
               input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def create_ensemble_model(input_shape, learning_rate=0.001):
    """Create an ensemble model combining different architectures"""
    # Input layer
    input_layer = Input(shape=input_shape)
    
    # LSTM branch
    lstm = LSTM(64, return_sequences=True)(input_layer)
    lstm = BatchNormalization()(lstm)
    lstm = Dropout(0.3)(lstm)
    lstm = LSTM(32)(lstm)
    lstm = BatchNormalization()(lstm)
    lstm = Dense(16, activation='relu')(lstm)
    
    # Bidirectional LSTM branch
    bilstm = Bidirectional(LSTM(64, return_sequences=True))(input_layer)
    bilstm = BatchNormalization()(bilstm)
    bilstm = Dropout(0.3)(bilstm)
    bilstm = Bidirectional(LSTM(32))(bilstm)
    bilstm = BatchNormalization()(bilstm)
    bilstm = Dense(16, activation='relu')(bilstm)
    
    # CNN branch
    cnn = Conv1D(64, kernel_size=3, activation='relu', padding='same')(input_layer)
    cnn = BatchNormalization()(cnn)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = Conv1D(128, kernel_size=3, activation='relu', padding='same')(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = GlobalAveragePooling1D()(cnn)
    cnn = Dense(16, activation='relu')(cnn)
    
    # Combine branches
    combined = Concatenate()([lstm, bilstm, cnn])
    combined = Dense(32, activation='relu')(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.3)(combined)
    combined = Dense(16, activation='relu')(combined)
    output = Dense(1, activation='sigmoid')(combined)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output)
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def plot_training_history(history, output_dir):
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
    
    # Save history to file
    pd.DataFrame(history.history).to_csv(f'{output_dir}/training_history.csv', index=False)

def evaluate_model(model, X_test, y_test, output_dir, threshold=0.5):
    """Evaluate model with comprehensive metrics"""
    # Get predictions
    y_pred_proba = model.predict(X_test).flatten()
    
    # Try different thresholds to find optimal one
    thresholds = np.arange(0.1, 0.9, 0.05)
    f1_scores = []
    balanced_accuracy_scores = []
    mcc_scores = []
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        f1 = f1_score(y_test, y_pred)
        ba = balanced_accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        f1_scores.append(f1)
        balanced_accuracy_scores.append(ba)
        mcc_scores.append(mcc)
    
    # Find optimal threshold based on MCC
    optimal_idx = np.argmax(mcc_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"Optimal threshold based on MCC: {optimal_threshold:.2f}")
    
    # Use optimal threshold for final evaluation
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=['SELL', 'BUY'])
    print("\nClassification Report:")
    print(report)
    
    # Save report to file
    with open(f'{output_dir}/classification_report.txt', 'w') as f:
        f.write(f"Optimal threshold: {optimal_threshold:.2f}\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n")
        f.write(f"Matthews Correlation Coefficient: {mcc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['SELL', 'BUY'], 
                yticklabels=['SELL', 'BUY'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (Threshold: {optimal_threshold:.2f})')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    
    # ROC curve
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/roc_curve.png')
    
    # Precision-Recall curve
    plt.figure(figsize=(10, 8))
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/pr_curve.png')
    
    # Threshold vs Metrics plot
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, f1_scores, 'o-', label='F1 Score')
    plt.plot(thresholds, balanced_accuracy_scores, 'o-', label='Balanced Accuracy')
    plt.plot(thresholds, mcc_scores, 'o-', label='Matthews Correlation Coefficient')
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', 
                label=f'Optimal Threshold ({optimal_threshold:.2f})')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Performance Metrics vs. Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/threshold_metrics.png')
    
    # Save threshold-to-metrics mapping
    threshold_metrics = pd.DataFrame({
        'threshold': thresholds,
        'f1_score': f1_scores,
        'balanced_accuracy': balanced_accuracy_scores,
        'mcc': mcc_scores
    })
    threshold_metrics.to_csv(f'{output_dir}/threshold_metrics.csv', index=False)
    
    print(f"Evaluation plots saved to {output_dir}/")
    
    return optimal_threshold

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load data
    df = load_and_preprocess_data(args.data)
    
    # Engineer features
    df_engineered = engineer_features(df)
    
    # Prepare sequences
    X_train, X_test, y_train, y_test = prepare_sequences(
        df_engineered, 
        sequence_length=args.sequence_length,
        test_size=args.test_size,
        balance_method=args.balance_method if args.balance_method != 'focal_loss' and args.balance_method != 'class_weight' else 'none'
    )
    
    # Get input shape
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Create model based on model_type
    print(f"Creating {args.model_type} model...")
    if args.model_type == 'simple':
        model = create_simple_lstm_model(input_shape, args.learning_rate)
    elif args.model_type == 'bidirectional':
        model = create_bidirectional_lstm_model(input_shape, args.learning_rate)
    elif args.model_type == 'cnn_lstm':
        model = create_cnn_lstm_model(input_shape, args.learning_rate)
    else:  # ensemble
        model = create_ensemble_model(input_shape, args.learning_rate)
    
    # Print model summary
    model.summary()
    
    # If using focal loss, recompile the model
    if args.balance_method == 'focal_loss':
        print("Using Focal Loss for class imbalance...")
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        model.compile(
            optimizer=Adam(learning_rate=args.learning_rate),
            loss=focal_loss,
            metrics=['accuracy']
        )
    
    # Prepare callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=f'{args.output_dir}/model_checkpoint.h5',
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
        ),
        TensorBoard(
            log_dir=f'{args.output_dir}/logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1
        )
    ]
    
    # Calculate class weights if using class_weight method
    class_weights = None
    if args.balance_method == 'class_weight':
        print("Using class weights for class imbalance...")
        class_weights = get_class_weights(y_train)
        print(f"Class weights: {class_weights}")
    
    # Train model
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history, args.output_dir)
    
    # Evaluate model
    optimal_threshold = evaluate_model(model, X_test, y_test, args.output_dir)
    
    # Save model
    model.save(f'{args.output_dir}/improved_lstm_model.h5')
    print(f"Model saved to {args.output_dir}/improved_lstm_model.h5")
    
    # Save optimal threshold
    with open(f'{args.output_dir}/optimal_threshold.txt', 'w') as f:
        f.write(str(optimal_threshold))
    
    # Save model architecture diagram
    try:
        plot_model(model, to_file=f'{args.output_dir}/model_architecture.png', show_shapes=True, show_layer_names=True)
        print(f"Model architecture diagram saved to {args.output_dir}/model_architecture.png")
    except Exception as e:
        print(f"Could not generate model architecture diagram: {e}")
    
    print("\nTraining and evaluation completed!")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    main() 