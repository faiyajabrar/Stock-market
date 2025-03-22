import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load and prepare data
print("Loading and preprocessing data...")
df = pd.read_csv('infolimpioavanzadoTarget.csv')
df = df.ffill().bfill()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.ffill().bfill()

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)

# Feature selection - use only numeric columns
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
if 'TARGET' in numeric_cols: numeric_cols.remove('TARGET')
if 'date' in numeric_cols: numeric_cols.remove('date')
print(f"Using {len(numeric_cols)} numeric features")

# Normalize features
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)

# Create sequences
sequence_length = 30
X, y = [], []
for i in range(len(df_scaled) - sequence_length):
    X.append(df_scaled.iloc[i:i+sequence_length].values)
    y.append(df['TARGET'].iloc[i+sequence_length])
X, y = np.array(X), np.array(y)

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

# Build and compile model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, len(numeric_cols))),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
print("Training model...")
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
    verbose=1
)

# Evaluate model
print("Evaluating model...")
y_pred = (model.predict(X_test) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.tight_layout()
plt.savefig('optimized_training_history.png')

# Save model
model.save('optimized_lstm_model.h5')
print("Model saved as 'optimized_lstm_model.h5'")

# Prediction function
def predict_stock(model_path='optimized_lstm_model.h5', data=None, last_n_days=5):
    if data is None:
        data = df
    
    model = load_model(model_path)
    results = pd.DataFrame()
    
    # Make predictions
    predictions = model.predict(X)
    binary_preds = (predictions > 0.5).astype(int)
    
    # Create results dataframe
    results['date'] = data.iloc[sequence_length:]['date'].reset_index(drop=True)
    results['close'] = data.iloc[sequence_length:]['close'].reset_index(drop=True)
    results['prediction'] = binary_preds
    results['probability'] = predictions
    
    # Show last n predictions
    print(f"\nLast {last_n_days} predictions:")
    last_n = results.tail(last_n_days)
    for _, row in last_n.iterrows():
        signal = "BUY" if row['prediction'] == 1 else "SELL"
        # Fixed: Removed [0] index from probability as it's already a float
        print(f"{row['date'].strftime('%Y-%m-%d')}: {signal} (prob: {row['probability']:.4f}, close: {row['close']:.2f})")
    
    return results

# Make predictions using the trained model
predict_stock() 