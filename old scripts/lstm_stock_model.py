import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. Data Exploration & Cleaning
print("Step 1: Loading and exploring data...")
df = pd.read_csv('infolimpioavanzadoTarget.csv')

# Basic info
print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

# Check missing values
missing_values = df.isnull().sum()
print("\nColumns with missing values:")
print(missing_values[missing_values > 0])

# Handle missing values
df = df.ffill()  # Forward fill
df = df.bfill()  # Backward fill

# Handle infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.ffill()
df = df.bfill()

# Ensure date column is datetime
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    # Sort by date
    df.sort_values('date', inplace=True)
    print("\nDate range:", df['date'].min(), "to", df['date'].max())

# 2. Feature Selection & Preprocessing
print("\nStep 2: Feature selection and preprocessing...")

# Identify numeric columns only
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
# Remove TARGET from features if it exists in numeric_cols
if 'TARGET' in numeric_cols:
    numeric_cols.remove('TARGET')
# Remove date from features if it exists in numeric_cols
if 'date' in numeric_cols:
    numeric_cols.remove('date')

print(f"Using {len(numeric_cols)} numeric features")

# Target variable
target = 'TARGET'
if target not in df.columns:
    print(f"WARNING: '{target}' column not found! Please check your data.")
    exit()

# Normalize features
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)

# 3. Sequence Data Preparation
print("\nStep 3: Preparing sequence data...")
# Define sequence length (lookback period)
sequence_length = 30

# Create sequences
X = []
y = []

for i in range(len(df_scaled) - sequence_length):
    X.append(df_scaled.iloc[i:i+sequence_length].values)
    y.append(df[target].iloc[i+sequence_length])

X = np.array(X)
y = np.array(y)

# Train-test split (80% train, 20% test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# 4. LSTM Model Design & Training
print("\nStep 4: Building and training LSTM model...")
# Define model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, len(numeric_cols))),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# 5. Evaluation
print("\nStep 5: Evaluating model...")
# Make predictions
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

# Evaluate
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
plt.savefig('training_history.png')
print("Training history plot saved as 'training_history.png'")

# 6. Save model
print("\nStep 6: Saving model...")
model.save('lstm_stock_model.h5')
print("Model saved as 'lstm_stock_model.h5'")

print("\nLSTM model implementation complete!") 