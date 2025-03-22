import os
import subprocess

def main():
    # Create output directory
    output_dir = 'precision_model'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run the false positive reduction script
    print("Running false positive reduction script...")
    subprocess.run([
        'python', 'reduce_false_positives.py',
        '--model', 'improved_model/improved_lstm_model.h5',
        '--data', 'infolimpioavanzadoTarget.csv',
        '--scaler', 'feature_scaler.pkl',
        '--sequence_length', '30',
        '--output_dir', output_dir
    ])
    
    print(f"\nPrecision optimization completed!")
    print(f"Results saved to {output_dir}/")
    print(f"Use {output_dir}/predict_precision.bat to make predictions with reduced false positives")

if __name__ == "__main__":
    main() 