import os
import subprocess
from datetime import datetime

def get_current_date():
    return datetime.today().strftime('%Y-%m-%d')

def main():
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=" * 50)
        print("             STOCK MARKET PREDICTOR TOOL")
        print("=" * 50)
        print("\nPlease select a prediction model:\n")
        print("[1] Balanced Model (Default)")
        print("    - Good balance between buy and sell predictions")
        print("    - Threshold: 0.25\n")
        print("[2] Precision-Focused Model")
        print("    - Minimizes false buy signals")
        print("    - Higher precision, but lower recall")
        print("    - Threshold: 0.303\n")
        print("[3] Exit\n")
        print("=" * 50)
        
        model_choice = input("Enter your choice (1-3): ").strip()
        
        if model_choice == "1":
            model_type = "balanced"
            threshold = 0.25000000000000006
            model_path = "improved_model/improved_lstm_model.h5"
        elif model_choice == "2":
            model_type = "precision"
            threshold = 0.3030081
            model_path = "improved_model/improved_lstm_model.h5"
        elif model_choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")
            input("Press Enter to continue...")
            continue
        
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=" * 50)
        print("             STOCK MARKET PREDICTOR TOOL")
        print("=" * 50)
        print(f"\nSelected model: {model_type}\n")
        
        date = input(f"Date (YYYY-MM-DD) [{get_current_date()}]: ").strip()
        if not date:
            date = get_current_date()
        
        days = input("Number of days to predict [5]: ").strip()
        days = days if days.isdigit() else "5"
        
        plot_input = input("Generate plot? (y/n) [y]: ").strip().lower()
        plot_flag = "--plot" if plot_input != "n" else ""
        
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=" * 50)
        print("             STOCK MARKET PREDICTOR TOOL")
        print("=" * 50)
        print("\nRunning prediction with the following parameters:\n")
        print(f" - Model: {model_type}")
        print(f" - Date: {date}")
        print(f" - Days: {days}")
        print(f" - Threshold: {threshold}")
        print(f" - Plot: {'yes' if plot_flag else 'no'}\n")
        print("=" * 50)
        
        command = ["python", "predict_improved.py", "--date", date, "--days", days, "--threshold", str(threshold), "--model", model_path]
        if plot_flag:
            command.append("--plot")
        
        subprocess.run(command)
        
        input("\nPrediction complete! Press Enter to return to the menu...")

if __name__ == "__main__":
    main()
