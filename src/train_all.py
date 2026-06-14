import subprocess
import sys

def run_training():
    models = ['baseline', 'attention', 'lstm']
    
    for model in models:
        print("\n" + "="*50)
        print(f"Starting Training for: {model.upper()}")
        print("="*50)
        
        cmd = [
            "C:\\Users\\Shreshtha Shrinivas\\AppData\\Local\\Programs\\Python\\Python312\\python.exe",
            "-m", "src.train",
            "--model", model,
            "--epochs", "25",
            "--batch_size", "64",
            "--lr", "1e-4"
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Stream output in real time
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            
        process.wait()
        
        if process.returncode == 0:
            print(f"Training for {model.upper()} finished successfully.")
        else:
            print(f"Training for {model.upper()} failed with code {process.returncode}.")

if __name__ == '__main__':
    run_training()
