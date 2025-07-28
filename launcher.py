import subprocess
import time
import sys
import os

# Define the path to the script to be monitored
# It's assumed 'main.py' is in the same directory as this watchdog script.
MAIN_SCRIPT_PATH = "main.py"

def run_main_script():
    """
    Runs the main.py script as a subprocess and returns the process object.
    """
    print(f"Starting {MAIN_SCRIPT_PATH}...")
    try:
        # Use sys.executable to ensure the same Python interpreter is used
        # that is running the watchdog script.
        # We use Popen without shell=True for better control and security.
        process = subprocess.Popen([sys.executable, MAIN_SCRIPT_PATH])
        print(f"{MAIN_SCRIPT_PATH} started with PID: {process.pid}")
        return process
    except FileNotFoundError:
        print(f"Error: {sys.executable} not found. Make sure Python is in your PATH.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error starting {MAIN_SCRIPT_PATH}: {e}", file=sys.stderr)
        return None

def main():
    """
    Main function to continuously monitor and restart main.py if it crashes.
    """
    print(f"Watchdog script started. Monitoring {MAIN_SCRIPT_PATH}...")

    # Check if main.py exists before starting the monitoring loop
    if not os.path.exists(MAIN_SCRIPT_PATH):
        print(f"Error: {MAIN_SCRIPT_PATH} not found in the current directory.", file=sys.stderr)
        print("Please make sure 'main.py' exists before running this watchdog script.", file=sys.stderr)
        sys.exit(1)

    while True:
        process = run_main_script()
        if process is None:
            # If run_main_script failed to start the process, wait and retry
            time.sleep(5) # Wait a bit before trying to restart
            continue

        # Wait for the process to terminate and get its return code
        return_code = process.wait()

        if return_code != 0:
            print(f"{MAIN_SCRIPT_PATH} crashed with exit code {return_code}. Restarting in 3 seconds...")
            # Add a small delay before restarting to prevent a busy loop
            # if the script crashes immediately upon start.
            time.sleep(3)
        else:
            print(f"{MAIN_SCRIPT_PATH} exited gracefully with exit code {return_code}. Exiting watchdog.")
            # If main.py exits gracefully, break the loop to terminate the watchdog script
            break

if __name__ == "__main__":
    main()
