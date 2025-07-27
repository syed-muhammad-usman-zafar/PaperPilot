#!/usr/bin/env python

import subprocess
import sys
import os

def run_paperpilot():
    print("Starting PaperPilot...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    research_dir = os.path.join(script_dir, "research_coauthor")
    os.chdir(research_dir)
    
    try:
        print("Launching PaperPilot in your browser")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "ui/streamlit_ui.py", 
            "--server.port", "8501",
            "--browser.serverAddress", "localhost"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running PaperPilot: {e}")
        input("Press Enter to exit...")
    except KeyboardInterrupt:
        print("\nPaperPilot stopped by user")

if __name__ == "__main__":
    run_paperpilot()
