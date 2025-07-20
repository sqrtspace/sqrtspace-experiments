#!/usr/bin/env python3
"""
Launch the Space-Time Tradeoffs Dashboard
"""

import subprocess
import sys
import os

def main():
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("Streamlit not found. Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Launch the dashboard
    print("Launching Space-Time Tradeoffs Dashboard...")
    print("Opening in your default browser...")
    
    os.system("streamlit run app.py")

if __name__ == "__main__":
    main()