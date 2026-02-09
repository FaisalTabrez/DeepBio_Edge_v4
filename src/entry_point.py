import sys
import os
from streamlit.web import cli as stcli

def main():
    """
    Entry point for the PyInstaller executable.
    Redirects stdout/stderr to the logger and launches Streamlit.
    """
    # Set PYTHONPATH if needed, though PyInstaller handles this internally.
    # We construct the command to run the app.py
    
    sys.argv = [
        "streamlit",
        "run",
        os.path.join(os.path.dirname(__file__), "interface", "app.py"),
        "--global.developmentMode=false",
        "--server.headless=true",
    ]
    
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
