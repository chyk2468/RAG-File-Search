@echo off
TITLE Local RAG System Launcher
CLS

:MENU
ECHO ================================================
ECHO           LOCAL RAG SYSTEM LAUNCHER
ECHO ================================================
ECHO.
ECHO  1. Run Streamlit UI (Recommended)
ECHO  2. Run CLI Mode
ECHO  3. Exit
ECHO.
ECHO ================================================
SET /P M=Select an option (1-3): 

IF %M%==1 GOTO STREAMLIT
IF %M%==2 GOTO CLI
IF %M%==3 GOTO EOF

:STREAMLIT
ECHO Launching Streamlit UI...
python -m streamlit run app.py
PAUSE
GOTO MENU

:CLI
ECHO Launching CLI Mode...
python main.py
PAUSE
GOTO MENU

:EOF
EXIT