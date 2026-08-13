# Project Agent Rules

## Streamlit

Do not change the existing Streamlit setup.

Do not kill Python processes to restart Streamlit.

Do not use:
- `Get-Process python | Stop-Process`
- `taskkill` to kill Python
- `Start-Process` followed by waiting for Streamlit
- a command that starts Streamlit and waits for it to exit

Streamlit is a long-running server.

If Streamlit is already running, leave it running and use the existing server.

For tests, use short-lived commands such as:

`.venv\Scripts\python.exe -m pytest`

Only start, stop, or restart Streamlit if the user explicitly asks.

Do not modify project configuration just to work around Streamlit startup.