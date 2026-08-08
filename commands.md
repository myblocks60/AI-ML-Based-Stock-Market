# Commands to run the project

Follow these steps to run the NSE Stock Screener & Analyzer application.

## 1. Create a Virtual Environment

```powershell
python -m venv venv
```

## 2. Activate the Virtual Environment

### Windows (PowerShell)
```powershell
.\venv\Scripts\Activate.ps1
```

### Windows (Command Prompt)
```cmd
.\venv\Scripts\activate.bat
```

### macOS / Linux
```bash
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Streamlit Dashboard

```bash
streamlit run app.py --server.port 8501
```
