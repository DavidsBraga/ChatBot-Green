# IT Setup Instructions for Bootcamp Candidates

## 🎯 Purpose

This document provides step-by-step instructions for IT staff to prepare machines for the Deloitte Tech Experience Bootcamp - Gen AI Challenge.

## 📋 Prerequisites

Each candidate machine must have:

1. **Python 3.12.4** installed and accessible via command line
2. **Internet connection** for pip package installation
3. **Text editor or IDE** (VS Code recommended)
4. **Sufficient disk space** (~2GB for Python environment + packages)

---

## ✅ Step 1: Verify Python Installation

1. Open Command Prompt (cmd) or PowerShell
2. Run the following command:

   ```cmd
   python --version
   ```
3. **Expected output**: `Python 3.12.4`
4. If "python" is not recognized:

   - Ensure Python 3.12.4 is installed from: https://www.python.org/downloads/
   - During installation, **check "Add Python to PATH"**
   - Restart Command Prompt after installation
   - Verify again with `python --version`

---

## 📦 Step 2: Extract Project Files

1. **Locate the provided zip file**: `gen-ai-business-case-recruiting.zip`
2. **Extract to a local folder** (NOT on network drive or OneDrive):

   - Right-click zip file → "Extract All..."
   - Extract to: `C:\Bootcamp\gen-ai-business-case-recruiting\`

3. **Verify extraction**: The folder should contain:

   ```
   gen-ai-business-case-recruiting/
   ├── main.py
   ├── gradio_app.py
   ├── requirements.txt
   ├── .env                  # ⚠️ IMPORTANT: Must be present
   ├── README.md
   ├── IT_INSTRUCTIONS.md
   ├── data/                 # Folder with documents
   └── src/                  # Source code folder
   ```
4. **Critical check**: Ensure `.env` file is present (may be hidden on some systems)

   - Enable "Show hidden files" in File Explorer

---
