# Python App Usage Guide

This guide will help you set up and run the python application

## Prerequisites

1. **Download and Install Python**
   - Visit the official [Python website](https://www.python.org/downloads/) to download and install the latest version of Python for Windows.
   - Ensure that you check the option **"Add Python to PATH"** during installation.

2. **Create a Python Virtual Environment**
   - Open a command prompt (`cmd`) or PowerShell window.
   - Navigate to the folder where your project is located using the `cd` command:
     ```bash
     cd path\to\your\project
     ```
   - Run the following command to create a virtual environment:
     ```bash
     python -m venv myvenv
     ```
     This will create a new folder named `myvenv` where the virtual environment will reside.

3. **Activate the Virtual Environment**
   - To activate the virtual environment, run the following command:
       ```bash
       myvenv\Scripts\activate
       ```
     - You should now see `(myvenv)` at the beginning of the command line prompt, indicating that the environment is active.

---

## Common Issues with Python Environment on Windows

- **Issue:** The virtual environment activation doesn’t work in PowerShell.
  - **Solution:** In PowerShell, you may get a security policy error. To allow scripts to run, execute the following command in PowerShell:
    ```bash
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
    ```
  - Then try activating the environment again using the PowerShell command mentioned earlier.

- **Issue:** The `python` command isn’t recognized.
  - **Solution:** Ensure that Python was added to your system PATH during installation. If it wasn't, you may need to reinstall Python and make sure to select the option **"Add Python to PATH"**.

- **Issue:** Package installation fails due to missing `pip` or outdated `pip`.
  - **Solution:** Upgrade `pip` by running the following command:
    ```bash
    python -m pip install --upgrade pip
    ```

---

## Installing Dependencies

1. **Install Required Packages**
   - Ensure that your virtual environment is active (`(venv)` should be visible in your command prompt).
   - Use the `requirements.txt` file to install all the dependencies:
     ```bash
     pip install -r requirements.txt
     ```

   This will automatically install all the packages listed in `requirements.txt` for your project.

---

## Running the Application

1. **Run the App**
   - Once all dependencies are installed, you can run the main application by executing:
     ```bash
     python main.py
     ```

---

## Additional Notes

- You can deactivate the virtual environment anytime by running:
  ```bash
  deactivate
