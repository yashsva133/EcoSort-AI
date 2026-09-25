# Contributing to EcoSort AI

First off, thank you for considering contributing to **EcoSort AI**! It's people like you that help make sustainable AI technology accessible to everyone.

## Code of Conduct
Please ensure respectful, constructive, and inclusive interactions across all discussions and pull requests.

## How Can I Contribute?

### 1. Reporting Bugs
- Ensure the bug was not already reported by searching on GitHub under [Issues](https://github.com/yashsva133/EcoSort-AI/issues).
- If you're unable to find an open issue addressing the problem, open a new one. Be sure to include a clear title, reproduction steps, expected vs. actual behavior, and system environment (OS, Python version, GPU/CPU).

### 2. Suggesting Enhancements
- Open a feature request under [Issues](https://github.com/yashsva133/EcoSort-AI/issues).
- Describe the proposed feature, the use-case (e.g. new waste category, edge hardware support like Coral TPU / Jetson), and any relevant background.

### 3. Pull Requests
1. Fork the repo and create your branch from `main`:
   ```bash
   git checkout -b feature/my-new-feature
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Test your changes locally to ensure the server and edge API function without regression:
   ```bash
   python app.py
   python client_example.py
   ```
4. Commit your changes with clear, descriptive commit messages:
   ```bash
   git commit -m "feat(telemetry): add density constant for tetra-pak packaging"
   ```
5. Push to your fork and submit a Pull Request!

## Development Guidelines
- Follow PEP 8 style guidelines for Python code.
- Keep dependencies minimal to ensure edge devices (e.g., Raspberry Pi) can run cleanly.
- Ensure all new API endpoints return clear JSON with standard error handling.

Thank you for helping make automated waste management cleaner, smarter, and greener!
