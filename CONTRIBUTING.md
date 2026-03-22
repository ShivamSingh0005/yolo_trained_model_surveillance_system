# Contributing to Surveillance System

Thank you for your interest in contributing! Here are some guidelines.

## How to Contribute

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/surveillance-system.git
cd surveillance-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Code Style

- Follow PEP 8 guidelines
- Add docstrings to functions and classes
- Keep functions focused and modular
- Add comments for complex logic

## Testing

Before submitting a PR:
```bash
# Quick test
python quick_start.py train
python quick_start.py metrics
```

## Reporting Issues

- Use the issue tracker
- Provide clear description
- Include error messages and logs
- Specify your environment (OS, Python version, GPU)

## Feature Requests

We welcome feature requests! Please:
- Check if it already exists
- Provide clear use case
- Explain expected behavior

## Questions?

Feel free to open an issue for questions or discussions.
