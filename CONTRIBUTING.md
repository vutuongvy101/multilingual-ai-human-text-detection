# Contributing to Multilingual AI-Human Text Detection

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/multilingual-ai-human-text-detection.git
   cd multilingual-ai-human-text-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests**
   ```bash
   pytest tests/
   ```

4. **Run linting**
   ```bash
   black src/ tests/
   flake8 src/ tests/
   mypy src/
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

6. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

This project follows these style guidelines:

- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Google-style docstrings**: Documentation

## Testing

- Write unit tests for new functions
- Aim for >80% code coverage
- Test both success and failure cases
- Use descriptive test names

## Documentation

- Update README.md for user-facing changes
- Add docstrings to all public functions
- Update type hints
- Keep examples up to date

## Commit Messages

Follow conventional commit format:
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation
- `style:` - Code style changes
- `refactor:` - Code refactoring
- `test:` - Testing
- `chore:` - Maintenance

## Reporting Issues

When reporting bugs, please include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/logs

## Feature Requests

For feature requests, please:
- Check if the feature already exists
- Describe the use case clearly
- Explain why it's needed
- Consider implementation complexity

## License

By contributing to this project, you agree that your contributions will be licensed under the same MIT License that covers the project.