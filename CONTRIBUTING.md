# Contributing to Document Layout Analysis

Thank you for your interest in contributing to the Document Layout Analysis project! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic knowledge of machine learning and document processing

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/document-layout-analysis.git
   cd document-layout-analysis
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

4. **Install Development Tools**
   ```bash
   pip install black flake8 pytest pytest-cov
   ```

## 📋 Contribution Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions small and focused

### Commit Messages

Use clear, descriptive commit messages:
```
feat: add YOLOv8 model support
fix: resolve OCR text extraction issue
docs: update API documentation
test: add unit tests for batch processing
```

### Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code following the style guidelines
   - Add tests for new functionality
   - Update documentation if needed

3. **Run Tests**
   ```bash
   python -m pytest test_suite.py -v
   python -m pytest --cov=modern_layout_analyzer
   ```

4. **Code Formatting**
   ```bash
   black .
   flake8 .
   ```

5. **Submit Pull Request**
   - Provide clear description of changes
   - Reference any related issues
   - Include screenshots for UI changes

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest test_suite.py -v

# Run specific test categories
python -m pytest test_suite.py::TestDocumentLayoutAnalyzer -v
python -m pytest test_suite.py::TestAPIIntegration -v

# Run with coverage
python -m pytest test_suite.py --cov=modern_layout_analyzer --cov-report=html
```

### Writing Tests

- Write unit tests for new functions
- Add integration tests for API endpoints
- Include performance tests for critical paths
- Mock external dependencies

## 📚 Documentation

### Code Documentation

- Use Google-style docstrings
- Include type hints
- Document complex algorithms
- Provide usage examples

### API Documentation

- Update API documentation for new endpoints
- Include request/response examples
- Document error codes and messages

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment Information**
   - Python version
   - Operating system
   - Package versions

2. **Steps to Reproduce**
   - Clear, numbered steps
   - Sample input data
   - Expected vs actual behavior

3. **Additional Context**
   - Error messages
   - Screenshots
   - Log files

## 💡 Feature Requests

For feature requests, please:

1. **Check Existing Issues**
   - Search for similar requests
   - Check if already planned

2. **Provide Details**
   - Clear description of the feature
   - Use cases and benefits
   - Implementation suggestions (optional)

## 🏗 Architecture Guidelines

### Adding New Models

1. **Create Model Class**
   ```python
   class YourModel:
       def __init__(self):
           # Initialize model
       
       def detect(self, image):
           # Implement detection logic
           return layout_blocks
   ```

2. **Integrate with Analyzer**
   ```python
   def _analyze_with_your_model(self, image):
       # Add to ModernDocumentLayoutAnalyzer
   ```

3. **Add Tests**
   ```python
   def test_your_model_integration(self):
       # Test the new model
   ```

### Adding New Features

1. **Plan the Feature**
   - Define requirements
   - Consider API design
   - Plan database changes

2. **Implement Incrementally**
   - Start with core functionality
   - Add tests as you go
   - Update documentation

3. **Consider Backward Compatibility**
   - Maintain existing APIs
   - Add deprecation warnings
   - Provide migration guides

## 🔧 Development Tools

### Code Quality Tools

- **Black**: Code formatting
- **Flake8**: Linting
- **Pytest**: Testing framework
- **Coverage**: Test coverage

### IDE Configuration

#### VS Code
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black"
}
```

#### PyCharm
- Configure Python interpreter to use virtual environment
- Enable code inspection
- Set up run configurations for tests

## 📊 Performance Considerations

### Optimization Guidelines

1. **Image Processing**
   - Resize large images before analysis
   - Use appropriate image formats
   - Consider memory usage

2. **Model Loading**
   - Lazy load models
   - Cache loaded models
   - Consider model size

3. **Database Operations**
   - Use batch operations
   - Index frequently queried columns
   - Consider connection pooling

## 🚀 Release Process

### Version Numbering

We use semantic versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version number updated
- [ ] CHANGELOG.md updated
- [ ] Release notes prepared

## 🤝 Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Provide constructive feedback
- Focus on the code, not the person

### Communication

- Use clear, professional language
- Be patient with questions
- Provide helpful responses
- Stay on topic

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and ideas
- **Email**: For private matters

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to the Document Layout Analysis project!
