.. _contributing:

Contributing to Med-STORM
========================

Thank you for your interest in contributing to Med-STORM! We welcome contributions from the community to help improve this project. This guide will help you get started with contributing.

.. contents:: Table of Contents
   :depth: 3
   :local:
   :backlinks: top

Code of Conduct
---------------

By participating in this project, you agree to abide by our :ref:`code_of_conduct`. Please report any unacceptable behavior to [your-email@example.com].

How to Contribute
-----------------

1. **Reporting Bugs**
   - Check if the bug has already been reported in the `GitHub Issues <https://github.com/your-username/med-storm/issues>`_.
   - If not, create a new issue with a clear title and description.
   - Include steps to reproduce, expected behavior, and actual behavior.
   - Add error messages and your environment details.

2. **Suggesting Enhancements**
   - Open an issue with the "enhancement" label.
   - Describe the feature and why it would be useful.
   - Include any relevant examples or references.

3. **Code Contributions**
   - Fork the repository and create a feature branch.
   - Follow the coding standards and guidelines below.
   - Add tests for new functionality.
   - Update documentation as needed.
   - Submit a pull request with a clear description of changes.

4. **Documentation**
   - Fix typos and improve clarity.
   - Add missing documentation.
   - Translate documentation to other languages.

Setting Up the Development Environment
------------------------------------

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/med-storm.git
   cd med-storm
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -e .[dev]
   pre-commit install
   ```

4. **Run Tests**
   ```bash
   pytest
   ```

Coding Standards
----------------

1. **Code Style**
   - Follow `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ style guide.
   - Use type hints for all function signatures.
   - Write docstrings following the Google style guide.
   - Keep lines under 88 characters (Black's default).

2. **Documentation**
   - Document all public APIs with docstrings.
   - Update relevant documentation when making changes.
   - Use reStructuredText for documentation files.

3. **Testing**
   - Write unit tests for new functionality.
   - Ensure all tests pass before submitting a PR.
   - Add integration tests for complex features.
   - Aim for at least 80% test coverage.

4. **Version Control**
   - Create a new branch for each feature or bugfix.
   - Write clear, descriptive commit messages.
   - Keep commits small and focused.
   - Rebase your branch on the latest main before submitting a PR.

Pull Request Process
-------------------

1. Fork the repository and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints (run `black`, `isort`, `flake8`).
6. Issue that pull request!

After your pull request is merged, you can safely delete your branch.

Development Tools
----------------

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Static type checking
- **pytest**: Testing framework
- **pre-commit**: Git hooks for code quality

Reporting Security Issues
------------------------

Please report security issues to [security@example.com]. We'll address them as soon as possible.

License
-------

By contributing, you agree that your contributions will be licensed under the `MIT License <LICENSE>`_.
