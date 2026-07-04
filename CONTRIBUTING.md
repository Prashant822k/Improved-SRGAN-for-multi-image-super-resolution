# Contributing to TerraGAN

Thanks for your interest in TerraGAN! This is an actively maintained research project
and we welcome contributions of all kinds — bug reports, feature suggestions,
documentation improvements, and code.

---

## How to Contribute

### 🐛 Reporting Bugs

1. Check [existing issues](https://github.com/Prashant822k/Improved-SRGAN-for-multi-image-super-resolution/issues)
   to see if the bug has already been reported.
2. If not, open a new issue with:
   - A clear title
   - Steps to reproduce
   - Expected vs. actual behaviour
   - Your environment (OS, Python version, GPU, PyTorch version)

### 💡 Feature Requests

Open an issue with the `enhancement` label describing:
- The problem you're trying to solve
- Your proposed solution
- Why it fits the project's direction (see the [Roadmap](README.md#roadmap))

### 🔧 Code Contributions

1. **Fork** the repository
2. **Create a branch** for your feature/fix:
   ```bash
   git checkout -b feature/deformable-alignment
   ```
3. **Make your changes** — keep commits small and focused
4. **Run the linter** before pushing:
   ```bash
   pip install flake8 black
   flake8 src/ scripts/
   black --check src/ scripts/
   ```
5. **Open a Pull Request** against `main` with a clear description of changes

---

## Code Style

- **Formatter**: [black](https://black.readthedocs.io/) (line length 100)
- **Linter**: [flake8](https://flake8.pycqa.org/) (max line length 100, ignore E203, W503)
- **Docstrings**: Google-style
- **Type hints**: encouraged for all public functions

---

## Project Structure

See the [Project Structure](README.md#project-structure) section of the README.

---

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE).
