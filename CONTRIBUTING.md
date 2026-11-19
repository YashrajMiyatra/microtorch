# Contributing to MicroTorch

Thank you for your interest in contributing to **MicroTorch**! We welcome contributions ranging from bug fixes and documentation improvements to new neural network layers and optimizers.

## Development Workflow

1. **Fork the repository** on GitHub.
2. **Clone your fork** and create a feature branch:
   ```bash
   git checkout -b feature/my-enhancement
   ```
3. **Run the test suite** to ensure existing functionality remains intact:
   ```bash
   python3 -m unittest discover tests -v
   ```
4. **Implement your changes** with accompanying unit tests in the `tests/` directory.
5. **Commit your changes** following conventional commit messages:
   ```bash
   git commit -m "feat(nn): implement Dropout layer"
   ```
6. **Submit a Pull Request** describing your implementation and test coverage.

## Code Standards
- Zero external dependencies: do not introduce third-party libraries into core modules.
- Strict type annotations on public functions.
- 100% unit test coverage for new operators or mathematical routines.
