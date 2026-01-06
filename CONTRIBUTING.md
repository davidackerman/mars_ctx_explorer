# Contributing to Mars Biosignature Detection

Thank you for your interest in contributing to this project!

## Development Setup

1. **Install Pixi**: Follow instructions at https://pixi.sh
2. **Clone and install**:
   ```bash
   git clone https://github.com/yourusername/mars_astrobio.git
   cd mars_astrobio
   pixi install
   pixi shell
   ```
3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Code Standards

- **Formatting**: Black (line length 100)
- **Linting**: Ruff
- **Type hints**: Use type annotations where appropriate
- **Docstrings**: NumPy style for public APIs
- **Testing**: Pytest with >80% coverage goal

## Testing

```bash
# Run all tests
pixi run test

# Run specific test file
pytest tests/unit/test_pds_client.py -v

# Run with coverage
pytest --cov=mars_biosig --cov-report=html
```

## Code Quality

```bash
# Format code
pixi run format

# Lint code
pixi run lint

# Type check
pixi run typecheck
```

## Pull Request Process

1. **Create feature branch**: `git checkout -b feature/your-feature`
2. **Make changes with tests**: Ensure new code has test coverage
3. **Run tests**: `pixi run test` - all tests must pass
4. **Format and lint**: `pixi run format && pixi run lint`
5. **Commit and push**: Use clear, descriptive commit messages
6. **Open PR**: Include description of changes and any related issues

## Architecture Guidelines

### Data Pipeline
- All PDS access goes through `pds_client.py`
- Instrument-specific logic in `downloaders/`
- Parsers should be pure functions (no side effects)
- Cache aggressively to avoid re-downloading

### Models
- Inherit from `BaseModel` abstract class
- Configuration via YAML files in `configs/`
- No hardcoded hyperparameters in code
- Document architecture decisions

### Training
- Use `Trainer` class for orchestration
- Log metrics to TensorBoard
- Save checkpoints regularly
- Support resuming from checkpoints

## Adding New Features

### New Instrument Support

1. Create downloader: `src/mars_biosig/data/downloaders/new_instrument.py`
2. Add parser if needed: `src/mars_biosig/data/parsers/`
3. Create config: `configs/data/new_instrument.yaml`
4. Add tests: `tests/unit/test_new_instrument.py`
5. Update documentation

### New Model Architecture

1. Create model: `src/mars_biosig/models/your_model.py`
2. Inherit from `BaseModel`
3. Create config: `configs/models/your_model.yaml`
4. Add tests: `tests/unit/test_your_model.py`
5. Document in `docs/model_architecture.md`

## Scientific Contributions

If you're contributing scientific insights:

- Document methodology in `docs/`
- Include references to publications
- Provide example notebooks in `notebooks/`
- Explain biosignature taxonomy in `docs/biosignature_taxonomy.md`

## Questions?

Feel free to open an issue for:
- Bug reports
- Feature requests
- Documentation improvements
- Questions about the codebase

## Code of Conduct

- Be respectful and constructive
- Focus on the science and the code
- Help others learn and grow
- Cite sources and give credit

## License

By contributing, you agree that your contributions will be licensed under the BSD 3-Clause License.
