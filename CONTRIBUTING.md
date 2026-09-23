# Contributing to Mars CTX Explorer

## Setup

```bash
git clone https://github.com/davidackerman/mars_ctx_explorer.git
cd mars_ctx_explorer
pixi install
pixi shell
pre-commit install
```

## Standards

- Formatting: Black, line length 100 (`pixi run format`)
- Linting: Ruff (`pixi run lint`)
- Tests: pytest (`pixi run test`); add tests for new behaviour where practical
- Type hints on public functions; NumPy-style docstrings

## Pull requests

1. Branch from `main`.
2. Keep commits focused; explain *why* in the message body.
3. Make sure `pixi run lint` and `pixi run test` pass.
4. Open a PR describing what changed and how you checked it.
