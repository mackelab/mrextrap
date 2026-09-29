.PHONY: install test lint format smoke export

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .

smoke:
	mrextrap smoke

# fresh-history export for the public repository (never pushes): make export DEST=../mrextrap-public
export:
	tools/export_public_repo.sh $(DEST)
