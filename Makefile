VENV := .venv
PY := $(VENV)/bin/python
GRAPHRAG := $(VENV)/bin/graphrag

.PHONY: install lint type test demo eval run clean

install:
	uv venv --python 3.12 $(VENV)
	uv pip install -p $(VENV) -e ".[dev]"

lint:
	$(VENV)/bin/ruff check .

type:
	$(VENV)/bin/mypy src

test:
	$(VENV)/bin/pytest -q

demo:
	$(PY) scripts/demo.py

eval:
	$(GRAPHRAG) eval --corpus data/corpus --gold data/gold/qa_pairs.json --k 5

run:
	$(GRAPHRAG) serve --host 127.0.0.1 --port 8000

clean:
	rm -rf artifacts .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
