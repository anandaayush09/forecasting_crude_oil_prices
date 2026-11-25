.PHONY: test lint fmt data baselines
test:
	python -m pytest tests -q
data:
	@PYTHONPATH=src python -m oilfc.data.build
baselines:
	@PYTHONPATH=src python experiments/vol_risk_brent.py ladder-only

lint:
	@python -m ruff check . && python -m ruff format --check .
fmt:
	@python -m ruff check --fix . && python -m ruff format .
