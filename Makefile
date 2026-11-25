.PHONY: test lint fmt data baselines live
test:
	python -m pytest tests -q
data:
	@PYTHONPATH=src python -m oilfc.data.build
baselines:
	@PYTHONPATH=src python experiments/vol_risk_brent.py ladder-only
live:
	@PYTHONPATH=src python experiments/live_run.py
	@PYTHONPATH=src python experiments/live_score.py

lint:
	@python -m ruff check . && python -m ruff format --check .
fmt:
	@python -m ruff check --fix . && python -m ruff format .
