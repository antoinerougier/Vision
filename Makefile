# Developer entry points. Every CI job calls the same commands, so "it works on my machine"
# and "it works in CI" mean the same thing.

PYTHON ?= python
CONFIG ?= configs/default.yaml
IMAGE  ?= isr-restore:local

.PHONY: help install lint format test train export evaluate gate evaluate-tracking gate-tracking pipeline serve viz docker-build docker-run clean

help:  ## List the available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install:  ## Install the package with every extra (editable)
	$(PYTHON) -m pip install -e ".[train,serve,dev]"

lint:  ## Static analysis (ruff): style, imports, common bugs
	ruff check .
	ruff format --check .

format:  ## Auto-format and auto-fix what can be fixed
	ruff check --fix .
	ruff format .

test:  ## Unit + integration tests with coverage
	$(PYTHON) -m pytest --cov=isr_restore --cov-report=term-missing

train:  ## Train the detector (clean data) then the restoration network
	$(PYTHON) -m isr_restore.train_detector --config $(CONFIG)
	$(PYTHON) -m isr_restore.train_restoration --config $(CONFIG)

export:  ## Export both models to ONNX (with a numerical parity check)
	$(PYTHON) -m isr_restore.export --config $(CONFIG)

evaluate:  ## Benchmark the ONNX models, write reports/
	$(PYTHON) -m isr_restore.evaluate --config $(CONFIG)

gate:  ## Benchmark AND fail (exit 1) if quality floors are not met
	$(PYTHON) -m isr_restore.evaluate --config $(CONFIG) --gate

evaluate-tracking:  ## Benchmark tracking + 10-minute forecast on simulated sequences
	$(PYTHON) -m isr_restore.evaluate_tracking --config $(CONFIG)

gate-tracking:  ## Same benchmark, AND fail (exit 1) if the tracking floors are not met
	$(PYTHON) -m isr_restore.evaluate_tracking --config $(CONFIG) --gate

pipeline: train export gate gate-tracking  ## Full ML pipeline: train -> export -> both quality gates

serve:  ## Run the API locally with auto-reload
	uvicorn isr_restore.api:app --reload --port 8000

viz:  ## Start the API and print the dashboard address (http://localhost:8000/viz)
	@echo "Dashboard: http://localhost:8000/viz"
	uvicorn isr_restore.api:app --port 8000

docker-build:  ## Build the serving image
	docker build -t $(IMAGE) .

docker-run:  ## Run the serving image on http://localhost:8000
	docker run --rm -p 8000:8000 $(IMAGE)

clean:  ## Remove caches and local run artefacts
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml logs tmp_check
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
