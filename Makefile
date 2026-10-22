.PHONY: test lint models image
test:
	.venv/bin/python -m pytest -q

lint:
	.venv/bin/ruff check src tests

models:
	python3 scripts/convert_models.py --out artifacts/models --weight-format fp16
	python3 scripts/parity.py --models artifacts/models --fixtures fixtures --dtype fp16

image:
	docker buildx build -t ghcr.io/gyhu-labs/reserv:dev-bge .
