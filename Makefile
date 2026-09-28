install:
	pip install -r requirements.txt
ingest:
	python scripts/ingest.py
index:
	python scripts/index.py
run:
	python -m app
test:
	python -m pytest tests -q
eval:
	python -m evaluation.run
