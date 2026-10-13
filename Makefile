test:
	python -m pytest -q

check:
	python -m scripts.check_setup

evaluate:
	python evaluate.py

api:
	uvicorn api:app --reload
