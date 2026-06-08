.PHONY: install profile train analyze test api app docker-build docker-up docker-down

install:
	python -m pip install -r requirements.txt

profile:
	PYTHONPATH=src python -m bank_marketing_strategy.cli profile

train:
	PYTHONPATH=src python -m bank_marketing_strategy.cli train

analyze:
	PYTHONPATH=src python -m bank_marketing_strategy.cli analyze

test:
	python -m pytest

api:
	PYTHONPATH=src uvicorn bank_marketing_strategy.api:app --reload

app:
	PYTHONPATH=src streamlit run app/streamlit_app.py

docker-build:
	docker compose build

docker-up:
	docker compose up --build

docker-down:
	docker compose down
