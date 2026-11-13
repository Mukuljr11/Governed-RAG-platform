.RECIPEPREFIX := >
.PHONY: up down lint test eval-ask eval-agent

up:
> docker compose up --build -d

down:
> docker compose down

lint:
> uv run ruff check .

test:
> uv run pytest -q

eval-ask:
> uv run python evals/run_evals.py --mode ask

eval-agent:
> uv run python evals/run_evals.py --mode agent