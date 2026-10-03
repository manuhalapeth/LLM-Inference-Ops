# Run on the GPU box from the repo root.

up:       ## start the full stack
	docker compose up -d --build

down:     ## stop the stack (keeps model weights and dashboards)
	docker compose down

logs:     ## follow logs from every service
	docker compose logs -f

ps:       ## show service status
	docker compose ps

smoke:    ## one request through gateway -> NGINX -> vLLM, saved to results/
	python3 scripts/smoke_test.py --wait 1800 --save

test:     ## gateway unit tests (needs gateway/requirements-dev.txt installed)
	cd gateway && python3 -m pytest -q tests

.PHONY: up down logs ps smoke test
