# Run on the GPU box from the repo root.

# Load .env so scripts see MODEL_NAME, VLLM_IMAGE, etc.
-include .env
export

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

trace:    ## follow single requests through every layer, saved to results/01_end_to_end/
	python3 scripts/trace_request.py

agent-env: ## Python environment for the CrewAI agent
	python3 -m venv .venv || (apt-get install -y -qq python3-venv && python3 -m venv .venv)
	.venv/bin/pip install -q -r agent/requirements.txt

agent:    ## run the CrewAI agent once, saved to results/01_end_to_end/
	.venv/bin/python agent/app.py

up-mooncake: ## start the stack with Mooncake as a KV cache tier
	VLLM_CONFIG=mooncake_store docker compose -f docker-compose.yml -f docker-compose.mooncake.yml up -d --build
	docker compose -f docker-compose.yml -f docker-compose.mooncake.yml restart nginx

kv-experiment: ## working set bigger than the GPU KV cache, saved to results/02_kv_cache_mooncake/
	python3 scripts/kv_cache_experiment.py --label $(or $(LABEL),$(VLLM_CONFIG))

eval:     ## run the eval set against the live stack, saved to results/03_harnesses_and_traces/
	python3 evals/run_evals.py --label $(or $(LABEL),harnesses_on)

drills:   ## trigger each kind of failure on purpose (stops and starts vLLM)
	python3 scripts/failure_drills.py

test:     ## unit tests (needs gateway/requirements-dev.txt installed)
	cd gateway && python3 -m pytest -q tests
	python3 -m pytest -q loadtest/tests evals/tests

.PHONY: up down logs ps smoke trace agent-env agent up-mooncake kv-experiment eval drills test
