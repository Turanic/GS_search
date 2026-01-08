all:
	docker compose up -d

.PHONY: redis-up
redis-up:
	docker compose up -d redis

.PHONY: vectorizer-up
vectorizer-up:
	docker compose up -d vectorizer

.PHONY: retrieval-up
retrieval-up: redis-up vectorizer-up
	docker compose up -d retrieval

.PHONY: import-vsd-fr
import-vsd-fr: redis-up vectorizer-up
	docker compose up -d importer-vsd-fr

.PHONY: import-public-fr
import-public-fr: redis-up vectorizer-up
	docker compose up -d importer-public-fr

.PHONY: import
import: import-vsd-fr import-public-fr

.PHONY: build
build:
	docker compose build

.PHONY: test
test:
	go test -race ./...

.PHONY: clean
clean:
	docker compose down -v --rmi all --remove-orphans

.PHONY: logs
logs:
	docker compose logs -f

.PHONY: down
down:
	docker compose down

.PHONY: restart
restart:
	docker compose restart






