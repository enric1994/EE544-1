run:
	@docker-compose up -d
restart:
	@make down
	@make run
devel:
	@docker exec -it my_container bash
build:
	@docker-compose build
down:
	@docker-compose down -v
status:
	@docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
logs:
	@docker logs my_container -f