# Codex Instructions

## Runtime and Test Environment

- Run and test this project only through the Docker Compose service defined in `docker-compose.yml`.
- Do not execute project programs, tests, scripts, Python modules, package managers, or build commands directly on the host machine.
- Host-side shell commands are limited to repository inspection, file editing, Git operations, and Docker Compose orchestration.
- Use `docker compose exec hyworld2 ...` when the `hyworld2` service is already running.
- Use `docker compose run --rm hyworld2 ...` for one-off commands when no long-lived container is needed.
- If a command would import project code, install dependencies, generate outputs, run tests, launch inference, or build project artifacts, run it inside the Compose container.
