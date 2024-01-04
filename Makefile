all:
	source .venv/bin/activate; \
	pip install --editable .

venv:
	rm -rf .venv
	python -m venv --copies .venv
	source .venv/bin/activate; \
	pip install --upgrade pip; \
	pip install wheel; \

wheel:
	rm -rf dist build *.egg-info
	pip wheel . --no-deps --wheel-dir dist

install: wheel
	pip install dist/*.whl
