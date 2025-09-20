VENV=.venv
PY=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

.PHONY: venv install run build-index watch-pdfs convert-pdfs
.PHONY: prepare-chunks dev-install

venv:
	python3 -m venv $(VENV)

install: venv
	$(PY) -m pip install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt

run:
	# Start via gunicorn for stability
	$(VENV)/bin/gunicorn -b 127.0.0.1:5000 -w 1 app:app --log-file gunicorn.log --log-level info

build-index:
	$(PY) manage.py build

convert-pdfs:
	$(PY) scripts/pdf_to_txt.py --archive archive

watch-pdfs:
	$(PY) scripts/pdf_to_txt.py --watch --archive archive

prepare-chunks:
	$(PY) scripts/prepare_chunks.py

dev-install: install
	$(PIP) install -r requirements-dev.txt
