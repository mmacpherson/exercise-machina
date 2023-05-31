SHELL:=/bin/bash
PROJECT=exercise-machina
VERSION=3.10.11
VENV=${PROJECT}-${VERSION}
VENV_DIR=$(shell pyenv root)/versions/${VENV}
VENV_BIN=${VENV_DIR}/bin
PYTHON=${VENV_BIN}/python
JUPYTER_ENV_NAME="python (${VENV})"


## Make sure you have `pyenv` and `pyenv-virtualenv` installed beforehand
##
## https://github.com/pyenv/pyenv
## https://github.com/pyenv/pyenv-virtualenv
##
## On a Mac: $ brew install pyenv pyenv-virtualenv
##
## Configure your shell via:
##   https://github.com/pyenv/pyenv#set-up-your-shell-environment-for-pyenv
##

# .ONESHELL:
DEFAULT_GOAL: help
.PHONY: help run clean build venv ipykernel update

# Colors for echos
ccend=$(shell tput sgr0)
ccbold=$(shell tput bold)
ccgreen=$(shell tput setaf 2)
ccso=$(shell tput smso)

clean: ## >> remove all environment and build files
	@echo ""
	@echo "$(ccso)--> Removing virtual environment $(ccend)"
	pyenv virtualenv-delete --force ${VENV}
	rm -f .python-version

build: ##@main >> build the virtual environment with an ipykernel for jupyter and install requirements
	@echo ""
	@echo "$(ccso)--> Build $(ccend)"
	$(MAKE) install
	$(MAKE) ipykernel

venv: $(VENV_DIR) ## >> setup the virtual environment

$(VENV_DIR):
	@echo "$(ccso)--> Create pyenv virtualenv $(ccend)"
	pyenv install --skip-existing $(VERSION)
	pyenv virtualenv --force ${VERSION} ${VENV}
	echo ${VENV} > .python-version

requirements.txt: requirements.in
	$(VENV_BIN)/pip-compile --upgrade requirements.in --output-file requirements.txt

install: venv requirements.txt ##@main >> update requirements.txt inside the virtual environment
	@echo "$(ccso)--> Updating packages $(ccend)"
	$(PYTHON) -m pip install -U pip wheel pip-tools
	$(VENV_BIN)/pip-sync requirements.txt

ipykernel: venv ##@main >> install a Jupyter iPython kernel using our virtual environment
	@echo ""
	@echo "$(ccso)--> Install ipykernel to be used by jupyter notebooks $(ccend)"
	$(PYTHON) -m pip install ipykernel jupyter jupyter_contrib_nbextensions nb-black watermark
	$(PYTHON) -m ipykernel install --user --name=$(VENV) --display-name=$(JUPYTER_ENV_NAME)
	$(PYTHON) -m jupyter contrib nbextension install --user

# Other commands.
hooks: ##@options >> install pre-commit hooks
	pre-commit install

update-hooks: ##@options >> bump all hooks to latest versions
	pre-commit autoupdate

run-hooks: ##@options >> run hooks over staged files
	pre-commit run

run-hooks-all-files: ##@options >> run hooks over ALL files in workspace
	pre-commit run -a

# And add help text after each target name starting with '\#\#'
# A category can be added with @category
HELP_FUN = \
	%help; \
	while(<>) { push @{$$help{$$2 // 'options'}}, [$$1, $$3] if /^([a-zA-Z\-\$\(]+)\s*:.*\#\#(?:@([a-zA-Z\-\)]+))?\s(.*)$$/ }; \
	print "usage: make [target]\n\n"; \
	for (sort keys %help) { \
	print "${WHITE}$$_:${RESET}\n"; \
	for (@{$$help{$$_}}) { \
	$$sep = " " x (32 - length $$_->[0]); \
	print "  ${YELLOW}$$_->[0]${RESET}$$sep${GREEN}$$_->[1]${RESET}\n"; \
	}; \
	print "\n"; }

help: ##@help >> Show this help.
	@perl -e '$(HELP_FUN)' $(MAKEFILE_LIST)
	@echo ""
	@echo "Note: to activate the environment in your local shell type:"
	@echo "   $$ pyenv activate $(VENV)"
