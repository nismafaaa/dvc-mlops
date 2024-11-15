.PHONY: all create-env select-env data train evaluate deploy clean setup_dirs install

# Default conda environment name and Python version
CONDA_ENV ?= mlops-makefile
PYTHON_VERSION ?= 3.8

# Detect the OS (Windows or Linux) to adjust the conda activation command
ifeq ($(OS),Windows_NT)
	CONDA_ACTIVATE = call activate $(CONDA_ENV)
else
	CONDA_ACTIVATE = . ~/miniconda3/etc/profile.d/conda.sh && conda activate $(CONDA_ENV)
endif

# Main target that runs everything
all: create-env install data train evaluate deploy

# Check if the Conda environment exists, otherwise create it
create-env:
	@echo "Checking if Conda environment $(CONDA_ENV) exists..."
	@if ! conda env list | grep -q "^$(CONDA_ENV)"; then \
		echo "Creating Conda environment $(CONDA_ENV) with Python $(PYTHON_VERSION)..."; \
		conda create -n $(CONDA_ENV) python=$(PYTHON_VERSION) -y; \
	fi

# Activate the environment and install dependencies
install:
	@echo "Installing dependencies..."
	$(CONDA_ACTIVATE) && \
	if [ -f environment.yml ]; then \
		conda env update -n $(CONDA_ENV) -f environment.yml; \
	else \
		python -m pip install --upgrade pip && pip install -r requirements.txt; \
	fi

# Ensure necessary directories exist
setup_dirs:
	@echo "Creating necessary directories..."
	@mkdir -p data model scripts

# Prepare the data
data: setup_dirs
	@echo "Preparing data..."
	$(CONDA_ACTIVATE) && python scripts/data_prep.py

# Train the model
train: data
	@echo "Training model..."
	$(CONDA_ACTIVATE) && python scripts/train_model.py

# Evaluate the model
evaluate: train
	@echo "Evaluating model..."
	$(CONDA_ACTIVATE) && python scripts/evaluate_model.py

# Deploy the model
deploy: evaluate
	@echo "Deploying model..."
	$(CONDA_ACTIVATE) && python scripts/deploy_model.py

# Clean the environment and directories
clean:
	@echo "Cleaning up..."
	rm -rf data/* model/* scripts/*