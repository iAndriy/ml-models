# Makefile for Python project on OS X
EPOCHS?=3

# Command to create virtual environment
run_cnn:
	venv/bin/python
pull_data:
	curl https://download.pytorch.org/tutorial/hymenoptera_data.zip -o /tmp/tmp.zip && unzip /tmp/tmp.zip -d src/transfer_learning/data && rm /tmp/tmp.zip

install:pull_data
	python3 -m venv venv
	@echo "Virtual environment 'venv' created."
	venv/bin/pip install -r requirements.txt
train:
	EPOCHS=$(EPOCHS) venv/bin/python src/transfer_learning/main.py
# Command to activate the virtual environment (for reference)
activate:
	@echo "To activate the virtual environment, run:"
	@echo "source venv/bin/activate"

# Command to install requirements (if you have a requirements.txt file)
install-requirements: install
	venv/bin/pip install -r requirements.txt
	@echo "Requirements installed."

# Command to deactivate the virtual environment (for reference)
deactivate:
	@echo "To deactivate the virtual environment, run:"
	@echo "deactivate"

# Clean up the virtual environment
clean:
	rm -rf venv
	@echo "Virtual environment 'venv' removed."

# PHONY targets to avoid conflicts with files of the same name
.PHONY: install activate install-requirements deactivate clean

