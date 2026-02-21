#!/bin/bash

# Start Jupyter Notebook
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' &
sleep 5

# Keep the container running
tail -f /dev/null