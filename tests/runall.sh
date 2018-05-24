#!/bin/bash

######################
# Run all tests      #
# Y. Behr 10/17      #
######################

python -m unittest discover -v
python verify_all_nzmetservice_data.py ./data


