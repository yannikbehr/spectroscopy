#!/bin/bash

######################
# Run all tests      #
# Y. Behr 10/17      #
######################

python2 -m unittest discover -v
python2 verify_all_nzmetservice_data.py ./data


