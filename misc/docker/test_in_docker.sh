#!/bin/bash

##############################
# Install spectroscopy in a  #
# docker image and run the   #
# tests.                     #
# 11/17 Y. Behr              #
##############################

docker rmi yadabe/spectroscopy
docker build --no-cache=true -t yadabe/spectroscopy .
 
