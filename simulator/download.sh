#!/bin/bash

wget -nc https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip
unzip simulator-linux.zip
mv "Default Linux desktop Universal.x86_64" simulator
mv "Default Linux desktop Universal_Data" simulator_Data
rm Default Linux desktop Universal.x86
