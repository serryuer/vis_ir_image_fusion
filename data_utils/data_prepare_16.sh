#! /bin/bash

nohup python data_prepare_16.py 32 16 > 32_16.log 2>&1 &
nohup python data_prepare_16.py 64 32 > 64_32.log 2>&1 &
nohup python data_prepare_16.py 128 32 > 128_32.log 2>&1 &


