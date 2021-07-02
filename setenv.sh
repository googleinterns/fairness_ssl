#!/bin/bash
sudo apt install python3-dev python3-virtualenv python3-tk imagemagick virtualenv
virtualenv -p python3 --system-site-packages env
. env/bin/activate

pip3 install -r requirements.txt
