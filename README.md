# lndpytools
This repository is a collection of handy Python scripts for lightning node management. More scripts will be added over time.

Some of these scripts, particularly `improvecentrality.py` are resource intensive and should not be run on node hardware.

Individual documentation for each script is contained in the top section of each file.

## Usage
### Setup

Download the repository

`$ git clone https://github.com/Gridflare/lndpytools.git`

Change into the new directory

`$ cd lndpytools`

Install requirements

`$ pip3 install -r requirements.txt --user`

### With LND gRPC
Most scripts use gRPC to connect to lnd directly after some setup, create the config file with

`$ python3 nodeinterface.py`

Double check `node.conf` and rerun nodeinterface, it will say `Connected to node <alias>` if everything is correct

### With describegraph.json
Instead of connecting to lnd, some scripts can use a fresh copy of `describegraph.json` in the lndpytools directory.
The json file will be preferred over connecting to LND where possible.
Create this file from lnd with

`$ lncli describegraph > describegraph.json`

### Running
You are now ready to run the scripts like so

`$ python3 checkchannels.py`

### Updates
You can download updates to the repo with

 `$ git pull`

