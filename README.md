# lndpytools
This repository is a collection of Python scripts that I use to manage my lightning node.

More scripts will be added over time and there is no guarantee of backwards compatibility.

## Usage

Download the repository

`$ git clone https://github.com/Gridflare/lndpytools.git`

Change into the new directory

`$ cd lndpytools`

Install requirements

`$ pip3 install -r requirements.txt --user`

Some scripts require a fresh copy of `describegraph.json` in the lndpytools directory. Create this file from lnd with

`$ lncli describegraph > describegraph.json`

You are now ready to run the scripts like so

`$ python3 improvecentrality.py`

You can download updates to the repo with

 `$ git pull`

Further documentation on each script is contained in the top section of each file.

