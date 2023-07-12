# lndpytools
This repository is a collection of handy Python scripts for lightning node management. More scripts will be added over time.

## Available scripts
```
checkchannels.py     LX Identify your least effective channels. WIP.
checklndconf.py         Check an lnd.conf file for some simple recommendations.
checkmail.py         L  Check for recent keysends with metadata.
improvecentrality.py  X Find nodes that maximize centrality to improve routing.
nodeview.py             Some box&whisker plots of node data.
plotforwards.py      L  Heatmap of recent forwarding activity.
relativefees.py         Summarize a node's fee policy vs neighbours.
setfeepolicy.py      L  A basic script for setting fees and HTLC size.
watchhtlcstream.py   L  Human-readable log and CSV of HTLC events in real time.

L - Requires a connection to LND
X - CPU intensive, do not run on node hardware.
```

Individual documentation for each script is contained in the top section of each file.

## Usage
### Setup

Download the repository

`$ git clone https://github.com/Gridflare/lndpytools.git`

Change into the new directory

`$ cd lndpytools`

Install requirements

`$ pip3 install -r requirements.txt --user`

(The `--user` flag avoids conflicts with python libs installed by your system)

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

`$ python3 checkmail.py`

### Updates
You can download updates to the repo with

 `$ git pull`

### Tor proxy for 1ml.com lookups

The `improvecentrality.py` script uses 1ml.com for information on other nodes. To use use Tor to access 1ml.com information use the `ALL_PROXY` environment variable. For example:

`$ ALL_PROXY="socks5://localhost:9050" improvecentrality.py`.

## Advanced
All scripts are based on `nodeinterface.py` or `lnGraph.py`.

`NodeInterface` provides an introspective thin wrapper around the LND gRPC API. It tries to be faithful to the official docs while being much less verbose. Many calls are still unsupported, work in progress.

`lnGraph` provides an interface for loading LN graph data from a JSON file or LND into NetworkX or iGraph. For computationally intense centrality measures, the `fastcentrality` module can translate a NetworkX graph into iGraph for improved performance.
