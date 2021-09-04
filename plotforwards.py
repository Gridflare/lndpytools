#!/usr/bin/env python3
"""
Creates a heatmap of recent forwards
Run `python3 plotforwards -h` for options`
"""

import argparse
import time
from datetime import timedelta
import random

import numpy as np
import matplotlib.pyplot as plt

from nodeinterface import NodeInterface

parser = argparse.ArgumentParser(description='Plot a diagram of recent forwards')
parser.add_argument('--days', type=int, default=90,
                    help='Specify lookback time, default 90')

# Below 2 are exclusive
group = parser.add_mutually_exclusive_group()
group.add_argument('--count', action="store_true",
                    help='Plot forward count instead of volume')
group.add_argument('--fees', action="store_true",
                    help='Plot fee returns instead of volume')

parser.add_argument('--perpeer', action="store_true",
                        help='Collect redundant channels by peer')
parser.add_argument('--nolabels', action="store_true",
                        help='Do not plot grid labels, for privacy when sharing #nodeart')
parser.add_argument('--shuffle', action="store_true",
                        help='Randomize order of channels, for privacy when sharing #nodeart')

parser.add_argument('--max', type=int, default=-1,
                    help='Don\'t plot pairs above this activity threshold')
parser.add_argument('--min', type=int, default=0,
                    help='Don\'t plot pairs below this activity threshold')

parser.add_argument('--aliasclip', type=int, default=20,
                    help='Shorten long aliases to this many characters, default 20')

parser.add_argument('--nogrid', action="store_true",
                        help='Do not plot grid lines')
parser.add_argument('--cmap', type=str, default='Blues',
                    help='Use an alternate colormap')

args = parser.parse_args()

mynode = NodeInterface.fromconfig()
fwdhisttime = timedelta(days=args.days)
print('Fetching forwarding history')
fwdhist = mynode.getForwardingHistory(
            int(time.time()-fwdhisttime.total_seconds()))

if args.count:
    plotattr = 'count'
elif args.fees:
    plotattr = 'fees'
else:
    plotattr = 'amount'

print('Fetching channel data')
channeldata = {}
for c in mynode.ListChannels().channels:
    channeldata[c.chan_id] = c

print('Processing data')

labels = []
if args.perpeer:
    id_list = []
    for i, c in sorted(channeldata.items()):
        rpk = c.remote_pubkey
        if rpk not in id_list:
            id_list.append(rpk)
else:
    # Sort to keep results consistent
    id_list = sorted(channeldata.keys())

if args.shuffle:
    random.shuffle(id_list)

if args.perpeer:
    for cid in id_list:
        labels.append(
            mynode.getAlias(cid)[:args.aliasclip])
else:
    for cid in id_list:
        labels.append(
            mynode.getAlias(channeldata[cid].remote_pubkey)[:args.aliasclip])

num_channels = len(labels)
dataarray = np.zeros((num_channels, num_channels))

for event in fwdhist:
    fee = event.fee_msat/1000

    id_in = event.chan_id_in
    id_out = event.chan_id_out
    if id_in not in channeldata or id_out not in channeldata:
        continue # Catch recently closed channels

    if args.perpeer:
        # Convert chan id to remote node id
        id_in = channeldata[id_in].remote_pubkey
        id_out = channeldata[id_out].remote_pubkey

    i1 = id_list.index(id_in)
    i2 = id_list.index(id_out)

    if args.count:
        v = 1
    elif args.fees:
        v = event.fee_msat/1000
    else:
        v = event.amt_out_msat

    dataarray[i1, i2] += v

if args.min > 0 or args.max > 0:
    print('Filtering output array')

    todelete = []
    for i, l in enumerate(id_list):
        activity = max(dataarray[i, :].max(), dataarray[:, i].max())
        if activity < args.min or (args.max > 0 and activity > args.max):
            todelete.append(i)

    # Have to reverse if deleting by index, remove highest index first
    for i in reversed(todelete):
        del id_list[i]
        del labels[i]

    dataarray = np.delete(dataarray, todelete, axis=0)
    dataarray = np.delete(dataarray, todelete, axis=1)

print('Plotting output')
fig, ax = plt.subplots()
im = ax.imshow(dataarray, cmap=args.cmap, vmin=0, vmax=dataarray.max())

# Force all ticks visible
tickrange = np.arange(len(labels))
ax.set_xticks(tickrange)
ax.set_yticks(tickrange)
ax.set_xticks(tickrange-.5, minor=True)
ax.set_yticks(tickrange-.51, minor=True)

# Hide ugly ticks + spines
ax.spines[:].set_visible(False)
ax.tick_params(which="minor", bottom=False, left=False)

# Label ticks
if args.nolabels:
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.tick_params(color='w')
else:
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.tick_params(color='silver')

    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=-45, ha="left",
             rotation_mode="anchor")

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)

if not args.nogrid:
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)

ax.set_title(f'Forwarding {plotattr} in the last {args.days} days')
ax.set_ylabel('Channel in')
ax.set_xlabel('Channel out')
fig.tight_layout()

# Adjust borders to keep labels on-screen
plt.subplots_adjust(bottom=0.2)

plt.show()
