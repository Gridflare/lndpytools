#!/usr/bin/env python3
"""
This script is provided for educational purposes, it sets channel
fees based on chain costs and imbalance.
This far more basic than similar tools and likely not flexible enough
for advanced operators, but it should give a good idea of what a fair
fee is to beginners.
Run `python3 setfeepolicy.py -h` for parameters.
"""

import argparse

import numpy as np

from lib.nodeinterface import NodeInterface

# Handle arguments
parser = argparse.ArgumentParser(description='Set channel rate fees based on simple heuristics')
parser.add_argument('--chainfee', type=int, default=6000,
                    help='The estimated cost in sats of opening and closing a channel, accounting for force close risk and hiring fees, default 6000')
parser.add_argument('--passes', type=float, default=2,
                    help='The number of times channels must be fully used to break even, default 2, fixed to 1 for sink channels')
parser.add_argument('--rebalfactor', type=int, default=10,
                    help='This factor controls how agressively fees will adjust with channel imbalance, default 10')
parser.add_argument('--basefee', type=int, default=50,
                    help='The base fee in msat to apply to all channels, default 50')
parser.add_argument('--minhtlc', type=int,
                    help='If basefee is 0 this will be automatically set to prevent free forwards')
parser.add_argument('--sink', action='append', default=[],
                    help='Specify the pubkeys of nodes that receive vastly more than they send')
parser.add_argument('--sinkpenalty', type=float, default=1.5,
                    help='Fee multiplier for sink channels, default 1.5')
parser.add_argument('--timelockdelta', type=int, default=40,
                    help='The time lock delta to apply to all channels, default 40')
parser.add_argument('--apply', action="store_true",
                    help='By default fees are suggested but not applied, set this flag to apply them')
parser.add_argument('--setmaxhtlc', action="store_true",
                    help='Adjust max htlc to try to avoid failed forwards')
parser.add_argument('--override', action='append', default=[],
                    help='Use this fee policy for this node. Format: pubkey,ppm')
args = parser.parse_args()

mynode = NodeInterface.fromconfig()

# Find channel ratios and median channel size
# Using median prevents overcharging for small channels and
# undercharging for large ones
chansizes = []
chanratios = {}
mychannels = mynode.ListChannels().channels
balancesbypeer = {}
for chan in mychannels:
    # Need channel size to get average
    effective_capacity = (chan.capacity
                          - chan.local_constraints.chan_reserve_sat
                          - chan.remote_constraints.chan_reserve_sat)
    chansizes.append(effective_capacity)

    # Need these ratios when printing
    remote_ratio = chan.remote_balance / chan.capacity
    chanratios[chan.channel_point] = remote_ratio

    if chan.remote_pubkey not in balancesbypeer:
        balancesbypeer[chan.remote_pubkey] = {
            'local':0,'remote':0, 'total':0}

    balancesbypeer[chan.remote_pubkey]['local'
        ] += chan.local_balance - chan.local_constraints.chan_reserve_sat
    balancesbypeer[chan.remote_pubkey]['remote'
        ] += chan.remote_balance - chan.remote_constraints.chan_reserve_sat
    balancesbypeer[chan.remote_pubkey]['total'] += effective_capacity

# Find the basic rate fee
basicratefee = args.chainfee / np.median(chansizes) / args.passes

# Modify the rate fee for each channel for channel balance
def imbalancemodifier(remote_ratio):
    # The [0-1] integral of the function must equal 1
    factor = 1+args.rebalfactor*(remote_ratio-0.5)**5
    return max(factor, 0) # Make sure it's not negative

if imbalancemodifier(0) <= 0:
    raise ValueError('--rebalfactor is too large, fees could be zero or negative')

ppm_overrides = {k:int(p) for k, p in map(lambda s: s.split(','), args.override)}

newratefeesbypeer = {}
minhtlcsbypeer = {}
maxhtlcsbypeer = {}
for rkey, balances in balancesbypeer.items():
    ratio = balances['remote'] / balances['total']
    ratefee = basicratefee * imbalancemodifier(ratio)

    if rkey in args.sink:
        # We only have one pass to profit from, account for this
        ratefee *= args.passes * args.sinkpenalty

    if rkey in ppm_overrides:
        ratefee = ppm_overrides[rkey]/1e6

    newratefeesbypeer[rkey] = ratefee

    # If base fee is 0, set min htlc to prevent free forwards
    if args.basefee <= 0:
        minhtlc = int(np.ceil(1/ratefee))
        if args.minhtlc:
            minhtlc = max(minhtlc, args.minhtlc)
        minhtlcsbypeer[rkey] = minhtlc
    elif args.minhtlc:
        minhtlcsbypeer[rkey] = args.minhtlc

    if args.setmaxhtlc:
        b = max(10e6, balances['local']*1000*0.6)
        mh = round(b, -int(np.log10(b+1))+1)
        maxhtlcsbypeer[rkey] = int(mh)

# Print the proposed fees
print('basefee rate  minhtlc maxhtlc remote  cap   Alias')
print(' (msat)  fee   (sat)   (ksat)  ratio (ksat)      ')
for chan in mychannels:
    rate_fee = newratefeesbypeer[chan.remote_pubkey]
    remote_ratio = chanratios[chan.channel_point]
    base_fee = args.basefee

    if chan.remote_pubkey in minhtlcsbypeer:
        minhtlc = minhtlcsbypeer[chan.remote_pubkey]
    else:
        minhtlc = chan.local_constraints.min_htlc_msat

    if chan.remote_pubkey in maxhtlcsbypeer:
        maxhtlc = maxhtlcsbypeer[chan.remote_pubkey]
    else:
        maxhtlc = chan.local_constraints.max_pending_amt_msat
    maxhtlc = int(maxhtlc/1e6)
    if maxhtlc >= 1e7:
        maxhtlc = '   ...'

    print('{:6} {:.4%} {:6} {:7} {:6.0%} {:6.0f} {}'.format(
            base_fee, rate_fee, minhtlc/1e3, maxhtlc, remote_ratio,
            chan.capacity/1e3, mynode.getAlias(chan.remote_pubkey)))

if args.apply:
    print('Applying fees')

    for chan in mychannels:
        kwargs = dict(
            chan_point = chan.channel_point,
            fee_rate = newratefeesbypeer[chan.remote_pubkey],
            base_fee_msat = args.basefee,
            time_lock_delta = args.timelockdelta,
        )
        if chan.remote_pubkey in minhtlcsbypeer:
           kwargs['min_htlc_msat_specified'] = True
           kwargs['min_htlc_msat'] = minhtlcsbypeer[chan.remote_pubkey]

        if maxhtlcsbypeer:
            kwargs['max_htlc_msat'] = maxhtlcsbypeer[chan.remote_pubkey]

        mynode.UpdateChannelPolicy(**kwargs)

    print('Fees applied')






