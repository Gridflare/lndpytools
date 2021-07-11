#!/usr/bin/env python3
"""
Use this script to check if your config had defaults that need changing.
If your config is in a non-default location, pass it as an argument.
"""
import configparser
import sys
import os

"""
TODO
check lnd version, scrape from https://github.com/lightningnetwork/lnd/releases/latest
lncli -v  or  lncli --version or lnd --version if lnd is not running
"""

if len(sys.argv) > 1:
    confpath = sys.argv[1]
    if not os.path.isfile(confpath):
        raise FileNotFoundError(f'Please check your spelling, your config was not found at {confpath}')
else:
    if os.path.isfile('lnd.conf'):
        confpath = 'lnd.conf'
    else:
        confpath = os.path.expanduser('~/.lnd/lnd.conf')
    if not os.path.isfile(confpath):
        raise FileNotFoundError(f'You config does not exist at the default location of {confpath}')

lndconf = configparser.ConfigParser(strict=False)
lndconf.read(confpath)


AppOpts = lndconf['Application Options']

issuecount = 0
if not AppOpts.get('alias'):
    issuecount += 1
    print('No alias set, other operators may have a hard time finding you')
if not AppOpts.get('color'):
    issuecount += 1
    print('No color set, a color helps you stand out in explorers')
if not AppOpts.get('minchansize'):
    issuecount += 1
    print('No minimum channel size, other nodes could open uneconomical channels to you')
if not AppOpts.getboolean('accept-keysend'):
    issuecount += 1
    print('Node is not configured to accept keysend payments')
if not AppOpts.getboolean('accept-amp'):
    issuecount += 1
    print('Node is not configured to accept AMP payments')

hastor = False
if 'Tor' in lndconf:
    hastor = lndconf['Tor'].getboolean('tor.active', False)

if not any((AppOpts.get('listen'), AppOpts.getboolean('nat'), hastor)):
    issuecount += 1
    print('Your node cannot accept channels, consider setting up tor:')
    print('https://wiki.ion.radar.tech/tutorials/nodes/tor')

wtcactive = False
if 'wtclient' in lndconf:
    wtcactive = lndconf['wtclient'].getboolean('wtclient.active', False)

if not wtcactive:
    issuecount += 1
    print('Watchtower client service (wtclient) is not activated')
    print('Learn more about watchtowers: https://openoms.gitbook.io/lightning-node-management/advanced-tools/watchtower')

BtcOpts = lndconf['Bitcoin']
if BtcOpts.getint('bitcoin.feerate', 1) == 1:
        issuecount += 1
        print('Found default fee settings, the default rate fee is low and could cause liquidity to get stuck')

autocompact = False
if 'bolt' in lndconf:
    autocompact = lndconf['bolt'].getboolean('db.bolt.auto-compact', False)

if not autocompact:
    issuecount += 1
    print('db.bolt.auto-compact is not enabled, this can help use less disk space')
    print('If you are on a 32-bit OS such as a Pi, your DB must never exceed 1GB')

print(issuecount, 'potential improvements were found')
print('See here for an explanation of all options in the config file')
print('https://github.com/lightningnetwork/lnd/blob/master/sample-lnd.conf')
