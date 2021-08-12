#!/usr/bin/env python3
"""
This is my take on a script for monitoring/dumping htlc events.
It prints to stdout and in CSV format to htlcstream.csv
There is no configurability unlike smallworlnd's stream-lnd-htlcs
"""

import time
import csv

from nodeinterface import NodeInterface

mynode = NodeInterface.fromconfig()

print('Caching channel info')
mychannels = {}
for chan in mynode.ListChannels().channels:
    mychannels[chan.chan_id] = chan

def getAlias4ChanID(chanid):
    if chanid not in mychannels:
        print('ERROR: Unknown chanid', chanid)
        return 'ERROR: Unknown'

    chan = mychannels[chanid]
    alias = mynode.getAlias(chan.remote_pubkey)
    return alias

events = mynode.router.SubscribeHtlcEvents()
print('Successfully subscribed, now listening for events')
for i, event in enumerate(events):
    inchanid = event.incoming_channel_id
    outchanid = event.outgoing_channel_id
    inchan = mychannels[inchanid]
    outchan = mychannels[outchanid]

    outcome = event.ListFields()[-1][0].name
    eventinfo = getattr(event, outcome)

    amount = ' - '
    fee = ' - '
    if outcome == 'forward_event':
        amt_msat = eventinfo.info.outgoing_amt_msat
        amount = amt_msat/1000
        fee = eventinfo.info.incoming_amt_msat - amt_msat
        fee /= 1000

    eventtype = event.EventType.keys()[event.event_type]
    timetext = time.ctime(event.timestamp_ns/1e9)

    inalias = outalias = 'N/A'
    if eventtype != 'SEND':
        inalias = getAlias4ChanID(inchanid)
    if eventtype != 'RECEIVE':
        outalias = getAlias4ChanID(outchanid)

    print(eventtype, timetext,
          amount,'for', fee,
          inalias, f'{inchan.remote_balance}/{inchan.capacity}',
          '➜',
          outalias, f'{outchan.local_balance}/{outchan.capacity}',
          # ~ inchanid, '➜', outchanid,
          outcome,
          # ~ eventinfo,
            )

    with open('htlcstream.csv', 'a', newline='') as f:
        writer = csv.writer(f)

        if i % 30 == 0:
            writer.writerow(['Eventtype', 'Timestamp', 'Amount', 'Fee',
                             'Alias_in','Alias_out',
                             'Balance_in','Capacity_in',
                             'Balance_out', 'Capacity_out',
                             'Chanid_in','Chanid_out',
                             'Outcome', 'Details'])

        writer.writerow([eventtype, timetext, amount, fee,
                         inalias, outalias,
                         inchan.remote_balance, inchan.capacity,
                         outchan.local_balance, outchan.capacity,
                         str(inchanid), str(outchanid),
                         outcome, eventinfo])

