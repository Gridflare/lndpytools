#!/usr/bin/env python3
"""
This is my take on a script for monitoring/dumping htlc events.
It prints to stdout and in CSV format to htlcstream.csv
There is no configurability unlike smallworlnd's stream-lnd-htlcs
"""

import time
import csv
import traceback

from nodeinterface import NodeInterface

mynode = NodeInterface.fromconfig()

mychannels = {}
def getChanInfo(chanid):
    if chanid not in mychannels:
        for chan in mynode.ListChannels().channels:
            mychannels[chan.chan_id] = chan

        if chanid in mychannels:
            return mychannels[chanid]
        else:
            print('ERROR: Unknown chanid', chanid)
            return 'ERROR: Unknown'

def getAlias4ChanID(chanid):
    chan = getChanInfo(chanid)
    alias = mynode.getAlias(chan.remote_pubkey)
    return alias

events = mynode.router.SubscribeHtlcEvents()
print('Successfully subscribed, now listening for events')
for i, event in enumerate(events):
    try:
        inchanid = event.incoming_channel_id
        outchanid = event.outgoing_channel_id

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

        note = ''
        inalias = outalias = 'N/A'
        inrbal = incap = outlbal = outcap = '-'
        if inchanid:
            inalias = getAlias4ChanID(inchanid)
            inchan = getChanInfo(inchanid)
            inrbal = inchan.remote_balance
            incap = inchan.capacity

        if outchanid:
            outalias = getAlias4ChanID(outchanid)
            outchan = getChanInfo(outchanid)
            outlbal = outchan.local_balance
            outcap = outchan.capacity

        if outcome == 'link_fail_event':
            if eventtype == 'RECEIVE' and eventinfo.failure_string == "invoice not found":
                    note = 'Probe detected.'
            elif eventtype == 'FORWARD' and outcome == 'forward_event':
                if amount > inrbal:
                    note += f'Insufficient remote liquidity with {inalias}.'
                elif amount > outlbal:
                    note += f'Insufficient local liquidity with {outalias}.'

        print(eventtype, timetext,
              amount,'for', fee,
              inalias, f'{inrbal}/{incap}',
              '➜',
              outalias, f'{outlbal}/{outcap}',
              # ~ inchanid, '➜', outchanid,
              outcome,
              # ~ eventinfo,
              note,
                )

        with open('htlcstream.csv', 'a', newline='') as f:
            writer = csv.writer(f)

            if i % 30 == 0:
                writer.writerow(['Eventtype', 'Timestamp', 'Amount', 'Fee',
                                 'Alias_in','Alias_out',
                                 'Balance_in','Capacity_in',
                                 'Balance_out', 'Capacity_out',
                                 'Chanid_in','Chanid_out',
                                 'Outcome', 'Details', 'Note'])

            writer.writerow([eventtype, timetext, amount, fee,
                             inalias, outalias,
                             inrbal, incap,
                             outlbal, outcap,
                             f'{inchanid}', f'{outchanid}',
                             outcome, eventinfo, note])

    except Exception as e:
        print('Exception while handling event.', e)
        print(event)
        traceback.print_exc()
