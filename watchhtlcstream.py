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
    else:
        return mychannels[chanid]

def getAlias4ChanID(chanid):
    chan = getChanInfo(chanid)
    alias = mynode.getAlias(chan.remote_pubkey)
    return alias

def getFailureAttribute(einfo, attr):
    i = getattr(einfo, attr)
    x = einfo.DESCRIPTOR.fields_by_name[attr]

    return x.enum_type.values_by_number[i].name

forward_event_cache = {}
def popamountsfromcache(key):
    amount = forward_event_cache[key]['amt']
    fee = forward_event_cache[key]['fee']
    del forward_event_cache[key]
    return amount, fee

def main():
    events = mynode.router.SubscribeHtlcEvents()
    print('Successfully subscribed, now listening for events')
    for i, event in enumerate(events):
        try:
            inchanid = event.incoming_channel_id
            outchanid = event.outgoing_channel_id

            outcome = event.ListFields()[-1][0].name
            eventinfo = getattr(event, outcome)
            eventtype = event.EventType.keys()[event.event_type]
            timetext = time.ctime(event.timestamp_ns/1e9)

            in_htlc_id = event.incoming_htlc_id
            out_htlc_id = event.outgoing_htlc_id

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

            # Extract forward amount data, if available
            amount = fee = '-'
            if hasattr(eventinfo, 'info'):
                if eventinfo.info.outgoing_amt_msat > 0:
                    amt_msat = eventinfo.info.outgoing_amt_msat
                    amount = amt_msat/1000
                    fee = (eventinfo.info.incoming_amt_msat - amt_msat)/1000

                elif eventinfo.info.incoming_amt_msat > 0:
                    amt_msat = eventinfo.info.incoming_amt_msat
                    amount = amt_msat/1000

            # Add a note to quickly point out common scenarios
            note = ''
            fwdcachekey = (in_htlc_id, out_htlc_id, inchanid, outchanid)
            if outcome == 'forward_event':
                note = 'HTLC in flight.'
                forward_event_cache[fwdcachekey] = {'amt':amount, 'fee':fee}

            elif outcome == 'forward_fail_event':
                note = 'Non-local fwding failure.'
                if fwdcachekey in forward_event_cache:
                    # This data is only found in forward_event, need to fetch it from cache
                    amount, fee = popamountsfromcache(fwdcachekey)

            elif outcome == 'link_fail_event':
                failure_string = eventinfo.failure_string
                failure_detail = getFailureAttribute(eventinfo, 'failure_detail')
                wire_failure = getFailureAttribute(eventinfo, 'wire_failure')

                if eventtype == 'RECEIVE' and failure_detail == 'UNKNOWN_INVOICE':
                    note += 'Probe detected. '

                note += f'Failure(wire: {wire_failure}, detail: {failure_detail}, string: {failure_string})'

            elif outcome == 'settle_event' and  eventtype == 'FORWARD':
                note = 'Forward successful.'
                if fwdcachekey in forward_event_cache:
                    # This data is only found in forward_event, need to fetch it from cache
                    amount, fee = popamountsfromcache(fwdcachekey)

            print(eventtype,
                  in_htlc_id, out_htlc_id,
                  timetext, amount,'for', fee,
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
                    writer.writerow(['Eventtype', 'Htlc_id_in', 'Htlc_id_out',
                                    'Timestamp', 'Amount', 'Fee',
                                     'Alias_in','Alias_out',
                                     'Balance_in','Capacity_in',
                                     'Balance_out', 'Capacity_out',
                                     'Chanid_in','Chanid_out',
                                     'Outcome', 'Details', 'Note'])

                writer.writerow([eventtype,
                                 event.incoming_htlc_id,
                                 event.outgoing_htlc_id,
                                 timetext, amount, fee,
                                 inalias, outalias,
                                 inrbal, incap,
                                 outlbal, outcap,
                                 f"{inchanid}", f"{outchanid}",
                                 outcome, eventinfo, note])

        except Exception as e:
            print('Exception while handling event.', e)
            print(event)
            traceback.print_exc()

if __name__ == '__main__':
    main()
