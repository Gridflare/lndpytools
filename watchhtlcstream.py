#!/usr/bin/env python3
"""
This is my take on a script for monitoring/dumping htlc events.
It prints to stdout and in CSV format to htlcstream.csv
There is no configurability unlike smallworlnd's stream-lnd-htlcs
"""

import time
import csv
import traceback
import argparse

from nodeinterface import NodeInterface

mynode = NodeInterface.fromconfig()

mychannels = {}
lastchannelfetchtime = 0
chandatatimeout = 15

def getChanInfo(chanid):
    """
    Fetches channel data from LND
    Uses a cache that times out after `chandatatimeout` seconds
    Also queries closed channels if `chanid` is not in open channels
    """
    global lastchannelfetchtime
    uptodate = (time.time() - lastchannelfetchtime < chandatatimeout)

    if uptodate and chanid in mychannels:
        return mychannels[chanid]

    for chan in mynode.ListChannels().channels:
        mychannels[chan.chan_id] = chan

    lastchannelfetchtime = time.time()

    if chanid in mychannels:
        return mychannels[chanid]

    for chan in mynode.ClosedChannels().channels:
        mychannels[chan.chan_id] = chan

    if chanid in mychannels:
        return mychannels[chanid]

    print('ERROR: Unknown chanid', chanid)
    return None

def getAlias4ChanID(chanid):
    chan = getChanInfo(chanid)
    if chan is None:
        return chanid
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

def subscribeEventsPersistent():
    failures = 0

    while True:
        events = mynode.router.SubscribeHtlcEvents()
        try:
            _ = mynode.GetInfo() # test connection
            failures = 0
            print('Connected to LND. Waiting for first event...')
            for e in events:
                yield e
        except StopIteration:
            raise
        except Exception as e:
            details = 'no details'
            try:
                details = e.details()
            except:
                pass

            print('Error:', details)
            unavailable = ('Connection refused' in details)
            unready = ('not yet ready' in details or 'wallet locked' in details)
            terminated = (details == "htlc event subscription terminated")

            if any((unavailable, unready, terminated)):
                failures += 1
                timeout = min(4**failures, 60*60*2)
                print(f'Could not connect to lnd, retrying in {timeout}s')
                time.sleep(timeout)
                continue

            print('Unhandled exception:', repr(e))
            raise e


def main():
    parser = argparse.ArgumentParser(description='Script for monitoring/dumping htlc events')
    parser.add_argument('--persist', action="store_true",
                    help='Automatically reconnect to LND')
    args = parser.parse_args()

    if args.persist:
        events = subscribeEventsPersistent()
    else:
        events = mynode.router.SubscribeHtlcEvents()
        print('Now listening for events')

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
                incap = inchan.capacity
                inrbal = inchan.remote_balance

            if outchanid:
                outalias = getAlias4ChanID(outchanid)
                outchan = getChanInfo(outchanid)
                # If channel is unknown (closed?) cannot guarantee these values exist
                outcap = getattr(outchan, 'capacity', 'UNKNOWN')
                outlbal = getattr(outchan, 'local_balance', 'UNKNOWN')

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
                note = '💸 HTLC in flight.'
                forward_event_cache[fwdcachekey] = {'amt':amount, 'fee':fee}

            elif outcome == 'forward_fail_event':
                note = '❌ Downstream fwding failure.'
                if fwdcachekey in forward_event_cache:
                    # This data is only found in forward_event, need to fetch it from cache
                    amount, fee = popamountsfromcache(fwdcachekey)

            elif outcome == 'link_fail_event':
                failure_string = eventinfo.failure_string
                failure_detail = getFailureAttribute(eventinfo, 'failure_detail')
                wire_failure = getFailureAttribute(eventinfo, 'wire_failure')

                if eventtype == 'RECEIVE' and failure_detail == 'UNKNOWN_INVOICE':
                    note += 'Probe detected. '

                note += f'❌ Failure(wire: {wire_failure}, detail: {failure_detail}, string: {failure_string})'


            elif outcome == 'settle_event' and  eventtype == 'FORWARD':
                note = '✅ Forward successful.'
                if fwdcachekey in forward_event_cache:
                    # This data is only found in forward_event, need to fetch it from cache
                    amount, fee = popamountsfromcache(fwdcachekey)

            elif outcome == 'settle_event':
                note = '✅'

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
            print('Exception while handling event.', repr(e))
            print(event)
            traceback.print_exc()

if __name__ == '__main__':
    main()
