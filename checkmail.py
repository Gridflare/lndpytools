#!/usr/bin/env python3
"""
This script checks your node's 60 most recent invoices for keysend
payments with custom_records and attempts to decode the contents.
Fields that cannot be decoded are printed as is.
"""

import time

from lib.nodeinterface import NodeInterface


mynode = NodeInterface.fromconfig()

invoices = mynode.ListInvoices(
                num_max_invoices=60,
                reversed=True,
                ).invoices

for invoice in invoices:
    if not invoice.settled:
        continue

    htlc_records = []
    for htlc in invoice.htlcs:
        custom_records = htlc.custom_records
        if custom_records:
            htlc_records.append(custom_records)

    if htlc_records:
        print('\nReceived',invoice.value_msat/1000,'sats on',time.ctime(invoice.settle_date))
        print('-'*40)

        for custom_records in htlc_records:
            for k, v in custom_records.items():
                print('Key:', k)
                try:
                    print(v.decode(),'\n')
                except UnicodeDecodeError:
                    # This is a hack, but it works
                    print(str(v).strip("b'"),'\n')

