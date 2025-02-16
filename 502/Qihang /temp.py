# %%
import gc
import itertools
import fileinput
import json
import sqlite3
from collections import deque
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import ciso8601
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns
import sortedcontainers
import statsmodels.api as sm


class HDBL2:
    def __init__(self) -> None:
        pass

    @classmethod
    def get(cls, date, sym, nlvl: int = 50):
        log_files = cls.get_files(date, sym, 'l2')
        lines = list(fileinput.input(log_files))
        read = cls.parse(lines, nlvl)
        return read.loc[read['time'] > '2001-01-01'].reset_index(drop=True)

    @classmethod
    def get_files(cls, date: str, sym: str, dtype: str = 'l2', log_path='/home/hdb/cbat/l2/raw/') -> list:
        def f(x):
            x = x.split('.')[-1]
            return 0 if x == 'log' else int(x)
        p = Path(log_path)
        d = pd.to_datetime(date).strftime('%Y%m%d')
        fkey = f'{d}.{dtype}.{sym}.log'

        files = sorted(list(map(str, p.glob(f'{fkey}*'))), key=f, reverse=True)
        return files

    @classmethod
    def load_line(cls, logline: str) -> dict:
        if 'ENDENDEND' in logline:
            return {}
        return json.loads(logline.strip('\n').split(' - ')[1].replace("'", '"'))

    @classmethod
    def parse(cls, lines, nlvl: int = 50) -> pd.DataFrame:
        bidque = sortedcontainers.SortedDict()
        askque = sortedcontainers.SortedDict()
        N = len(lines)
        epochs = [0] * N
        bps = [[]] * N
        bqs = [[]] * N
        aps = [[]] * N
        aqs = [[]] * N
        lvls = [[]] * N
        for i, line in enumerate(lines[:]):
            # rl.keys(): dict_keys(['channel', 'client_id', 'timestamp', 'sequence_num', 'events'])
            rl = cls.load_line(line)
            if not rl:
                continue
            match rl['channel']:
                case 'l2_data':
                    events = rl['events']  # usually just 1 event in events
                    for evn in events:
                        match evn['type']:
                            # evn.keys(): dict_keys(['type', 'product_id', 'updates'])
                            case 'snapshot' | 'update':
                                # usually multiple updates in one evn
                                updates = evn['updates']
                                # only one event_time across all updates within one event
                                time = updates[0]['event_time']
                                epoch_time = int(ciso8601.parse_datetime(time).strftime('%s%f'))
                                # upd.keys(): dict_keys(['side', 'event_time', 'price_level', 'new_quantity'])
                                for upd in updates:
                                    side = upd['side']
                                    px = float(upd['price_level'])
                                    qty = float(upd['new_quantity'])

                                    if side == 'bid':
                                        bidque[px] = qty
                                        if qty == 0:
                                            bidque.pop(px)
                                    else:
                                        askque[px] = qty
                                        if qty == 0:
                                            askque.pop(px)

                                epoch_ = epoch_time
                                bp_ = list(bidque.keys()[-nlvl:])[::-1]
                                bq_ = list(bidque.values()[-nlvl:])[::-1]
                                ap_ = list(askque.keys()[:nlvl])
                                aq_ = list(askque.values()[:nlvl])
                                lvl_ = list(range(nlvl))

                                epochs[i] = epoch_
                                bps[i] = bp_
                                bqs[i] = bq_
                                aps[i] = ap_
                                aqs[i] = aq_
                                lvls[i] = lvl_
                            case _:
                                print(i, evn['type'])
                                raise KeyError("unknown event type")

                case 'subscriptions':
                    next
                case _:
                    raise KeyError("unknow message channel")

        df0 = pd.DataFrame({'epoch': epochs,
                            'bp': bps, 'bq': bqs,
                            'ap': aps, 'aq': aqs, 'lvl': lvls}).dropna().sort_values('epoch')
        df0['time'] = pd.to_datetime(df0['epoch'], unit='us')
        cols_ = ['time', 'bp', 'bq', 'ap', 'aq', 'lvl']
        table = df0[cols_].iloc[1:].copy()
        table = table.loc[table['bp'] < table['ap']]
        return table
