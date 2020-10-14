import re
import numpy as np
from datetime import datetime, timedelta

def process_doy(data, key, **kwargs):
    times = data[key]
    doys = np.zeros_like(times)
    for i, time in enumerate(times):
        dt = datetime.fromtimestamp(time)
        doys[i] = dt.timetuple().tm_yday
    data['doy'] = np.concatenate([
        np.sin(doys / 365 * 2 * np.pi).reshape(-1, 1),
        np.cos(doys / 365 * 2 * np.pi).reshape(-1, 1),
    ], axis=1)

def process_tod(data, key, **kwargs):
    times = data[key]
    tods = np.zeros_like(times)
    for i, time in enumerate(times):
        dt = datetime.fromtimestamp(time)
        tods[i] = dt.hour * 60 + dt.minute
    data['tod'] = np.concatenate([
        np.sin(tods / 24 / 60 * 2 * np.pi).reshape(-1, 1),
        np.cos(tods / 24 / 60 * 2 * np.pi).reshape(-1, 1),
    ], axis=1)

def process_season(data, key, **kwargs):
    times = data[key]
    days = int(kwargs['days'])
    seasons = np.zeros_like(times).astype('float')
    for i, time in enumerate(times):
        dt = datetime.fromtimestamp(time)
        seasons[i] = (dt - datetime(1970, 1, 1)).days % days
    data['season_%i' % days] = np.concatenate([
        np.sin(seasons / days * 2 * np.pi).reshape(-1, 1),
        np.cos(seasons / days * 2 * np.pi).reshape(-1, 1),
    ], axis=1)
