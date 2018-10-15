import scipy.signal as scisig
import numpy as np


def downsample_db(db, ToFs: int):
    i = 0
    for rec in db.db:
        sig = rec.ecg
        q = int(np.ceil(rec.Fs/ToFs))
        resampled_sig = scisig.decimate(sig, q, ftype='iir', n=20, zero_phase=True)
        db.update_record(resampled_sig, i)
        i += 1

    return db
