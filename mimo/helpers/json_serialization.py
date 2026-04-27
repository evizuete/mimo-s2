import json
from datetime import date, datetime

import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        # pandas / datetime
        if isinstance(obj, (pd.Timestamp, datetime, date)):
            return obj.isoformat()

        if isinstance(obj, pd.Timedelta):
            return str(obj)

        if obj is pd.NaT:
            return None

        # numpy
        if isinstance(obj, (np.integer,)):
            return int(obj)

        if isinstance(obj, (np.floating,)):
            return float(obj)

        if isinstance(obj, (np.bool_,)):
            return bool(obj)

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # pandas containers opcionales
        if isinstance(obj, pd.Series):
            return obj.to_dict()

        return super().default(obj)