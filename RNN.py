import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import db


SEQ_LEN = 60
FUTURE_PREIOD_PREDICT = 3

COMM_TO_PREDICT = 'ES'

def classify(current, future):
  if float(future) > float(current):
    return 1
  else:
    return 0

sym_array = ['/ES', '/CL', '/GC']
main_df = pd.DataFrame()
""" request data from DB """
for sym in sym_array:
  data = db.get_symbol(sym)
  sym = sym[1:]
  print(sym)
  df = pd.DataFrame.from_dict(data).rename(columns={"close":f"{sym}_close", "volume":f"{sym}_volume"})

  if(len(main_df) == 0):
    main_df = df
  else:
    main_df = main_df.join(df)

for c in main_df.columns:
  print(c)


print(main_df.head())



