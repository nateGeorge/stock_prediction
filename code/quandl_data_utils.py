# core
import os
import datetime

# installed
import quandl
import pandas as pd
from pytz import timezone
from concurrent.futures import ProcessPoolExecutor

# custom
from utils import get_home_dir


DEFAULT_STORAGE = '/home/nate/eod_data/'
# get todays date for checking if files up-to-date
MTN = timezone('America/Denver')
TODAY = datetime.datetime.now(MTN)
WEEKDAY = TODAY.weekday()
HOUR = TODAY.hour

HOME_DIR = get_home_dir()

Q_KEY = os.environ.get('quandl_api')

quandl.ApiConfig.api_key = Q_KEY

spy_vix = {}
closes = {}
dates = {}
for i in range(1, 10):
    print(i)
    spy_vix[i] = quandl.get("CHRIS/CBOE_VX" + str(i))
    closes[i] = spy_vix[i]['Close']
    dates[i] = spy_vix[i].index

for i in range(1, 10):
    print(i)
    print(dates[i][0])

spy_vix_df = pd.DataFrame({i:closes[i] for i in range(1, 8)})
spy_vix_df.index = dates[1]

spy_vix_df.dropna(inplace=True)

# import matplotlib
# matplotlib.use('tkagg')  # Or any other X11 back-end
import matplotlib.pyplot as plt

plt.plot(spy_vix_df.iloc[-1])
plt.show()
