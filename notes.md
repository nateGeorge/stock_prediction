5/1/2018

used simple_quandl_analysis.py to do some basic filtering -- looked for high short % (>10), high days to cover (>10), and positive roc
looked for stocks seeming to go into a short squeeze, especially before earnings
worked day of TDOC earnings, but then reversed the next day


5/7/2018
tried neural net on short data -- filtered by marked cap, short %, days to cover, with not much effect.  scaled input and output data -- couldn't get good results until both were scaled.
probably would be better if scaled to tanh or sigmoid function

5/11/2018
all picks from previously are looking good, but whole market is up.  TDOC went up on earnings, then down, then back up.  WETF was a good pick.  ONVO is doing awesome, but could've been had for much cheaper (almost 10%)
PAYC is down, but seems like holding on for another day is a good idea.  The short squeeze may well be over, but the long-term trend is up.

When screening by short_close_corr_rocr_20d, it seems the inverse of what I'd expect -- highly negative for those going into a big positive correlation, and highly positive for those going into a big negative correlation.

WING is in a great short squeeze right now -- need to figure out a way to better detect this

problem with short_close_corr_rocr_20d -- when the corr is close to 0, the value of short_close_corr_rocr_20d spikes up.  
