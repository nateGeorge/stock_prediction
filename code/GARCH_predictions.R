source('/home/nate/github/stock_prediction/code/quandl_utils.R')

dt <- load.latest.feather()

spy <- dt[Ticker == 'SPY']

library(rugarch)
library(xts)
library(PerformanceAnalytics)

# first need to get returns, as rugarch uses returns to fit model
spy.xts <- as.xts.data.table(spy[, c('Date', 'Adj_Close'), with = F])

spy.ret <- na.omit(CalculateReturns(spy.xts))


# variance targeting means the sample stddev is equal to unconditional variance (long-run variance)
# garch in-mean model means higher reward correlates with higher volatility
garchspec <- ugarchspec(mean.model = list(armaOrder = c(1, 1), archm = TRUE, archpow = 2),
                          variance.model = list(model = "gjrGARCH",
                                                variance.targeting = T),
                          distribution.model = "sstd")

garchfit <- ugarchfit(data = spy.ret, spec = garchspec)

garchvol <- sigma(garchfit) 

plot(garchvol)

plot(garchvol['2018/2019'])

plot(garchfit)

garchforecast <- ugarchforecast(fitORspec = garchfit, 
                                n.ahead = 5)

garchforecast

# gamma is the gjr leverage parameter (negative returns more strongly auto-correlated than positive returns due to leverage effect)
# ar1 and ma1 are ARIMA parameters -- ar1 positive means momentum, i.e. positive follows positive
# mu is average return
# 
coef(garchfit)


# check which coefs are statistically significant
# suggests rule-of-thumb of t-stat greater than 2 (probably around same as p-val > 0.05)
garchfit@fit$matcoef


# higher is better
likelihood(garchfit)

# todo -- add MSE analysis

# lower is better
infocriteria(garchfit)
