#!/bin/bash


#Latest Changes:

# ~ October 22
# Updated all historical data grabbing functions to re-sort and store the data with the date of the latest stock
# value at the bottom of the file rather than the top.

# ~ October 15 ?
# Added historical data append function
#	Looks for stocks in folder "Historical" -- expects them to have the lastest date for data on the bottom line
#	Appends the latest stock data to the end of the file





# https://code.google.com/p/yahoo-finance-managed/wiki/YahooFinanceAPIs
# https://code.google.com/p/yahoo-finance-managed/wiki/CSVAPI
# https://code.google.com/p/yahoo-finance-managed/wiki/csvQuotesDownload
# https://code.google.com/p/yahoo-finance-managed/wiki/csvHistQuotesDownload

historial_base_url="http://ichart.yahoo.com/table.csv?"
quote_base_url="http://download.finance.yahoo.com/d/quotes.csv?"

stocklist="../stockdata/stocks.txt"

read weekday month day hms tz year <<< `date`
# could also do like this
# weekday = now | cut -c1-4
monthnum=`date -d "$month 1" "+%m"`



function query_yahoo()
{
	echo "Hello"
}

function get_historial_data()
{
	echo "Hello"
}

# a		=	starting month	(we have to subtract 1 from it)
# b		=	starting day
# c		=	year

# d		=	ending month (we have to subtract 1 from it)
# e		=	ending day
# f		= 	ending year

# g		=	interval		(possible interval values are: d=daily, w=weekly, m=monthly)

# add static value 		&ignore=.csv 			(who knows why)

# example:
# http://ichart.yahoo.com/table.csv?s=GOOG&a=0&b=0&c=0&d=12&e=31&f=2010&g=d&ignore=.csv
# should grab daily values for google for every day in 2010


function get_historic_values_all_stocks()
{
	directory="../stockdata/daily_"`date +%F`
	if [ ! -d "$directory" ]; then
		echo 'making directory';
		mkdir $directory;
	fi
	cat $stocklist | while read stock;
	do
		# if file doesn't exist				or		file size is equal to 0
		if [ ! -f "$directory/$stock.csv" ] || [ `du "$directory/$stock.csv" | sed 's#	.*##'` -eq 0 ]
			then
			# download all of the available data for the stock in 1 foul swoop
			# example: http://ichart.yahoo.com/table.csv?s=IBM&a=0&b=0&c=0&d=12&e=31&f=2012&g=w
			echo -ne "downloading $stock...\r";
			curl -s "${historial_base_url}s=${stock}&a=0&b=0&c=0&d=${monthnum}&e=${day}&f=${year}" > "$directory/$stock.csv"
			# Now let's fix the data so it's in our format: Sort the data so the latest value is at the bottom
			# versus the top.
			topline=`head -n 1 "$directory/$stock.csv"`
			linecount=`cat "$directory/$stock.csv" | wc -l`
			tempfile=`mktemp stocktemp.XXXXXX`
			echo $topline > $tempfile;
			tail -n "$(($linecount - 1))" "$directory/$stock.csv" >> $tempfile
			mv $tempfile "$directory/$stock.csv"
			sleep 1;
		fi
	done
}

function get_current_values_all_stocks()
{
	directory="../stockdata/current_"`date +%F`"-"`date +%H`
	if [ ! -d "$directory" ]; then
		echo 'making directory';
		mkdir $directory;
	fi
	# https://code.google.com/p/yahoo-finance-managed/wiki/enumQuoteProperty
	propertyvalues="c8g3a0b2a5a2b0b3b6b4c1c0m7m5k4j5p2k2c6c3c4h0g0m0m2w1w4r1d0y0e0j4e7e9e8q0m3f6l2g4g1g5g6v1v7d1l1k1k3t1l0l3j1j3i0n0n4t8o0i5r5r0r2m8m6k5j6p0p6r6r7p1p5s6s1j2s7x0s0t7d2t6f0m4v0k0j0w0"
	cat $stocklist | while read stock;
	do
		# if file doesn't exist				or		file size is equal to 0
		if [ ! -f "$directory/$stock.csv" ] || [ `du "$directory/$stock.csv" | sed 's#	.*##'` -eq 0 ]
			then
			# download all of the available data for the stock in 1 foul swoop
			curl -s "${quote_base_url}s=${stock}&f=${propertyvalues}&e=.csv" > "$directory/$stock.csv"
			# Now let's fix the data so it's in our format: Sort the data so the latest value is at the bottom
			# versus the top.
			topline=`head -n 1 "$directory/$stock.csv"`
			linecount=`cat "$directory/$stock.csv" | wc -l`
			tempfile=`mktemp stocktemp.XXXXXX`
			echo $topline > $tempfile;
			tail -n "$(($linecount - 1))" "$directory/$stock.csv" >> $tempfile
			mv $tempfile "$directory/$stock.csv"
			sleep 1;
		fi
	done
}

function get_current_values_all_stocks_batch()
{
	directory="../stockdata/current_"`date +%F`"-"`date +%H`
	if [ ! -d "$directory" ]; then
		echo 'making directory';
		mkdir $directory;
	fi
	# https://code.google.com/p/yahoo-finance-managed/wiki/enumQuoteProperty
	propertyvalues="c8g3a0b2a5a2b0b3b6b4c1c0m7m5k4j5p2k2c6c3c4h0g0m0m2w1w4r1d0y0e0j4e7e9e8q0m3f6l2g4g1g5g6v1v7d1l1k1k3t1l0l3j1j3i0n0n4t8o0i5r5r0r2m8m6k5j6p0p6r6r7p1p5s6s1j2s7x0s0t7d2t6f0m4v0k0j0w0"
	s_total=`cat $stocklist | grep -Ev ^$ | wc -l`
	s_count=0
	while [ $s_count -lt $s_total ]
		do
		tailc=200
		headc=$(($s_count + $tailc))
		s_list=`cat $stocklist | head -n $headc | tail -n $tailc | grep -Ev ^$`; s_list=`echo $s_list | sed 's# #,#g'`;
		curl -s "${quote_base_url}s=${s_list}&f=${propertyvalues}&e=.csv" > "$directory/stocks_${s_count}-${headc}.csv";
		sleep 1;
		s_count=$(($s_count + $tailc))
	done
}

function get_historic_values_append()
{
	directory="historical"
	ls $directory | while read stockfile;
	do
		stock=`echo $stockfile | sed 's#.csv##'`
		stock_file=$directory/$stockfile
		# get the last line appended to the file
		# assumes last date is at the bottom of the file (as it should be)
		lastdate=(`tail -n 1 | sed 's#,.*##' | sed 's#-# #g'`)
		lastday=${lastdate[3]}
		lastmonth=${lastdate[2]}
		lastyear=${lastdate[1]}
		# make a temp file to store the new data
		tempfile=`mktemp tempnewdata.XXXXXXX`
		# grab the data from that date on
		# example: http://ichart.yahoo.com/table.csv?s=IBM&a=0&b=0&c=0&d=12&e=31&f=2012&g=w
		curl -s "${historial_base_url}s=${stock}&a=$lastmonth&b=$lastday&c=$lastyear&d=${monthnum}&e=${day}&f=${year}" > "$tempfile"
			#append data lines of new data in reverse order to stock data file
			tail -n $(($(`cat $tempfile | wc -l`) - 1)) $tempfile >> $stock_file
			rm $tempfile;
			sleep 1;
	done
}

if [ "${1}" == "historic_all_stocks" ]
	then
	get_historic_values_all_stocks
elif [ "${1}" == "update_historic_all_stocks" ]
	then
	get_historic_values_append
elif [ "${1}" == "current_all_stocks" ]
	then
	get_current_values_all_stocks
elif [ "${1}" == "current_all_stocks_batch" ]
	then
	get_current_values_all_stocks_batch
else
	echo "-----------------------------"
	echo "What do you want to do?"
	echo "bash scrape_yahoo.sh <command>"
	echo ""
	echo "Current Commands are:"
	echo "historic_all_stocks"
	echo "update_historic_all_stocks"
	echo "current_all_stocks"
	echo "current_all_stocks_batch"
	echo ""
fi
