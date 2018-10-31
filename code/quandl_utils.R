DEFAULT.STORAGE = '/home/nate/Dropbox/data/eod_data/'

get.latest.eod <- function(){
  files <- list.files(DEFAULT.STORAGE, '*.h5')
  info <- file.info(paste(DEFAULT.STORAGE, files, sep=''))
  latest.f <- rownames(info[order(info$mtime, decreasing=TRUE), ])[1]
  chunks <- strsplit(latest.f, '_')[[1]]
  date <- chunks[length(chunks)]
  date <- substr(date, 1, nchar(date) - 3)
  return(date)
}


latest.eod <- NULL

if (is.null(latest.eod)) {
  latest.eod <- get.latest.eod()
}

library(feather)

df <- read_feather(paste0(DEFAULT.STORAGE, 'EOD_', latest.eod, '.ft'))

# tried loading hdf, but couldn't read blosc-compression
# https://www.kaggle.com/jeffmoser/read-hdf-function-for-r
# https://www.r-bloggers.com/load-a-pythonpandas-data-frame-from-an-hdf5-file-into-r/
# https://github.com/hhoeflin/hdf5r