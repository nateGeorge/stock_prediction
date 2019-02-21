DEFAULT.STORAGE = '/home/nate/Dropbox/data/eod_data/'

create.reticulate.env <- function(){
  conda_create("r-reticulate", packages='python=3.5')
  # install pytables for reading hdf5 -- should be installled with pandas
  # conda_install("r-reticulate", packages='pytables')
  conda_install("r-reticulate", packages='pandas')
}

get.latest.eod <- function(){
  files <- list.files(DEFAULT.STORAGE, '*.h5')
  info <- file.info(paste(DEFAULT.STORAGE, files, sep=''))
  latest.f <- rownames(info[order(info$mtime, decreasing=TRUE), ])[1]
  chunks <- strsplit(latest.f, '_')[[1]]
  date <- chunks[length(chunks)]
  date <- substr(date, 1, nchar(date) - 3)
  return(date)
}

get.latest.h5.file <- function(){
  latest.eod <- get.latest.eod()
  return(paste0(DEFAULT.STORAGE, 'EOD_', latest.eod, '.h5'))
}


latest.eod <- NULL

if (is.null(latest.eod)) {
  latest.eod <- get.latest.eod()
}

load.pandas.df <- function() {
  # load with pandas -- currently not working properly
  library(reticulate)
  # doesn't seem to be necessary, but just in case you need it...
  # https://rstudio.github.io/reticulate/articles/versions.html
  # path_to_python <- "/home/nate/anaconda3/bin/python"
  # use_python(path_to_python)
  pd <- import("pandas")
  # doesn't work well because there are duplicate row labels (date as index)
  df <- pd$read_hdf(get.latest.h5.file())
}

load.latest.feather <- function(){
  # can load with feather, but then have to create ft file from pandas
  library(feather)
  latest.eod <- get.latest.eod()
  df <- read_feather(paste0(DEFAULT.STORAGE, 'EOD_', latest.eod, '.ft'))
  return(df)
}

# tried loading hdf, but couldn't read blosc-compression
# https://www.kaggle.com/jeffmoser/read-hdf-function-for-r
# https://www.r-bloggers.com/load-a-pythonpandas-data-frame-from-an-hdf5-file-into-r/
# https://github.com/hhoeflin/hdf5r