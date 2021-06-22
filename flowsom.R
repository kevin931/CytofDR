library(FlowSOM)
library(tidyverse)

args <- commandArgs(trailingOnly = TRUE)

read_path <- args[1]
clusters <- as.numeric(args[2])
save_path <- args[3]
col_names <- as.logical(args[4])

df <- read_delim(read_path,
                 col_names = col_names,
                 delim="\t")

if (length(args) > 4) {
  drop_cols <- as.numeric(args[5:length(args)])
  df <- df[, -drop_cols]
}

# Median Impute Missing Values
if(any(is.na(df))) {
  for (col in 1:ncol(df)){
    if(any(is.na(df[,col]))) {
      df[which(is.na(df[,col])), col] <- median(df[[col]], na.rm=T)
    }
  }
}

print(colnames(df))

df_flow <- flowCore::flowFrame(as.matrix(df))

clustering <- FlowSOM(df_flow, scale = F, colsToUse = 1:ncol(df), nClus=clusters)
labels <- as_tibble(GetMetaclusters(clustering))

print(paste0("Saving to: ", save_path))
write_delim(labels, save_path, delim = "\t", col_names = FALSE)