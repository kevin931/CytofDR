library(tidyverse)
library(destiny)

args <- commandArgs(trailingOnly = TRUE)

read_path <- args[1]
dist_metric <- args[2]
save_path <- args[3]

embedding_path <- paste0(save_path, "/embedding/")
dir.create(embedding_path)
embedding_path <- paste0(embedding_path, "diffmap.txt")

time_path <- paste0(save_path, "/time.csv")

df <- read_delim(read_path,
                 col_names = T,
                 delim="\t")

start_time <- Sys.time()
diffmap <- DiffusionMap(df, distance = dist_metric)
end_time <- Sys.time()

time <- as.numeric(difftime(end_time, start_time, units = "secs"))
time <- tibble("diffmap", time)

embedding <- tibble(diffmap$DC1, diffmap$DC2)

write_delim(x = embedding,
            file=embedding_path,
            col_names = FALSE,
            delim = "\t")
write_csv(time, file=time_path, col_names = FALSE)