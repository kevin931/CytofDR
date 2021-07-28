library(Seurat)
library(tidyverse)

args <- commandArgs(trailingOnly = TRUE)

read_path <- args[1]
save_path <- args[2]
col_names <- as.logical(args[3])

df <- read_delim(read_path, col_names = col_names, delim="\t")

if (length(args) > 3) {
  n_genes <- as.numeric(args[4])
} else if (any(colSums(df)==0)) {
  df <- df[,-which(colSums(df)==0)]
}

names <- colnames(df)
df <- t(as.matrix(df))
rownames(df) <- names
colnames(df) <- paste0("Obs", 1:ncol(df))

df <- CreateSeuratObject(df, project = "df")

if (length(args)>3) {
  df <- FindVariableFeatures(df, selection.method = "vst", nfeatures = n_genes)
  df <- ScaleData(df, do.scale = F, do.center = F)
  df <- RunPCA(df, features = VariableFeatures(object = df))
} else {
  df <- ScaleData(df, do.scale = F, do.center = F)
}

if (nrow(df)>20) {
  df <- RunPCA(df, features=names)
  df <- FindNeighbors(df, dims=1:20)
} else{
  # Use the full matrix for clustering
  mat <- t(as.matrix(df@assays$RNA@data))
  df[["pca"]] <- CreateDimReducObject(embeddings=mat, key="PCA_", assay="RNA")
  df <- FindNeighbors(df, dims=1:nrow(df))
}

df <- FindClusters(df, resolution = 0.8)

labels <- as_tibble(Idents(df))
write_delim(labels,
            save_path,
            delim="\t", col_names = F)