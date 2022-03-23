devtools::install_github("daiyun02211/Geo2vec")

library(Geo2vec)
library(rtracklayer)
library(BSgenome.Hsapiens.UCSC.hg19)
library(TxDb.Hsapiens.UCSC.hg19.knownGene)
library(parallel)

tissueSeq <- function(gr, BSgenome){
  seq <- as.character(getSeq(BSgenome, gr))
  seq <- strsplit(seq, '')
  token_seq <- vector(mode='list', length=length(seq))
  for (idx in c(1:length(seq))){
    token_seq[idx] = list(as.integer(unlist(seq[idx]) == 'A') 
                          + 2 * as.integer(unlist(seq[idx]) == 'C') 
                          + 3 * as.integer(unlist(seq[idx]) == 'G') 
                          + 0 * as.integer(unlist(seq[idx]) == 'T'))
  }
  return(token_seq)
}

PATH_TO_Geoplus <- 
data_dir <- paste0(PATH_TO_Geoplus, "Examples/base_predictor/")
input <- import.bed(paste0(data_dir, 'example.bed'))

target_dir <- paste0(data_dir, 'processed/')
sequence_feature <- tissueSeq(input, BSgenome.Hsapiens.UCSC.hg19)
saveRDS(sequence_feature, paste0(target_dir, 'infer_token.rds'))

tx <- 'all'
long_tx <- tx == 'long'
encoding <- encGeo(input, TxDb.Hsapiens.UCSC.hg19.knownGene, type='chunkTX', exon_only=T, long_tx=long_tx, mRNA=T)
result <- mclapply(encoding, resizeChunk, window=17, region=TRUE, mc.cores=4)

xbytx_id <- result %>% names() %>% strsplit(split='-') %>% unlist()
x_id <- xbytx_id[seq(1, length(xbytx_id), 2)]
x_order <- x_id %>% as.integer() %>% order()
out <- result[x_order]
out <- out %>% data.frame() %>% t()

saveRDS(x_id, paste0(target_dir, 'infer_chunkTX_', tx, '_xid.rds'))
saveRDS(out, paste0(target_dir, 'infer_chunkTX_', tx, '_out.rds'))