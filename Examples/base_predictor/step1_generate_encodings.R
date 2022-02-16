devtools::install_github("daiyun02211/Geo2vec")

library(Geo2vec)
library(rtracklayer)
library(BSgenome.Hsapiens.UCSC.hg38)
library(EnsDb.Hsapiens.v79)
library(parallel)

PATH_TO_Geoplus <- 
data_dir <- paste0(PATH_TO_Geoplus, "Examples/base_predictor/")
input <- import.bed(paste0(data_dir, 'example.bed'))

target_dir <- paste0(data_dir, 'processed/')
sequence_feature <- encSeq(input, BSgenome.Hsapiens.UCSC.hg38, 250, type='token')
saveRDS(sequence_feature, paste0(target_dir, 'infer_token.rds'))

tx <- 'all' # 'long'
long_tx <- tx == 'long'
encoding <- encGeo(input, EnsDb.Hsapiens.v79, type='chunkTX', exon_only=T, long_tx=long_tx, mRNA=T)

result <- mclapply(encoding, resizeChunk, window=17, mc.cores=4)

xbytx_id <- result %>% names() %>% strsplit(split='-') %>% unlist()
x_id <- xbytx_id[seq(1, length(xbytx_id), 2)]
x_order <- x_id %>% as.integer() %>% order()
out <- result[x_order]
out <- out %>% data.frame() %>% t()

saveRDS(x_id, paste0(target_dir, 'infer_chunkTX_', tx, '_xid.rds'))
saveRDS(out, paste0(target_dir, 'infer_chunkTX_', tx, '_out.rds'))


