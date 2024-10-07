# usage: Rscript add_genes.R clusters.tsv expressions.tsv is_rna_seq [pval logFC  num_genes]
# exprs: a .tsv table genes in rows, samples in columns; between-sample normalized
# is_rna_seq: if expression data are from RNA-seq, set 1 and provide log2(x+1) of counts 
# output: clusters.with_genes.tsv 
suppressPackageStartupMessages(library("limma"))
suppressPackageStartupMessages(library("edgeR"))

args <- commandArgs(trailingOnly = TRUE)

clusters_file <- args[[1]]
exprs_file <- args[[2]]
rna_seq <- args[[3]]
outfile <- paste0(sub(".tsv","",clusters_file),".with_genes.tsv")

# check if the argument is available, if not, set a default value
pval_cutoff <- ifelse(length(args) >= 4, as.numeric(args[[4]]), 0.05)
logFC_cutoff <- ifelse(length(args) >= 5, as.numeric(args[[5]]), 1)
num_genes <- ifelse(length(args) >= 6, as.numeric(args[[6]]), 500) 

clusters <- read.delim(clusters_file, row.names = 1)
#clusters$samples <- strsplit(as.character(clusters$samples),' ',fixed=TRUE)

exprs <- t(read.delim(exprs_file, row.names = 1))
rownames(exprs) <- gsub("\\.", "-",rownames(exprs))
exprs <- exprs[sort(rownames(exprs)),]
exprs <- t(exprs)

if (rna_seq){
 
 exprs <- 2**exprs-1
 exprs <- DGEList(exprs)
}

find_DE_genes <- function(exprs, dm, rna_seq, pval_cutoff, logFC_cutoff, num_genes) {

    group1 <- "bic"
    group2 <- "bg"
    all_groups <- c(group1,group2)

    group <- as.factor(apply(dm, 1, function(x) { all_groups[as.logical(x)]}))
    
    #if (rna_seq){
    #keep_exprs <- filterByExpr(exprs, group=group)
    #cat(paste0("Genes passed filterByExprs: ", length(keep_exprs[keep_exprs])))
    #exprs <- exprs[keep_exprs,, keep.lib.sizes=FALSE]
    #exprs <- calcNormFactors(exprs, method = "upperquartile") # 
    #}
    
    # making simple contrast matrix
    contrasts_matrix <- data.frame(c(1,-1))
    rownames(contrasts_matrix) <- all_groups
    colnames(contrasts_matrix) <- c(paste0(group1," - ",group2))
    contrasts_matrix <- as.matrix(contrasts_matrix)

    if (rna_seq){
        voom_results <-  voom(exprs, dm, plot = F, save.plot=F)
        fit <- lmFit(voom_results, dm)
    } else{
        fit <- lmFit(exprs, dm)
    }
    contr_fit <- contrasts.fit(fit, contrasts_matrix)
    result <- eBayes(contr_fit)
    #table_res <- topTable(result, adjust="BH",resort.by="P",p.value=0.05,lfc=2,confint=TRUE,number=500)
    #if (dim(table_res)[[1]]<=1) {
    table_res <- topTable(result, adjust="BH",resort.by="P",p.value=pval_cutoff,lfc=logFC_cutoff,confint=TRUE,number=num_genes)
    #}
    de_up <- sort(row.names(table_res[table_res$logFC>0,]))
    de_down <- sort(row.names(table_res[table_res$logFC<0,]))
    de_both <- unique(sort(c(de_up,de_down)))
    n_genes = length(de_both)
    
    #print(c(length(de_up),length(de_down)))
    de_up <- trimws(paste(de_up, collapse=" "))
    de_down <- trimws(paste(de_down, collapse=" "))
    de_both <- trimws(paste(de_both, collapse=" "))

    return(list("genes"=de_both,"n_genes"=n_genes,"genes_up"=de_up,"genes_down"=de_down))
}

make_design_matrix <- function(bic_samples, exprs) {
  dm <- t(as.data.frame(matrix(0, ncol = 2, nrow = length(colnames(exprs))), row.names = colnames(exprs)))
  row.names(dm) <- c("bic", "bg")
  dm <- t(dm)
    
  bic_samples <- gsub("\\.", "-", bic_samples)
  matching_rows <- row.names(dm) %in% bic_samples
  dm[matching_rows, "bic"] = 1
  dm[!matching_rows, "bg"] = 1
  return(dm)
}

add_genes_to_clusters <- function(row,exprs=exprs,rna_seq=rna_seq, pval_cutoff=pval_cutoff, logFC_cutoff=logFC_cutoff, num_genes=num_genes) {
    bic_samples <- row["samples"]
    bic_samples <- as.character(bic_samples)
    bic_samples <- strsplit(bic_samples,' ',fixed=TRUE)
    bic_samples <- unlist(bic_samples)
    dm <- make_design_matrix(bic_samples,exprs)
    biomarkers <- find_DE_genes(exprs, dm,rna_seq, pval_cutoff, logFC_cutoff, num_genes)
    return (biomarkers)
}



df <- apply(clusters,1,add_genes_to_clusters,exprs=exprs,rna_seq=rna_seq, pval_cutoff=pval_cutoff, logFC_cutoff=logFC_cutoff, num_genes=num_genes)
df <- as.data.frame(do.call(cbind,df))
df<-t(df)

df<-data.frame(df)
df$genes<- as.character(df$genes)
df$genes_up <- as.character(df$genes_up)
df$genes_down <- as.character(df$genes_down)
df$n_genes <- as.integer(df$n_genes)
# add new columns
clusters<-cbind(clusters,df)


write.table(clusters,file = outfile,sep = "\t",quote = FALSE, col.names=NA)

cat(outfile,"\n")
