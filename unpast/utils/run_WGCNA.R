# usage: Rscript run_WGCNA.R binarized_expressions.tsv [deepSplit:0,1,2,3,4] [detectCutHeight:(0-1)] [network type:signed_hybrid|unsigned] [max_power precluster:T/F]
suppressPackageStartupMessages(library("WGCNA"))

args <- commandArgs(trailingOnly = TRUE)

fileBinExprs <- args[[1]]
fileModules <- paste0(sub(".tsv","",fileBinExprs),".modules.tsv")

deepSplit <- as.integer(args[[2]])
detectCutHeight <- as.numeric(args[[3]])
nt <- args[[4]]

if (nt=="signed_hybrid"){ 
    nt <- "signed hybrid"
}
#nt <- "signed hybrid" # networkType = "unsigned", "signed hybrid"

max_power <- as.numeric(args[[5]])

precluster <- as.logical(args[[6]])


datExpr <- read.csv(fileBinExprs,check.names=FALSE,sep = "\t",header = TRUE,row.names=1)
datExpr[] <- lapply(datExpr, as.numeric)

#### finding power threshold #### 
powers = c(1:max_power) #c(c(1:10), seq(from = 12, to=20, by=2))
rsqcuts <- seq(from = 0.90, to=0.05, by=-0.05)
i = 0
power <- NA
while ((is.na(power)) && (i<length(rsqcuts))){
# Call the network topology analysis function
 i<-i+1
 sft <- pickSoftThreshold(datExpr, powerVector = powers,verbose = 0,
                          networkType = nt,RsquaredCut =rsqcuts[[i]],)  
 power <- sft$powerEstimate
}
if (is.na(power)){
 print(cat("\nno power selected with any R^2 threshold between 0.9 and 0.05; was set to 1","\n"))
 power <- 1
}

print(cat("\nR^2 threshold:",rsqcuts[[i]],"power:",power,"\n"))
print(cat("networkType:",nt,"\n"))



#### find modules ####
# if maxBlockSize is not exceeded, no pre-clustering is performed
# not performed if < 100 features 
if (precluster){
    maxBlockSize = max(100,as.integer(ncol(datExpr)/2))
} else {
maxBlockSize =as.integer(ncol(datExpr)+1) #  to switch off pre-clustering, much faster and less modules
}
print(cat("n_features:",ncol(datExpr),"\n"))
print(cat("maxBlockSize:",maxBlockSize,"\n"))
print(cat("pre-clustering:",precluster,"\n"))
# number of pre-clustering centers 
# equals min(ncol(datExpr)/20, 200), minimal for possible maxBlockSize is 5
#nPreclusteringCenters = as.integer(min(ncol(datExpr)/20, 100*ncol(datExpr)/maxBlockSize)) # default
#nPreclusteringCenters = as.integer(min(ncol(datExpr)/20, 10*ncol(datExpr)/maxBlockSize)) 
#print(cat("nPreclusteringCenters:",nPreclusteringCenters,"\n"))

net = blockwiseModules(datExpr, power = power,
TOMType = "unsigned", 
networkType = nt, 
minModuleSize = 2,
numericLabels = TRUE,
maxBlockSize = maxBlockSize, 
#nPreclusteringCenters = nPreclusteringCenters, 
detectCutHeight = detectCutHeight, #detectCutHeight = 0.995,
mergeCutHeight =0.05, # only modules with ME correlated with r>1-0.05 are merged
deepSplit = deepSplit,
verbose = 0)

moduleLabels = net$colors

sink(fileModules)
cat(paste0("module","\t","size","\t","genes","\n",sep=' '))
for (i in unique(moduleLabels)){
    cat(paste0(i,"\t",length(names(moduleLabels[moduleLabels==i])),"\t",paste(names(moduleLabels[moduleLabels==i]),collapse=" "),"\n"))
}
sink()

# save eigengenes 
#print(paste0(sub(".tsv","",fileBinExprs),".MEs.tsv"))
#write.table(net$MEs,file = paste0(sub(".tsv","",fileBinExprs),".MEs.tsv"),sep = "\t",quote = FALSE)
#cat(fileModules,"\n")