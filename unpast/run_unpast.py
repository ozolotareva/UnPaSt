#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

def unpast(exprs_file: pd.DataFrame,
        basename: str ='',
        out_dir: str ="./",
        save: bool =True,
        load: bool = False,
        ceiling: float= 3,
        bin_method: str = "kmeans", 
        clust_method: str  = "WGCNA", 
        min_n_samples: int = 5, 
        show_fits: list = [],
        pval: float = 0.01,
        directions: list = ["DOWN","UP"], 
        modularity: float =1/3,
        similarity_cutoffs = -1, # for Louvain
        ds: int = 3,
        dch: float = 0.995,
        max_power: int = 10, 
        precluster: bool = True,
        rpath: str ="", # for WGCNA
        #cluster_binary: bool = False, 
        merge: float = 1,
        seed: int = 42,
        verbose: bool = False,
        plot_all: bool = False,
        e_dist_size: int = 10000,
        standradize: bool = True):
    
    import sys
    from time import time
    from unpast.utils.method import prepare_input_matrix
    
    start_time = time()
    
    # make sure that out_dir has '/' suffix
    if out_dir[-1] != '/':
        out_dir += '/'
        
    if not basename: 
        from datetime import datetime
        now = datetime.now()
        basename = "unpast_" + now.strftime("%y.%m.%d_%H:%M:%S")
        print("set output basename to", basename, file = sys.stdout)
        
    # read inputs
    exprs = pd.read_csv(exprs_file, sep="\t",index_col=0)
    if verbose:
        print("Read input from ",exprs_file, file=sys.stdout)
        print("\t{} features x {} samples".format(exprs.shape[0],exprs.shape[1]), file=sys.stdout)
    if exprs.shape[1]<5:
        print("Input matrix must contain at least 5 samples (columns), but only %s columns are found."%
              exprs.shape[1], file=sys.stderr)
        
    if exprs.shape[0]<2:
        print("Input matrix must contain at least 2 features (rows), but only %s rows are found."%
              exprs.shape[0], file=sys.stderr)
    
    if exprs.shape[1]<5 or exprs.shape[0]<2:
        sys.exit(1)
    
    if min_n_samples < 2:
        print("The minimal number of samples in a bicluster `min_n_samples` must be >= 2.",
              file = sys.stderr)
        sys.exit(1)
    
    if min_n_samples > int((exprs.shape[1]/2) // 1):
        print("The minimal number of samples in a bicluster `min_n_samples` must be not greater than half of the cohort size.", file = sys.stderr)
        sys.exit(1)
        
    if verbose:
        print("Mininal number of samples in a bicluster:",min_n_samples ,file=sys.stdout)
    if min_n_samples < 5:
        print("\tmin_n_samples is recommended to be >= 5", file= sys.stderr)
    
    e_dist_size = max(e_dist_size,int(1.0/pval*10))
    if verbose:
        print("The size of empirical SNR distribution: %s"%e_dist_size, file=sys.stdout)

    # check if input is standardized (between-sample)
    # if necessary, standardize and limit values to [-ceiling,ceiling]
    exprs = prepare_input_matrix(exprs,
                                 min_n_samples=min_n_samples,
                                 standradize=standradize,
                                 ceiling=ceiling,
                                 verbose = verbose
                                )
        
    ######### binarization #########
    from unpast.utils.method import binarize
    
    binarized_features, stats, null_distribution  = binarize(out_dir+basename, exprs=exprs,
                                 method=bin_method, save = save, load=load,
                                 min_n_samples = min_n_samples,pval=pval,
                                 plot_all = plot_all,show_fits = show_fits,
                                 verbose= verbose,seed=seed,
                                 prob_cutoff=0.5, n_permutations=e_dist_size)
    
    bin_data_dict = {}
    stats = stats.loc[stats["pval"]<=pval,:]
    features_up = set(stats.loc[stats["direction"]=="UP",:].index.values)
    features_up = sorted(set(binarized_features.columns.values).intersection(set(features_up)))
    features_down = stats.loc[stats["direction"]=="DOWN",:].index.values
    features_down = sorted(set(binarized_features.columns.values).intersection(set(features_down)))

    df_up = binarized_features.loc[:,features_up]
    df_down = binarized_features.loc[:,features_down]
    if directions[0] == "BOTH":
        bin_data_dict["BOTH"] = pd.concat([df_up,df_down],axis=1)
    else:
        bin_data_dict["UP"]  = df_up
        bin_data_dict["DOWN"]  = df_down 
        
    ######### gene clustering #########
    if verbose:
        print("Clustering features ...\n",file=sys.stdout)
    feature_clusters, not_clustered, used_similarity_cutoffs = [], [], []
    if clust_method == "Louvain":
        from unpast.utils.method import run_Louvain
        from unpast.utils.method import get_similarity_jaccard
        
        for d in directions:
            df = bin_data_dict[d]
            if df.shape[0]>1:

                similarity = get_similarity_jaccard(df,verbose = verbose)
                #similarity = get_similarity_corr(df,verbose = verbose)

                if similarity_cutoffs  == -1: # guess from the data
                    similarity_cutoffs = np.arange(0.3,0.9,0.01)
                # if similarity cuttofs is a single value turns it to a list
                try: 
                    similarity_cutoffs = [elem for elem in similarity_cutoffs]
                except:
                    similarity_cutoffs = [similarity_cutoffs]

                # if modularity m is defined, choses a similarity cutoff corresponding to this modularity
                # and rund Louvain clustering
                modules, single_features, similarity_cutoff = run_Louvain(similarity,
                                                                          similarity_cutoffs = similarity_cutoffs,
                                                                          m = modularity, 
                                                                          verbose = verbose)
                used_similarity_cutoffs.append(similarity_cutoff)
                feature_clusters+= modules
                not_clustered+= single_features
            elif df.shape[0]==1:
                not_clustered+= list(df.index.values)
                used_similarity_cutoffs.append(None)
        used_similarity_cutoffs = ",".join(map(str,used_similarity_cutoffs))
        
    elif "WGCNA" in clust_method:
        if clust_method == "iWGCNA":
            from unpast.utils.method import run_WGCNA_iterative
            WGCNA_func = run_WGCNA_iterative
        else:
            from unpast.utils.method import run_WGCNA
            WGCNA_func = run_WGCNA

        for d in directions:
            # WGCNA tmp file prefix
            tmp_prefix = out_dir+basename+ "."+bin_method+".pval="+str(pval)+".seed="+str(seed)+"."+d
            df = bin_data_dict[d] 
            if df.shape[0]>1:
                modules, single_features = WGCNA_func(df,tmp_prefix=tmp_prefix, 
                                                      deepSplit=ds,detectCutHeight=dch,nt = "signed_hybrid",
                                                      max_power = max_power, precluster=precluster,
                                                     verbose = verbose,rpath = rpath)  
                feature_clusters+= modules
                not_clustered+= single_features
    
    else:
        print("'clust_method' must be 'WGCNA', 'iWGCNA', or 'Louvain'.",file=sys.stderr)
    
    ######### making biclusters #########
    if len(feature_clusters)==0:
        if verbose:
            print("No biclusters found",file = sys.stderr)
        return pd.DataFrame()
        
    from unpast.utils.method import make_biclusters
    biclusters = make_biclusters(feature_clusters,
                                 binarized_features,
                                 exprs,
                                 method = bin_method,
                                 merge = merge,
                                 min_n_samples=min_n_samples,
                                 min_n_genes=2,
                                 seed = seed,
                                 cluster_binary=False,
                                 verbose = verbose)

    
    from unpast.utils.io import write_bic_table
    suffix  = ".seed="+str(seed)+".bin="+bin_method+",pval="+str(pval)+",clust="+clust_method+",direction="+"-".join(directions)
    if "WGCNA" in clust_method:
        suffix2 = ",ds="+str(ds)+",dch="+str(dch)+",max_power="+str(max_power)+",precluster="+str(precluster)
        modularity, similarity_cutoff = None, None
    elif clust_method == "Louvain":
        suffix2 = ",m="+str(round(modularity,2))
        ds, dhs = None, None
    write_bic_table(biclusters, out_dir+basename+suffix+suffix2+".biclusters.tsv",to_str=True,
                    add_metadata=True, seed = seed, min_n_samples = min_n_samples, pval = pval,
                    bin_method = bin_method, clust_method = clust_method, directions = directions,
                    similarity_cutoff = used_similarity_cutoffs,
                    m=modularity, ds = ds, dch = dch, 
                    max_power=max_power, precluster=precluster,
                    merge = merge)

    if verbose:
        print(out_dir+basename+suffix+suffix2+".biclusters.tsv", file = sys.stdout)
        print("Total runtime: {:.2f} s".format(time()-start_time ), file = sys.stdout)
    
    return biclusters    

def parse_args():
    parser = argparse.ArgumentParser("UnPaSt identifies differentially expressed biclusters in a 2-dimensional matrix.")
    parser.add_argument('--seed',metavar=42, default=42, type=int, help="random seed")
    parser.add_argument('--exprs', metavar="exprs.z.tsv", required=True, 
                        help=".tsv file with between-sample normalized input data matrix. The first column and row must contain unique feature and sample ids, respectively. At least 5 samples (columns) and at least 2 features (rows) are required.")
    parser.add_argument('--out_dir', metavar="./", default="./", help  = 'output folder')
    parser.add_argument('--basename', metavar="biclusters.tsv", default = False, type=str, help  = 'output files prefix. If not specified, will be set to "results_"yy.mm.dd_HH:MM:SS""')
    parser.add_argument('--ceiling', default=3, metavar="3",  type=float, required=False, 
                        help="Absolute threshold for z-scores. For example, when set to 3, z-scores greater than 3 are set to 3 and z-scores less than -3 are set to -3. No ceiling if set to 0.")
    parser.add_argument('-s','--min_n_samples', metavar=5, default=5, type=int, help  = 'The minimal number of samples in a bicluster `min_n_samples` must be >= 2 and not greater than half of the cohort size.')
    parser.add_argument('-b','--binarization', metavar="kmeans", default="kmeans", type=str,
                        choices=["kmeans","ward",'GMM'], help='binarization method')
    parser.add_argument('-p','--pval', metavar=0.01, default=0.01, type=float, help  = 'binarization p-value')
    parser.add_argument('-c','--clustering', metavar="WGCNA", default="WGCNA", type=str,
                        choices=['Louvain', 'WGCNA'], help='feature clustering method')
    # Louvain parameters
    parser.add_argument('-m','--modularity', default=1/3, metavar="1/3", type=float, help='Modularity corresponding to a cutoff for similarity matrix (Louvain clustering)')
    parser.add_argument('-r','--similarity_cutoffs', default=-1, metavar="-1", type=float, help='A cutoff or a list of cuttofs for similarity matrix (Louvain clustering). If set to -1, will be chosen authomatically from [1/5,4/5] using elbow method.')
    # WGCNA parameters 
    parser.add_argument('--ds', default=3, metavar="3", type=int,choices=[0,1,2,3,4], help='deepSplit parameter, see WGCNA documentation')
    parser.add_argument('--dch', default=0.995, metavar="0.995", type=float, help='dynamicTreeCut parameter, see WGCNA documentation')
    parser.add_argument('--bidirectional', action='store_true', help='Whether to cluster up- and down-regulated features together.')
    parser.add_argument('--rpath', default="", metavar="", type=str, help='Full path to Rscript.')
    #parser.add_argument('--merge', default=1, metavar="1", type=float,help = "Whether to merge biclustres similar in samples with Jaccard index not less than the specified.")
    parser.add_argument('--load_binary', action='store_true', help = "loads binarized features from <basename>.<bin_method>.seed=42.binarized.tsv, statistics from *.binarization_stats.tsv and the background SNR distribution from <basename>.<bin_method>.n=<e_dist_size>.seed=42.background.tsv")
    parser.add_argument('--save_binary', action='store_true', help = "saves binarized features to a file named as <basename>.<bin_method>.seed=42.binarized.tsv. When feature clustering method is WGCNA, binarized features will be always saved. Also, files *.binarization_stats.tsv and *.background.tsv with binarization statistincs and background SNR distributions respectively will be created")
    parser.add_argument('-v','--verbose', action='store_true')
    #parser.add_argument('--plot', action='store_true', help = "show plots")
    
    
    return parser.parse_args()
    

if __name__ == "__main__":
    args = parse_args()
    
    directions = ["DOWN","UP"]
    if args.bidirectional:
        directions = ["BOTH"]
    
    biclusters = unpast(args.exprs, args.basename, out_dir=args.out_dir,  
                save = args.save_binary, load = args.load_binary,
                ceiling = args.ceiling,
                bin_method = args.binarization, 
                clust_method = args.clustering,
                pval = args.pval,
                directions = directions,
                min_n_samples = args.min_n_samples, 
                show_fits = [],
                modularity = args.modularity, similarity_cutoffs = args.similarity_cutoffs, # for Louvain
                ds = args.ds, dch = args.dch, rpath=args.rpath, precluster=True, # for WGCNA
                cluster_binary = False, 
                #merge = args.merge,
                seed = args.seed,
                #plot_all = args.plot,
                verbose = args.verbose)