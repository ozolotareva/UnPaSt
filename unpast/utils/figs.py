import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import gcf
import matplotlib.pyplot as plt


def draw_heatmap2(
    exprs,
    biclusters=pd.DataFrame(),
    annot=None,
    color_dict=None,
    figsize=(20, 10),
    dendrogram_ratio=(0.01, 0.0),  # space for dendrogram
    colors_ratio=(0.005, 0.02),
    bicluster_colors="black",  #
    no_bic_columns=False,  # do not show bicluster annotation in columns
    no_legend=False,
    no_cbar=False,
    cluster_rows=True,
    cluster_cols=False,  # enable hierarchical clustering of columns
    cluster_columns=True,
    xlabel="samples",
    col_labels=True,
    row_labels=False,
    color_range=(-3, 3),
    bic_prefix="bic_",
    plot_bg_genes=False,
    no_row_colors=True,
    highlight_row_labels=[],
    row_labels_black=False,
):
    """* exprs - expressions of genes to plot
       * biclusters in UnPaSt format
       * annot - annotation for samples, columns - subtypes, rows - samples 
       * color_dict - how to color each label
       * bicluster_colors - color for bicluster annotation "black","auto" or list 
       """
    bic_names = []
    ordered_genes = []
    row_colors = None
    col_colors = None
    sample_order = exprs.columns.values
    if type(annot) != type(None):
        cols = list(annot.columns.values)
        if not no_row_colors:
            row_colors = pd.DataFrame(
                data="white",
                index=exprs.index.values,
                columns=[bic_prefix + str(x) for x in biclusters.index.values],
            )

    else:
        annot = pd.DataFrame(index=exprs.columns)
        if not no_row_colors:
            row_colors = pd.DataFrame(index=exprs.index.values)
        cols = []
    if type(color_dict) != type(None):
        pass

    # list of bicluster colors
    if biclusters.shape[0] > 0:
        bic_colors = []
        if bicluster_colors == "black" or bicluster_colors == "redblue":
            bic_colors = ["black"] * biclusters.shape[0]
        elif bicluster_colors == "auto":
            palette = sns.color_palette("colorblind")
            # Get the first n colors from the palette
            bic_colors = sns.color_palette("colorblind",biclusters.shape[0]).as_hex()
            print("colors:", bic_colors)

        else:
            bic_colors = bicluster_colors
        bic_colors = dict(zip(biclusters.index.values, bic_colors))

        for row in biclusters.iterrows():
            bic_id = bic_prefix + str(row[0])
            s = row[1]["samples"]
            g = sorted(row[1]["genes_up"]) + sorted(row[1]["genes_down"])

            annot[bic_id] = "white"
            annot.loc[list(s), bic_id] = bic_colors[row[0]]

            if not no_row_colors:
                row_colors.loc[g, bic_id] = bic_colors[row[0]]

            g = [x for x in g if not x in ordered_genes]
            ordered_genes += g
            bic_names.append(bic_id)
    else:
        plot_bg_genes = True

    if plot_bg_genes:
        ordered_genes = ordered_genes + sorted(
            set(exprs.index.values).difference(set(ordered_genes))
        )

    col_colors = annot.loc[:, bic_names + cols]

    for col in reversed(cols):
        col_color_map = color_dict[col]
        col_colors[col] = col_colors[col].apply(lambda x: col_color_map[x])
        for subt in list(col_color_map.keys()):
            subt_samples = annot.loc[annot[col] == subt, :].index
            if cluster_columns:
                new_sample_order = [
                    x for x in sample_order if x not in subt_samples
                ] + [x for x in sample_order if x in subt_samples]
                sample_order = new_sample_order

    for col in reversed(bic_names):
        ordered_colors = [x for x in set(annot[col]) if not x == "white"] + ["white"]
        for subt in ordered_colors:
            subt_samples = annot.loc[annot[col] == subt, :].index
            if cluster_columns:
                new_sample_order = [
                    x for x in sample_order if x not in subt_samples
                ] + [x for x in sample_order if x in subt_samples]
                sample_order = new_sample_order

    # change column colors if redblue is chosen
    if bicluster_colors == "redblue":
        bic_ids = biclusters.loc[biclusters["direction"] == "UP", :].index.values
        bic_ids = [bic_prefix + str(bic_id) for bic_id in bic_ids]
        d = {"black": "red", "white": "blue"}
        col_colors.loc[:, bic_ids] = col_colors.loc[:, bic_ids].applymap(lambda x: d[x])
        bic_ids = biclusters.loc[biclusters["direction"] == "DOWN", :].index.values
        bic_ids = [bic_prefix + str(bic_id) for bic_id in bic_ids]
        d = {"black": "blue", "white": "red"}
        col_colors.loc[:, bic_ids] = col_colors.loc[:, bic_ids].applymap(lambda x: d[x])

    if no_bic_columns:
        col_colors = col_colors.loc[:, cols]

    vmin, vmax = color_range

    g = sns.clustermap(
        exprs.loc[ordered_genes, sample_order],
        figsize=figsize,
        col_cluster=cluster_cols,
        row_cluster=cluster_rows,
        dendrogram_ratio=dendrogram_ratio,
        colors_ratio=colors_ratio,
        cmap=sns.color_palette("coolwarm", as_cmap=True),
        vmin=vmin,
        vmax=vmax,
        xticklabels=col_labels,
        yticklabels=row_labels,
        col_colors=col_colors,
        row_colors=row_colors,
    )
    ax = g.ax_heatmap
    ax.set_ylabel("")
    ax.set_xlabel(xlabel)

    g.ax_row_dendrogram.set_visible(False)
    # g.cax.set_position([.10, .2, .03, .45])
    # from https://stackoverflow.com/questions/47350879/seaborn-clustermap-subplots-adjust-cancels-colour-bar-relocation
    dendro_box = g.ax_row_dendrogram.get_position()
    dendro_box.x0 = (dendro_box.x0 + 2 * dendro_box.x1) / 3
    g.cax.set_position(dendro_box)
    # Move the ticks to the left (https://stackoverflow.com/a/36939552/1878788)
    g.cax.yaxis.set_ticks_position("left")

    if no_cbar:
        g.ax_cbar.set_visible(False)

    # If there are row labels to highlight, format them in bold
    if highlight_row_labels:
        for row_tick in g.ax_heatmap.get_yticklabels():
            if row_tick.get_text() in highlight_row_labels:
                row_tick.set_weight("bold")
                for bic_id in biclusters.index.values:
                    if row_tick.get_text() in biclusters.loc[bic_id, "genes"]:
                        if not row_labels_black:
                            row_tick.set_color(bic_colors[bic_id])

    legends = []
    i = 0
    n_patches = 0
    for col in cols:
        patches = []
        col_color_map = color_dict[col]
        # add patches only for groups found in annotation
        plot_groups = [x for x in col_color_map.keys() if x in set(annot[col].values)]
        for group in plot_groups:
            p = g.ax_row_dendrogram.bar(
                0, 0, color=col_color_map[group], label=group, linewidth=0
            )
            patches.append(p)
        if not no_legend:
            if len(set(annot[col].values)) <= 10:
                # add the legend
                legends.append(
                    plt.legend(
                        patches,
                        plot_groups,
                        loc="upper left",
                        title=col,
                        ncol=10,
                        bbox_to_anchor=(0.05 * i + 0.06 * (n_patches), 0.95),
                        bbox_transform=gcf().transFigure,
                    )
                )
                n_patches += len(patches)
                if i > 0:
                    plt.gca().add_artist(legends[i - 1])
                i += 1
            else:
                # add the legend to the side
                legends.append(
                    plt.legend(
                        patches,
                        plot_groups,
                        loc="upper right",
                        title=col,
                        ncol=2,  # bbox_to_anchor=(0.05*i+0.06*(n_patches), 0.95),
                        bbox_transform=gcf().transFigure,
                    )
                )
                if i > 0:
                    plt.gca().add_artist(legends[i - 1])
                i += 1
    return g, sample_order, (row_colors, col_colors)


def order_one(
    exprs, s0, subt_dict, subt_order=["Her2", "Basal", "LumA", "LumB", "Normal"]
):
    all_samples = set()
    for subt in subt_dict.keys():
        all_samples = all_samples.union(subt_dict[subt])

    s0 = set(s0)
    bg = all_samples.difference(s0)

    ordered_samples = []
    ordered_sets = [s0, bg]
    for s_i in ordered_sets:
        for subt in subt_order:
            ordered_subset = list(s_i.intersection(subt_dict[subt]))
            # ordered_subset = list(exprs.loc[:,ordered_subset].sum().sort_values().index)
            ordered_samples += ordered_subset
    return ordered_samples


def order_two(
    s0, s1, subt_dict, subt_order=["Her2", "Basal", "LumA", "LumB", "Normal"]
):
    all_samples = set()
    for subt in subt_dict.keys():
        all_samples = all_samples.union(subt_dict[subt])
    s0 = set(s0)
    s1 = set(s1)

    overlap = s0.intersection(s1)
    s0_ = s0.difference(s1)
    s1_ = s1.difference(s0)
    bg = all_samples.difference(s0).difference(s1)

    ordered_samples = []
    ordered_sets = [overlap, s0_, s1_, bg]
    for s_i in ordered_sets:
        for subt in subt_order:
            ordered_subset = list(s_i.intersection(subt_dict[subt]))
            ordered_samples += ordered_subset
    return ordered_samples
