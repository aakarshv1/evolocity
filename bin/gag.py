from mutation import *
from evolocity_graph import *

np.random.seed(1)
random.seed(1)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='HIV gag sequence analysis')
    parser.add_argument('model_name', type=str,
                        help='Type of language model (e.g., esm1b, tape)')
    parser.add_argument('--namespace', type=str, default='gag',
                        help='Model namespace')
    parser.add_argument('--dim', type=int, default=512,
                        help='Embedding dimension')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--evolocity', action='store_true',
                        help='Analyze evolocity')
    args = parser.parse_args()
    return args

def load_meta(meta_fnames):
    metas = {}
    for fname in meta_fnames:
        with open(fname) as f:
            for line in f:
                if not line.startswith('>'):
                    continue
                accession = line[1:].rstrip()
                fields = line.rstrip().split('.')
                subtype, country, year, strain = (
                    fields[0], fields[1], fields[2], fields[3]
                )

                if year == '-':
                    year = None
                else:
                    year = int(year)

                subtype = subtype.split('_')[-1]
                subtype = subtype.lstrip('>0123')

                keep_subtypes = {
                    'A', 'A1', 'A1A2', 'A1C', 'A1D', 'A2', 'A3', 'A6',
                    'AE', 'AG', 'B', 'C', 'BC', 'D',
                    'F', 'F1', 'F2', 'G', 'H', 'J',
                    'K', 'L', 'N', 'O', 'P', 'U',
                }
                if subtype not in keep_subtypes:
                    subtype = 'Other'

                metas[accession] = {
                    'gene_id': accession,
                    'subtype': subtype,
                    'country': country,
                    'year': year,
                    'strain': strain,
                    'accession': fields[-1],
                }
    return metas

def process(args, fnames, meta_fnames):
    metas = load_meta(meta_fnames)

    seqs = {}
    for fname in fnames:
        for record in SeqIO.parse(fname, 'fasta'):
            accession = record.description
            meta = metas[accession]
            meta['seqlen'] = len(str(record.seq))
            if meta['seqlen'] < 480 or meta['seqlen'] > 515:
                continue
            if 'X' in record.seq:
                continue
            if record.seq not in seqs:
                seqs[record.seq] = []
            seqs[record.seq].append(meta)

    tprint('Found {} unique sequences'.format(len(seqs)))

    seqs = training_distances(seqs, namespace=args.namespace)

    return seqs

def setup(args):
    fnames = [ 'data/gag/gag.fa' ]
    meta_fnames = [ 'data/gag/gag.fa' ]

    import pickle
    cache_fname = 'target/ev_cache/gag_seqs.pkl'
    try:
        with open(cache_fname, 'rb') as f:
            seqs = pickle.load(f)
    except:
        seqs = process(args, fnames, meta_fnames)
        with open(cache_fname, 'wb') as of:
            pickle.dump(seqs, of)

    seq_len = max([ len(seq) for seq in seqs ]) + 2
    vocab_size = len(AAs) + 2

    model = get_model(args, seq_len, vocab_size,
                      inference_batch_size=1000)

    return model, seqs

def interpret_clusters(adata):
    clusters = sorted(set(adata.obs['louvain']))
    for cluster in clusters:
        tprint('Cluster {}'.format(cluster))
        adata_cluster = adata[adata.obs['louvain'] == cluster]
        for var in [ 'year', 'country', 'subtype' ]:
            tprint('\t{}:'.format(var))
            counts = Counter(adata_cluster.obs[var])
            for val, count in counts.most_common():
                tprint('\t\t{}: {}'.format(val, count))
        tprint('')

    cluster2subtype = {}
    for i in range(len(adata)):
        cluster = adata.obs['louvain'][i]
        if cluster not in cluster2subtype:
            cluster2subtype[cluster] = []
        cluster2subtype[cluster].append(adata.obs['subtype'][i])
    largest_pct_subtype = []
    for cluster in cluster2subtype:
        count = Counter(cluster2subtype[cluster]).most_common(1)[0][1]
        pct_subtype = float(count) / len(cluster2subtype[cluster])
        largest_pct_subtype.append(pct_subtype)
        tprint('\tCluster {}, largest subtype % = {}'
               .format(cluster, pct_subtype))
    tprint('Purity, Louvain and subtype: {}'
           .format(np.mean(largest_pct_subtype)))

def plot_umap(adata):
    sc.pl.umap(adata, color='louvain', save='_gag_louvain.png')
    sc.pl.umap(adata, color='subtype', save='_gag_subtype.png')
    sc.pl.umap(adata, color='year', save='_gag_year.png')

def seqs_to_anndata(seqs):
    keys = set([ key for seq in seqs for meta in seqs[seq] for key in meta ])

    X, obs = [], {}
    obs['n_seq'] = []
    obs['seq'] = []
    for seq in seqs:
        X.append(seqs[seq][0]['embedding'])
        for key in keys:
            if key == 'embedding':
                continue
            if key not in obs:
                obs[key] = []
            values = [ meta[key] for meta in seqs[seq] if key in meta ]
            if len(values) > 0:
                obs[key].append(Counter(values).most_common(1)[0][0])
            else:
                obs[key].append(None)
        obs['n_seq'].append(len(seqs[seq]))
        obs['seq'].append(str(seq))
    X = np.array(X)

    adata = AnnData(X)
    for key in obs:
        adata.obs[key] = obs[key]

    return adata

def evo_gag(args, model, seqs, vocabulary, namespace='gag'):
    #############################
    ## Visualize Gag landscape ##
    #############################

    adata_cache = 'target/ev_cache/gag_adata.h5ad'
    try:
        import anndata
        adata = anndata.read_h5ad(adata_cache)
    except:
        seqs = populate_embedding(args, model, seqs, vocabulary, use_cache=True)
        adata = seqs_to_anndata(seqs)
        adata.write(adata_cache)

    keep_subtypes = {
        'AE', 'B', 'C', 'BC', 'D',
    }
    adata.obs['simple_subtype'] = [
        subtype if subtype in keep_subtypes else 'A'
        for subtype in adata.obs['subtype']
    ]

    sc.pp.neighbors(adata, n_neighbors=40, use_rep='X')

    sc.tl.louvain(adata, resolution=1.)

    sc.set_figure_params(dpi_save=500)
    sc.tl.umap(adata, min_dist=0.7)
    plot_umap(adata)

    cache_prefix = 'target/ev_cache/gag_knn40'
    try:
        from scipy.sparse import load_npz
        adata.uns["velocity_graph"] = load_npz(
            '{}_vgraph.npz'.format(cache_prefix)
        )
        adata.uns["velocity_graph_neg"] = load_npz(
            '{}_vgraph_neg.npz'.format(cache_prefix)
        )
        adata.obs["velocity_self_transition"] = np.load(
            '{}_vself_transition.npy'.format(cache_prefix)
        )
        adata.layers["velocity"] = np.zeros(adata.X.shape)
    except:
        velocity_graph(adata, args, vocabulary, model)
        from scipy.sparse import save_npz
        save_npz('{}_vgraph.npz'.format(cache_prefix),
                 adata.uns["velocity_graph"],)
        save_npz('{}_vgraph_neg.npz'.format(cache_prefix),
                 adata.uns["velocity_graph_neg"],)
        np.save('{}_vself_transition.npy'.format(cache_prefix),
                adata.obs["velocity_self_transition"],)

    import scvelo as scv
    scv.tl.velocity_embedding(adata, basis='umap', scale=1.,
                              self_transitions=True,
                              use_negative_cosines=True,
                              retain_scale=False,
                              autoscale=True,)
    scv.pl.velocity_embedding(
        adata, basis='umap', color='year',
        save=f'_{namespace}_year_velo.png',
    )

    # Grid visualization.
    plt.figure()
    ax = scv.pl.velocity_embedding_grid(
        adata, basis='umap', min_mass=1., smooth=1.,
        arrow_size=1., arrow_length=3.,
        color='year', show=False,
    )
    plt.tight_layout(pad=1.1)
    plt.subplots_adjust(right=0.85)
    plt.savefig(f'figures/scvelo__{namespace}_year_velogrid.png', dpi=500)
    plt.close()

    # Streamplot visualization.
    plt.figure()
    ax = scv.pl.velocity_embedding_stream(
        adata, basis='umap', min_mass=3., smooth=1., linewidth=0.7,
        color='year', show=False,
    )
    sc.pl._utils.plot_edges(ax, adata, 'umap', 0.1, '#aaaaaa')
    plt.tight_layout(pad=1.1)
    plt.subplots_adjust(right=0.85)
    plt.savefig(f'figures/scvelo__{namespace}_year_velostream.png', dpi=500)
    plt.close()
    plt.figure()
    ax = scv.pl.velocity_embedding_stream(
        adata, basis='umap', min_mass=3.7, smooth=1., linewidth=0.7,
        color='simple_subtype', show=False,
    )
    sc.pl._utils.plot_edges(ax, adata, 'umap', 0.1, '#dddddd')
    plt.tight_layout(pad=1.1)
    plt.subplots_adjust(right=0.85)
    plt.savefig(f'figures/scvelo__{namespace}_subtype_velostream.png', dpi=500)
    plt.close()

    plot_pseudotime(
        adata, basis='umap', min_mass=1., smooth=0.5, levels=100,
        rank_transform=True,
        arrow_size=1., arrow_length=3., cmap='coolwarm',
        c='#aaaaaa', show=False,
        save=f'_{namespace}_pseudotime.png', dpi=500
    )

    scv.pl.scatter(adata, color=[ 'root_cells', 'end_points' ],
                   cmap=plt.cm.get_cmap('magma').reversed(),
                   save=f'_{namespace}_origins.png', dpi=500)
    scv.pl.scatter(adata, color='pseudotime',
                   cmap=plt.cm.get_cmap('magma').reversed(),
                   save=f'_{namespace}_pseudotime.png', dpi=500)

    nnan_idx = (np.isfinite(adata.obs['year']) &
                np.isfinite(adata.obs['pseudotime']))
    tprint('Pseudotime-time Spearman r = {}, P = {}'
           .format(*ss.spearmanr(adata.obs['pseudotime'][nnan_idx],
                                 adata.obs['year'][nnan_idx],
                                 nan_policy='omit')))
    tprint('Pseudotime-time Pearson r = {}, P = {}'
           .format(*ss.pearsonr(adata.obs['pseudotime'][nnan_idx],
                                adata.obs['year'][nnan_idx])))


if __name__ == '__main__':
    args = parse_args()

    AAs = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
        'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
        'Y', 'V', 'X', 'Z', 'J', 'U', 'B',
    ]
    vocabulary = { aa: idx + 1 for idx, aa in enumerate(sorted(AAs)) }

    model, seqs = setup(args)

    if 'esm' in args.model_name:
        vocabulary = { tok: model.alphabet_.tok_to_idx[tok]
                       for tok in model.alphabet_.tok_to_idx
                       if '<' not in tok and tok != '.' and tok != '-' }
        args.checkpoint = args.model_name
    elif args.model_name == 'tape':
        vocabulary = { tok: model.alphabet_[tok]
                       for tok in model.alphabet_ if '<' not in tok }
        args.checkpoint = args.model_name
    elif args.checkpoint is not None:
        model.model_.load_weights(args.checkpoint)
        tprint('Model summary:')
        tprint(model.model_.summary())

    if args.evolocity:
        if args.checkpoint is None and not args.train:
            raise ValueError('Model must be trained or loaded '
                             'from checkpoint.')
        no_embed = { 'hmm' }
        if args.model_name in no_embed:
            raise ValueError('Embeddings not available for models: {}'
                             .format(', '.join(no_embed)))

        evo_gag(args, model, seqs, vocabulary)