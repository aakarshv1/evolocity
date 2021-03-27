from mutation import *
from evolocity_graph import *

np.random.seed(1)
random.seed(1)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='HIV sequence analysis')
    parser.add_argument('model_name', type=str,
                        help='Type of language model (e.g., hmm, lstm)')
    parser.add_argument('--namespace', type=str, default='hiv',
                        help='Model namespace')
    parser.add_argument('--dim', type=int, default=512,
                        help='Embedding dimension')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Training minibatch size')
    parser.add_argument('--n-epochs', type=int, default=4,
                        help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint')
    parser.add_argument('--train', action='store_true',
                        help='Train model')
    parser.add_argument('--train-split', action='store_true',
                        help='Train model on portion of data')
    parser.add_argument('--test', action='store_true',
                        help='Test model')
    parser.add_argument('--embed', action='store_true',
                        help='Analyze embeddings')
    parser.add_argument('--semantics', action='store_true',
                        help='Analyze mutational semantic change')
    parser.add_argument('--combfit', action='store_true',
                        help='Analyze combinatorial fitness')
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
            if args.namespace == 'hiva' and \
               (not meta['subtype'].startswith('A')):
                continue
            if 'X' in record.seq:
                continue
            if record.seq not in seqs:
                seqs[record.seq] = []
            seqs[record.seq].append(meta)
    return seqs

def split_seqs(seqs, split_method='random'):
    train_seqs, test_seqs = {}, {}

    old_cutoff = 1900
    new_cutoff = 2008

    tprint('Splitting seqs...')
    for seq in seqs:
        # Pick validation set based on date.
        seq_dates = [
            meta['year'] for meta in seqs[seq]
            if meta['year'] is not None
        ]
        if len(seq_dates) == 0:
            test_seqs[seq] = seqs[seq]
            continue
        if len(seq_dates) > 0:
            oldest_date = sorted(seq_dates)[0]
            if oldest_date < old_cutoff or oldest_date >= new_cutoff:
                test_seqs[seq] = seqs[seq]
                continue
        train_seqs[seq] = seqs[seq]
    tprint('{} train seqs, {} test seqs.'
           .format(len(train_seqs), len(test_seqs)))

    return train_seqs, test_seqs

def setup(args):
    fnames = [ 'data/hiv/HIV-1_env_samelen.fa' ]
    meta_fnames = [ 'data/hiv/HIV-1_env_samelen.fa' ]

    seqs = process(args, fnames, meta_fnames)

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
    sc.tl.umap(adata, min_dist=1.)
    sc.pl.umap(adata, color='louvain', save='_hiv_louvain.png')
    sc.pl.umap(adata, color='subtype', save='_hiv_subtype.png')

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

def analyze_embedding(args, model, seqs, vocabulary):
    seqs = populate_embedding(args, model, seqs, vocabulary,
                              use_cache=True)

    adata = seqs_to_anndata(seqs)

    sc.pp.neighbors(adata, n_neighbors=200, use_rep='X')
    sc.tl.louvain(adata, resolution=1.)

    sc.set_figure_params(dpi_save=500)
    plot_umap(adata)

    interpret_clusters(adata)

def load_time_data():
    accession_to_data = {}
    with open('data/hiv/HIV-1_B_status_meta.txt') as f:
        header = [ field.replace(' ', '_').lower()
                   for field in f.readline().rstrip().split('\t') ]
        for line in f:
            fields = [ field.replace(' ', '_') if field else None
                       for field in line.rstrip().split('\t') ]
            accession = fields[1]
            data = { h_field: l_field for h_field, l_field in zip(header, fields) }
            if data['days_from_seroconversion'] == 'late' and \
               not data['fiebig_stage']:
                data['fiebig_stage'] = 'chronic'
            accession_to_data[accession] = data

    return accession_to_data, header

def tl_densities(adata, n_keep=5):
    dists = adata.obsp['distances']
    assert(dists.shape[0] == adata.X.shape[0])
    densities = []
    for i in range(dists.shape[0]):
        dists_i = dists[i]
        dists_i = sorted(np.array(dists_i[dists_i.nonzero()]).ravel())
        try:
            densities.append(np.mean(dists_i[:n_keep]))
        except:
            densities.append(float('nan'))
    adata.obs['density'] = densities

def plot_umap_keele2008(adata):
    sc.pl.umap(adata, color='corpus', save='_hiv_corpus.png')
    sc.pl.umap(adata, color='subtype', save='_hiv_subtype.png')
    sc.pl.umap(adata, color='status', save='_hiv_status.png')
    sc.pl.umap(adata, color='patient_code', save='_hiv_patient.png')
    sc.pl.umap(adata, color='louvain', save='_hiv_louvain.png')
    sc.pl.umap(adata, color='density', save='_hiv_density.png',
               vmax=0.7)

def evo_keele2008(args, model, seqs, vocabulary):
    #############################
    ## Visualize Env landscape ##
    #############################

    accession_to_data, new_fields = load_time_data()

    seqs = populate_embedding(
        args, model, seqs, vocabulary, use_cache=True, namespace='hiv'
    )
    for seq in seqs:
        for meta in seqs[seq]:
            meta['corpus'] = 'lanl'
            if meta['accession'] in accession_to_data:
                data = accession_to_data[meta['accession']]
                #meta.update(data)
                meta['status'] = data['fiebig_stage']
            else:
                #for key in new_fields:
                #    meta[key] = None
                meta['status'] = None

    from transfound import load_keele2008
    seqs_keele = load_keele2008()
    seqs_keele = populate_embedding(
        args, model, seqs_keele, vocabulary,
        use_cache=True, namespace='env_tf_keele2008'
    )
    for seq in seqs_keele:
        if seq in seqs:
            for meta in seqs[seq]:
                for key in seqs_keele[seq][0]:
                    meta[key] = seqs_keele[seq][0][key]
            seqs[seq] += seqs_keele[seq]
        else:
            seqs[seq] = seqs_keele[seq]

    adata = seqs_to_anndata(seqs)

    adata = adata[adata.obs.subtype == 'B']

    sc.pp.neighbors(adata, n_neighbors=10, use_rep='X')

    sc.tl.louvain(adata, resolution=1.)
    tl_densities(adata, n_keep=10)

    sc.set_figure_params(dpi_save=500)
    sc.tl.umap(adata, min_dist=1.)
    plot_umap_keele2008(adata)

    cache_prefix = 'target/ev_cache/env_knn10'
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
        sc.pp.neighbors(adata, n_neighbors=30, use_rep='X')
        velocity_graph(adata, args, vocabulary, model,
                       n_recurse_neighbors=0,)
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
        save='_env_year_velo.png',
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
    plt.savefig('figures/scvelo__env_year_velogrid.png', dpi=500)
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
    plt.savefig('figures/scvelo__env_year_velostream.png', dpi=500)
    plt.close()

    plot_pseudotime(
        adata, basis='umap', min_mass=1., smooth=0.5, levels=100,
        rank_transform=True,
        arrow_size=1., arrow_length=3., cmap='coolwarm',
        c='#aaaaaa', show=False,
        save='_env_pseudotime.png', dpi=500
    )

    adata.obs['root_cells'][adata.obs['root_cells'] < 1.] = 0
    adata.obs['end_points'][adata.obs['end_points'] < 1.] = 0

    scv.pl.scatter(adata, color=[ 'root_cells', 'end_points' ],
                   cmap=plt.cm.get_cmap('magma').reversed(),
                   save='_env_origins.png', dpi=500)
    scv.pl.scatter(adata, color='pseudotime',
                   cmap=plt.cm.get_cmap('magma').reversed(),
                   save='_env_pftime.png', dpi=500)

    nnan_idx = (np.isfinite(adata.obs['year']) &
                np.isfinite(adata.obs['pseudotime']))
    tprint('Pseudotime-time Spearman r = {}, P = {}'
           .format(*ss.spearmanr(adata.obs['pseudotime'][nnan_idx],
                                 adata.obs['year'][nnan_idx],
                                 nan_policy='omit')))
    tprint('Pseudotime-time Pearson r = {}, P = {}'
           .format(*ss.pearsonr(adata.obs['pseudotime'][nnan_idx],
                                adata.obs['year'][nnan_idx])))

    nnan_idx = (np.isfinite(adata.obs['density']) &
                np.isfinite(adata.obs['pseudotime']))
    tprint('Pseudotime-density Spearman r = {}, P = {}'
           .format(*ss.spearmanr(adata.obs['pseudotime'][nnan_idx],
                                 adata.obs['density'][nnan_idx],
                                 nan_policy='omit')))
    tprint('Pseudotime-density Pearson r = {}, P = {}'
           .format(*ss.pearsonr(adata.obs['pseudotime'][nnan_idx],
                                adata.obs['density'][nnan_idx])))

    pfs, statuses = [], []
    for pf, status in zip(adata.obs['pseudotime'], adata.obs['status']):
        if np.isfinite(pf) and status != 'None':
            if status == 'chronic':
                status = 7
            pfs.append(pf)
            statuses.append(int(status))
    tprint('Pseudotime-status Spearman r = {}, P = {}'
           .format(*ss.spearmanr(pfs, statuses)))
    tprint('Pseudotime-status Pearson r = {}, P = {}'
           .format(*ss.pearsonr(pfs, statuses)))
    tprint('TF/chronic t-test {}, P = {}'.
           format(*ss.ttest_ind(
               adata[adata.obs['status'] == '1'].obs['pseudotime'],
               adata[adata.obs['status'] == 'chronic'].obs['pseudotime'],
           )))

    adata_keele = adata[(adata.obs['status'] != 'None')]
    plt.figure()
    sns.violinplot(x='status', y='pseudotime', data=adata_keele.obs)
    plt.ylabel('pseudotime')
    plt.savefig('figures/scvelo__env_fitness_status.png', dpi=500)
    plt.close()
    plt.figure()
    sns.violinplot(x='status', y='density', data=adata_keele.obs)
    plt.savefig('figures/scvelo__env_density_status.png', dpi=500)
    plt.close()


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

    if args.train:
        batch_train(args, model, seqs, vocabulary, batch_size=5000)

    if args.train_split or args.test:
        train_test(args, model, seqs, vocabulary, split_seqs)

    if args.embed:
        if args.checkpoint is None and not args.train:
            raise ValueError('Model must be trained or loaded '
                             'from checkpoint.')
        no_embed = { 'hmm' }
        if args.model_name in no_embed:
            raise ValueError('Embeddings not available for models: {}'
                             .format(', '.join(no_embed)))
        analyze_embedding(args, model, seqs, vocabulary)

    if args.semantics:
        if args.checkpoint is None and not args.train:
            raise ValueError('Model must be trained or loaded '
                             'from checkpoint.')

        from escape import load_dingens2019
        tprint('Dingens et al. 2019...')
        seq_to_mutate, escape_seqs = load_dingens2019()
        positions = [ escape_seqs[seq][0]['pos'] for seq in escape_seqs ]
        min_pos, max_pos = min(positions), max(positions)
        analyze_semantics(
            args, model, vocabulary, seq_to_mutate, escape_seqs,
            min_pos=min_pos, max_pos=max_pos,
            beta=1., plot_acquisition=True,
        )

    if args.combfit:
        from combinatorial_fitness import load_haddox2018
        tprint('Haddox et al. 2018...')
        wt_seqs, seqs_fitness = load_haddox2018()
        strains = sorted(wt_seqs.keys())
        for strain in strains:
            analyze_comb_fitness(args, model, vocabulary,
                                 strain, wt_seqs[strain], seqs_fitness,
                                 beta=1.)

    if args.evolocity:
        if args.checkpoint is None and not args.train:
            raise ValueError('Model must be trained or loaded '
                             'from checkpoint.')
        no_embed = { 'hmm' }
        if args.model_name in no_embed:
            raise ValueError('Embeddings not available for models: {}'
                             .format(', '.join(no_embed)))

        evo_keele2008(args, model, seqs, vocabulary)
