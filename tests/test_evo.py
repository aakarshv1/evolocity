import sys
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 
import evolocity as evo
import numpy as np
import scanpy as sc

test_seqs = [
    'ATCGTGCAGTACCAGGTAACGTACGATCAGATTCA',
    'GGTCAGCTAGCTAAATGCAGCTAGCTTAAGTTAGC',
    'CCGATCGAGTCTAGCTAGCGATCGATATGCATCGA',
    'ACTGACGTAGCCCAGCTAGCTAGCCAGGGACTGAC',
    'ACTGACCCAGTTCACGTACGATCGACGTACGCATG',
    'ACGGTACGTACGGTTAGCCCTAGCTAGCCAGTCGA',
    'TCCCGATCGATCGACTGACTAGCTAGCTAGCATCG',
    'ACTAGCTAGCATCGATCGACTTTTACGTAGCGACT',
]

def test_einsum():
    from evolocity.tools.utils import prod_sum_obs, prod_sum_var, norm

    Ms, Mu = np.random.rand(5, 4), np.random.rand(5, 4)
    assert np.allclose(prod_sum_obs(Ms, Mu), np.sum(Ms * Mu, 0))
    assert np.allclose(prod_sum_var(Ms, Mu), np.sum(Ms * Mu, 1))
    assert np.allclose(norm(Ms), np.linalg.norm(Ms, axis=1))

def test_pipeline():
    adata = evo.pp.featurize_seqs(test_seqs, model_name="evo")
    # return evo.pp.featurize_seqs(test_seqs, model_name="evo")
    evo.pp.neighbors(adata)
    sc.tl.umap(adata)

    evo.tl.velocity_graph(adata, model_name="evo", score="basic_evo")
    evo.tl.velocity_embedding(adata, basis='umap')

    evo.pl.velocity_embedding(adata, basis='umap', scale=1., save=True)

    evo.tl.onehot_msa(adata)
    evo.tl.residue_scores(adata)
    evo.pl.residue_scores(adata, save="res_scores")

    evo.tl.terminal_states(adata)
    evo.tl.velocity_pseudotime(adata)

    evo.tl.random_walk(adata)
    
    assert(adata.X.shape[0] == len(test_seqs))
    assert('seq' in adata.obs)
    assert('seqs_msa' in adata.obs)

def test_pipeline_tape():
    adata = evo.pp.featurize_seqs(test_seqs)
    evo.pp.neighbors(adata)
    sc.tl.umap(adata)

    evo.tl.velocity_graph(adata, model_name='tape')
    evo.tl.velocity_embedding(adata, basis='umap')

    evo.pl.velocity_embedding(adata, basis='umap', scale=1.)

    assert(adata.X.shape[0] == len(test_seqs))
    assert('seq' in adata.obs)
    assert(adata.uns['model'].name_ == 'tape')

def test_pipeline_dataset():
    adata = evo.datasets.cytochrome_c()
    adata = evo.datasets.nucleoprotein()

if __name__=="__main__":
    test_pipeline()
