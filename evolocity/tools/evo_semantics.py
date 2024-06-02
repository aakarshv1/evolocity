import torch
import numpy as np
from typing import List
from your_model_import import StripedHyena, CharLevelTokenizer  # replace with your actual import path

# loading likelihood scores from pickle file
with open('../../../likelihoods/test_likelihoods.pkl', 'rb') as file:
    data = pickle.load(file)
basic_scores_list = []
bidirectional_scores_list = []
for seq, scores in data.items():
    basic_scores_list.append(scores['basic'])
    bidirectional_scores_list.append(scores['bidirectional'])
basic_array = np.array(basic_scores_list)
bidirectional_array = np.array(bidirectional_scores_list)

# load your sequences from a text file
with open("../tests/test_sequences.txt", "r") as file:
    sequences = [sequence.rstrip() for sequence in file]

# create a dictionary of sequences and their indices
seq_to_index = {seq: index for index, seq in enumerate(sequences)}

def score_sequences(
    seqs: List[str],
    model: StripedHyena,
    tokenizer: CharLevelTokenizer,
    reduce_method: str = 'mean',
    device: str = 'cuda:0',
) -> List[float]:
    """
    Computes the model log-likelihood scores for sequences in `seqs`.
    Uses `reduce_method` to take the mean or sum across the likelihoods at each 
    position (default: `'mean'`).

    Returns a list of scalar scores corresponding to the reduced log-likelihoods for
    each sequence.    
    """

    return [np.mean(basic_likelihoods[seq_to_index[seq]]) for seq in seqs]

def predict_sequence_prob_evo(
        seq, alphabet, model, repr_layers,
        batch_size=80000, verbose=False
):
    # pad 1 on left, 1 on right
    zero_pad = np.zeros(4)
    concatenated_result = np.concatenate((zero_pad, bidirectional_likelihoods[seq_to_index[seq]], zero_pad), axis=0)
    return concatenated_result
