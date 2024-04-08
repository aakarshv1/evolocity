import json

# likelihoods -> seq -> basic, bidirectional likelihoods

f = open("likelihoods.json")

likelihoods = json.load(f)

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

    return np.array(
        [
            likelihoods[seq].basic
            for seq in seqs
        ]
    )

def predict_sequence_prob_evo(
        seq, alphabet, model, repr_layers,
        batch_size=80000, verbose=False
):
    # pad 1 on left, 1 on right
    zero_pad = np.zeros(4)
    concatenated_result = np.concatenate((zero_pad, likelihoods[seq].bidirectional, zero_pad), axis=0)
    return concatenated_result