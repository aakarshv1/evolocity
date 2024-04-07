from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained
import numpy as np
import torch

def predict_sequence_prob_fb(
        seq, alphabet, model, repr_layers,
        batch_size=80000, verbose=False
):
    output = []

    batch_size = 1022
    for batchi in range(((len(seq) - 1) // batch_size) + 1):
        start = batchi * batch_size
        end = (batchi + 1) * batch_size

        seqs = [ seq[start:end] ]
        labels = [ 'seq0' ]

        dataset = FastaBatchedDataset(labels, seqs)
        batches = dataset.get_batch_indices(batch_size, extra_toks_per_seq=1)
        data_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=alphabet.get_batch_converter(),
            batch_sampler=batches
        )

        with torch.no_grad():
            for batch_idx, (labels, strs, toks) in enumerate(data_loader):
                if torch.cuda.is_available():
                    toks = toks.to(device="cuda", non_blocking=True)
                out = model(toks, repr_layers=repr_layers, return_contacts=False)
                lsoftmax = torch.nn.LogSoftmax(dim=1)
                logits = lsoftmax(out["logits"]).to(device="cpu").numpy()

        output.append(logits[0])

    for i, x in enumerate(output):
        if len(output) > 1 and i > 0:
            output[i] = output[i][1:]
        if len(output) > 1 and i < len(output) - 1:
            output[i] = output[i][:-1]

    out = np.vstack(output)
    print("Sequence Length:", len(seq))
    print("Shape of sequence prob output:", out.shape)
    return out

def embed_seqs_fb(
        model, seqs, repr_layers, alphabet,
        batch_size=4096, use_cache=False, verbose=True
):
    labels_full = [ 'seq' + str(i) for i in range(len(seqs)) ]

    embedded_seqs = {}

    max_len = max([ len(seq) for seq in seqs ])
    window_size = 1022
    n_windows = ((max_len - 1) // window_size) + 1
    for window_i in range(n_windows):
        start = window_i * window_size
        end = (window_i + 1) * window_size

        seqs_window = [ seq[start:end] for seq in seqs ]

        dataset = FastaBatchedDataset(labels_full, seqs_window)
        batches = dataset.get_batch_indices(batch_size, extra_toks_per_seq=1)
        data_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=alphabet.get_batch_converter(),
            batch_sampler=batches
        )

        with torch.no_grad():
            for batch_idx, (labels, strs, toks) in enumerate(data_loader):
                if torch.cuda.is_available():
                    toks = toks.to(device="cuda", non_blocking=True)
                out = model(toks, repr_layers=repr_layers, return_contacts=False)
                representations = {
                    layer: t.to(device="cpu")
                    for layer, t in out["representations"].items()
                }

                for i, label in enumerate(labels):
                    seq_idx = int(label[3:])
                    seq = seqs[seq_idx]
                    assert(len(representations.items()) == 1)
                    for _, t in representations.items():
                        representation = t[i, 1 : len(strs[i]) + 1]
                    if seq not in embedded_seqs:
                        embedded_seqs[seq] = []
                    embedded_seqs[seq].append({
                        f'embedding{window_i}': representation.numpy()
                    })

    for seq in embedded_seqs:
        embeddings = []
        for window_i in range(n_windows):
            for meta in embedded_seqs[seq]:
                if f'embedding{window_i}' in meta:
                    embedding_i = meta[f'embedding{window_i}']
                    if n_windows > 1 and window_i > 0:
                        embedding_i = embedding_i[1:]
                    if n_windows > 1 and window_i < n_windows - 1:
                        embedding_i = embedding_i[:-1]
                    embeddings.append(embedding_i)
                    break
        embedding = np.vstack(embeddings)
        embedded_seqs[seq] = [ { 'embedding': embedding } ]

    return embedded_seqs

def embed_seqs_evo(
        model, seqs, from_file=True, device='cuda:0',
        batch_size=512
):  
    if from_file:
        embeddings = torch.load(model.filename_)
    # else:
    #     input_ids, seq_lengths = prepare_batch(seqs, tokenizer, device=device)

    #     model, tokenizer = evo_model.model, evo_model.tokenizer
    #     model.to(device)
    #     model.eval()

    #     # monkey patch the unembed function with identity
    #     # this removes the final projection back from the embedding space into tokens
    #     # so the "logits" of the model is now the final layer embedding
    #     # see source for unembed - https://huggingface.co/togethercomputer/evo-1-131k-base/blob/main/model.py#L339
    #     class CustomEmbedding(nn.Module):
    #         def unembed(self, u):
    #             return u

    #     model.unembed = CustomEmbedding()
    #     embedding, _ = model(input_ids)
    embedded_seqs = {}
    for i in range(len(seqs)):
        embedded_seqs[seqs[i]] = [{'embedding': embeddings[i]}]
    return embedded_seqs
