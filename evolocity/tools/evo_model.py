class DummyEvo():
    def __init__(self, filename="embeddings/test_embeddings.pt"):
        self.name_ = 'evo'
        self.filename_ = filename
        self.vocabulary_ = {
                '<eod>': 0,
                '<eos>': 0,
                '<pad>': 1,
                'A': 65,
                'C': 67,
                'G': 71,
                'T': 84
            }