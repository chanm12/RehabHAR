from torch import nn


class QuantizedClassifier(nn.Module):
    def __init__(self, args):
        super(QuantizedClassifier, self).__init__()
        # Randomly initialized embeddings
        self.embeddings = nn.Embedding(args.vocab_size, args.embedding_size)
        self.embeddings.weight.requires_grad = True

        # RNN model
        self.rnn_1 = nn.LSTM(
            input_size=args.embedding_size,
            hidden_size=128,
            num_layers=2,
            bidirectional=False,
            batch_first=True,
            dropout=0.2,
        )

        shape = 128
        self.classifier = nn.Sequential(
            nn.Linear(shape, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, args.num_classes),
        )

    def forward(self, inputs):
        embeddings = self.embeddings(inputs)

        r_out, _ = self.rnn_1(embeddings, None)

        softmax = self.classifier(r_out[:, -1, :])

        return softmax
