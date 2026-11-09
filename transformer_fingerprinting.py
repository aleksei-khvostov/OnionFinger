import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence of length seq_len."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x


class TransformerFingerprintingModel(nn.Module):
    """
    Transformer-based website fingerprinting model.

    Input:  (batch_size, 1, seq_len)  — packet directions in {-1, 0, +1}
    Output: (batch_size, num_classes) — raw logits

    Designed to be lightweight for CPU/MPS training.
    Recommended defaults for seq_len=5000:
        d_model=128, nhead=4, num_layers=3  (full attention is still O(L^2))
    For faster experiments, downsample seq_len to 1000-2000.
    """

    def __init__(
        self,
        num_classes: int,
        seq_len: int = 5000,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model

        # Map direction token {-1,0,1} to indices {0,1,2}
        self.vocab_size = 3
        self.token_shift = 1  # -1 -> 0, 0 -> 1, 1 -> 2

        self.embedding = nn.Embedding(self.vocab_size, d_model)

        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.classifier.bias.data.zero_()
        self.classifier.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, seq_len) -> (batch, seq_len)
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        elif x.dim() != 2:
            raise ValueError(f"Expected input of shape (B, 1, L) or (B, L), got {x.shape}")

        # Convert {-1,0,1} to token indices {0,1,2}
        tokens = (x + self.token_shift).long().clamp(0, self.vocab_size - 1)

        # Embedding + positional encoding
        x = self.embedding(tokens) * math.sqrt(self.d_model)   # (B, L, d_model)
        x = self.pos_encoder(x)
        x = self.dropout(x)

        # Transformer encoder
        x = self.transformer_encoder(x)                        # (B, L, d_model)

        # Pooling (mean over sequence)
        x = x.mean(dim=1)                                      # (B, d_model)

        x = self.dropout(x)
        logits = self.classifier(x)                            # (B, num_classes)
        return logits
