"""
Policy Networks
================

Three architectures that all share the same interface:

    forward(action_history) → logits over arms

Each processes a sequence of past arm indices and outputs a distribution
over the next arm to pull. The architectures differ in how they encode
temporal dependencies in the action history.

Architecture summary:
    GRUPolicy         — Gated Recurrent Unit. Hidden state naturally tracks
                        sequence progress. ~3K params. The default.
    LSTMPolicy        — Long Short-Term Memory. Separate cell state may help
                        with longer secret sequences. ~4K params.
    TransformerPolicy — Self-attention over action history. The Transformer
                        Bandit. ~6K params. Mostly for fun and ablation.

All networks:
    Input:  action_history  — (batch, seq_len) long tensor of 0-indexed arm indices
    Output: logits          — (batch, n_arms) unnormalized log-probabilities
"""

import torch
import torch.nn as nn
import math


class GRUPolicy(nn.Module):
    """GRU-based policy network.

    Embeds each past action into a d-dimensional vector, processes the
    sequence with a GRU, and maps the final hidden state to arm logits.

    The hidden state h_t implicitly learns to encode 'how far along the
    secret sequence am I' — which is the latent state of this POMDP.
    """

    def __init__(self, n_arms: int = 10, embed_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.n_arms = n_arms
        self.embed = nn.Embedding(n_arms, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, n_arms)

    def forward(self, action_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_history: (batch, seq_len) long tensor of past arm indices.

        Returns:
            logits: (batch, n_arms) unnormalized scores for each arm.
        """
        x = self.embed(action_history)          # (batch, seq_len, embed_dim)
        _, h = self.gru(x)                      # h: (1, batch, hidden_dim)
        return self.head(h.squeeze(0))          # (batch, n_arms)


class LSTMPolicy(nn.Module):
    """LSTM-based policy network.

    Same interface as GRUPolicy. The separate cell state c_t provides an
    additional memory pathway, which may help when the secret sequence is
    long and the relevant signal is far back in the history.
    """

    def __init__(self, n_arms: int = 10, embed_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.n_arms = n_arms
        self.embed = nn.Embedding(n_arms, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, n_arms)

    def forward(self, action_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_history: (batch, seq_len) long tensor of past arm indices.

        Returns:
            logits: (batch, n_arms) unnormalized scores for each arm.
        """
        x = self.embed(action_history)          # (batch, seq_len, embed_dim)
        _, (h, _) = self.lstm(x)                # h: (1, batch, hidden_dim)
        return self.head(h.squeeze(0))          # (batch, n_arms)


class TransformerPolicy(nn.Module):
    """Transformer-based policy network. The Transformer Bandit.

    Self-attention over the action history lets the model attend to any
    past timestep directly — no recurrence bottleneck. A learned [CLS]-style
    readout token aggregates information for the policy head.

    Positional encoding is sinusoidal so the model understands ordering.
    Causal masking ensures each position only attends to past actions.
    """

    def __init__(
        self,
        n_arms: int = 10,
        embed_dim: int = 16,
        n_heads: int = 2,
        n_layers: int = 1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.n_arms = n_arms
        self.embed_dim = embed_dim

        self.embed = nn.Embedding(n_arms, embed_dim)
        self.pos_encoding = self._sinusoidal_encoding(max_seq_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(embed_dim, n_arms)

    def forward(self, action_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_history: (batch, seq_len) long tensor of past arm indices.

        Returns:
            logits: (batch, n_arms) unnormalized scores for each arm.
        """
        batch, seq_len = action_history.shape

        x = self.embed(action_history)          # (batch, seq_len, embed_dim)

        # Add positional encoding — extend on the fly if sequence exceeds max_seq_len
        if seq_len > self.pos_encoding.size(0):
            self.pos_encoding = self._sinusoidal_encoding(seq_len, self.embed_dim)
        positions = self.pos_encoding[:seq_len].to(x.device)
        x = x + positions.unsqueeze(0)          # broadcast over batch

        # Causal mask: each position attends only to itself and earlier
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        x = self.transformer(x, mask=causal_mask)  # (batch, seq_len, embed_dim)

        # Use the last position as the summary (like taking h_T from an RNN)
        last_hidden = x[:, -1, :]               # (batch, embed_dim)
        return self.head(last_hidden)            # (batch, n_arms)

    @staticmethod
    def _sinusoidal_encoding(max_len: int, d_model: int) -> torch.Tensor:
        """Standard sinusoidal positional encoding.

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
