import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

log = logging.getLogger(__name__)

class AttentionPooling(nn.Module):
    """
    A reusable module implementing standard attention pooling.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # 1. Learnable "query" vector representing [CLS] token or global context
        self.query = nn.Parameter(torch.randn(1, d_model))
        # 2. Linear layer to project input sequence as "keys"
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        # We can reuse key_projection to project queries, or create a separate projection for queries
        # self.query_projection = nn.Linear(d_model, d_model, bias=False)

    def forward(self, memory: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            memory (torch.Tensor): Output from Transformer Encoder.
                                   Shape: [batch_size, seq_len, d_model]
            padding_mask (torch.Tensor): Mask for padding positions.
                                         Shape: [batch_size, seq_len]
        Returns:
            torch.Tensor: Pooled sentence embedding vector.
                          Shape: [batch_size, d_model]
        """
        # 1. Project input sequence as "keys"
        # keys shape: [batch_size, seq_len, d_model]
        keys = self.key_projection(memory)

        # 2. Calculate dot product similarity between queries and keys (attention scores)
        # self.query: [1, d_model] -> [batch_size, 1, d_model]
        # keys.transpose: [batch_size, d_model, seq_len]
        # attn_scores shape: [batch_size, 1, seq_len]

        attn_scores = torch.bmm(self.query.unsqueeze(0).repeat(memory.size(0), 1, 1),
                                keys.transpose(1, 2))

        # 3. Apply padding mask
        # padding_mask: [batch_size, seq_len] -> [batch_size, 1, seq_len]
        attn_scores.masked_fill_(padding_mask.unsqueeze(1), -1e9)

        # 4. Calculate softmax to get attention weights
        # attn_weights shape: [batch_size, 1, seq_len]
        attn_weights = F.softmax(attn_scores, dim=-1)


        pooled = torch.bmm(attn_weights, memory)

        # Remove extra dimensions
        return pooled.squeeze(1) # Shape: [batch_size, d_model]

class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        # Traditional FFN is d_model -> dim_feedforward -> d_model
        # SwiGLU typically recommends dim_feedforward about 2/3 * 4 = 8/3 times d_model
        # LLaMA uses a fixed multiplier, e.g., 2.66
        hidden_dim = int(2 * dim_feedforward / 3) # Recommendation from LLaMA paper

        # Make it divisible by 8 for hardware acceleration
        hidden_dim = (hidden_dim + 7) // 8 * 8

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.silu is Swish(x * sigmoid(x))
        gate = F.silu(self.w1(x))
        up = self.w3(x)
        fuse = gate * up
        return self.w2(self.dropout(fuse))

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        # Use custom SwiGLU
        self.ffn = SwiGLUFeedForward(d_model, dim_feedforward, dropout)

        # Other components remain consistent with native Layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor | None = None, src_key_padding_mask: torch.Tensor | None = None,**kwargs) -> torch.Tensor:
        # Pre-Norm structure (consistent with norm_first=True setting)
        # 1. Self-attention part
        x = src
        attn_output, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x),
                                         attn_mask=src_mask,
                                         key_padding_mask=src_key_padding_mask)
        x = x + self.dropout1(attn_output)

        # 2. Feedforward network part
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_output)

        return x

class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        # Use custom SwiGLU
        self.ffn = SwiGLUFeedForward(d_model, dim_feedforward, dropout)

        # Other components remain consistent with native Layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: torch.Tensor | None = None, memory_mask: torch.Tensor | None = None,
                tgt_key_padding_mask: torch.Tensor | None = None,
                memory_key_padding_mask: torch.Tensor | None = None,**kwargs) -> torch.Tensor:
        # Pre-Norm structure
        x = tgt
        # 1. Masked self-attention
        self_attn_output, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x),
                                              attn_mask=tgt_mask,
                                              key_padding_mask=tgt_key_padding_mask)
        x = x + self.dropout1(self_attn_output)

        # 2. Cross attention
        cross_attn_output, _ = self.multihead_attn(self.norm2(x), memory, memory,
                                                   attn_mask=memory_mask,
                                                   key_padding_mask=memory_key_padding_mask)
        x = x + self.dropout2(cross_attn_output)

        # 3. Feedforward network
        ffn_output = self.ffn(self.norm3(x))
        x = x + self.dropout3(ffn_output)

        return x

class PositionalEncoding(nn.Module):
    # --- No changes needed ---
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

def check_nan(tensor, name="Tensor"):
    # --- No changes needed ---
    if torch.isnan(tensor).any():
        log.error(f"NaN detected in {name}!")
        raise RuntimeError(f"NaN detected in {name}")

class BlockbAE_GRUEncoder_TransformerDecoder_Gumbel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, dim_feedforward=2048,
                 num_encoder_layers=3, num_decoder_layers=3, dropout=0.1,
                 latent_dim=300, initial_temp=1.0, pad_id=0,share_weights=True, encoder_type: str = 'transformer', **kwargs): # Absorb extra parameters
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.pad_id = pad_id
        self.encoder_type = encoder_type
        self.register_buffer('temperature', torch.tensor(initial_temp))

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        log.info(f"Initializing VAE with [{self.encoder_type.upper()}] Encoder.")
        if self.encoder_type == 'transformer':
            encoder_layer = CustomTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
            self.attention_pooling = AttentionPooling(d_model)
        elif self.encoder_type == 'gru':
            self.encoder = nn.GRU(
                input_size=d_model,
                hidden_size=d_model, # To match decoder, hidden_size is typically set to d_model
                num_layers=num_encoder_layers,
                batch_first=True,
                dropout=dropout if num_encoder_layers > 1 else 0, # Dropout only effective with multiple layers
                bidirectional=True # Bidirectional GRU better captures context
            )
            self.encoder_output_transform = nn.Linear(d_model * 2, d_model)

        decoder_layer = CustomTransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        # =============================

        self.to_latent = nn.Linear(d_model, latent_dim)
        self.from_latent = nn.Linear(latent_dim, d_model,bias=False)
        self.fc_out = nn.Linear(d_model, vocab_size, bias=False)
        if share_weights:
            log.info("Sharing weights between embedding and output projection layers.")
            self.fc_out.weight = self.embedding.weight

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                if 'gru' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for Decoder."""
        mask = torch.triu(torch.ones((sz, sz), device=device, dtype=torch.bool), diagonal=1)
        return mask

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # --- Embedding and positional encoding (common) ---
        src_padding_mask = (src == self.pad_id)
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)

        if self.encoder_type == 'transformer':
            memory = self.encoder(src=src_emb, src_key_padding_mask=src_padding_mask)
            pooled = self.attention_pooling(memory, src_padding_mask)
        elif self.encoder_type == 'gru':
            lengths = (~src_padding_mask).sum(dim=1).cpu()
            packed_input = nn.utils.rnn.pack_padded_sequence(
                src_emb, lengths, batch_first=True, enforce_sorted=False
            )
            _, hidden = self.encoder(packed_input)
            last_hidden_forward = hidden[-2,:,:]
            last_hidden_backward = hidden[-1,:,:]
            pooled_raw = torch.cat([last_hidden_forward, last_hidden_backward], dim=1)
            pooled = torch.tanh(self.encoder_output_transform(pooled_raw))

        # --- Gumbel-Softmax sampling ---

        # 1. Get clean, raw logits from encoder
        base_logits = self.to_latent(pooled)

        # 2. (Retained) Perform numerical stability operation
        base_logits = 10.0 * torch.tanh(base_logits / 10.0)
        base_logits = torch.nan_to_num(base_logits, nan=0.0)

        # 3. Perform Gumbel-Softmax sampling to generate z
        logits_binary_for_gumbel = torch.stack([torch.zeros_like(base_logits), base_logits], dim=-1)

        z_gumbel = F.gumbel_softmax(
            logits_binary_for_gumbel,
            tau=self.temperature,
            hard=True,
            dim=-1
        )
        z_for_decoder = z_gumbel[..., 1]

        # 4. Return core components
        return base_logits, z_for_decoder

    def decode(self, z: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        # --- Use z as decoder memory, preserving information bottleneck ---
        # memory_z shape: [batch_size, 1, d_model]
        memory_z = self.from_latent(z).unsqueeze(1)
        # Expand to target sequence length for cross attention
        # memory shape: [batch_size, trg_seq_len, d_model]
        memory = memory_z.repeat(1, trg.size(1), 1)

        trg_emb = self.embedding(trg) * math.sqrt(self.d_model)
        trg_emb = self.pos_encoder(trg_emb)

        # --- Create masks for decoder ---
        trg_padding_mask = (trg == self.pad_id)
        trg_causal_mask = self._generate_square_subsequent_mask(trg.size(1), trg.device)

        # --- Transformer Decoder forward pass ---
        output = self.transformer_decoder(
            tgt=trg_emb,
            memory=memory,
            tgt_mask=trg_causal_mask,
            tgt_key_padding_mask=trg_padding_mask
        )

        logits = self.fc_out(output)
        logits = 10.0 * torch.tanh(logits / 10.0)
        return logits

    def forward(self, src: torch.Tensor, trg: torch.Tensor):
        # --- Adjust forward signature ---
        q, z = self.encode(src)
        return self.decode(z, trg)

    @torch.no_grad()
    def get_binary_representation(self, src: torch.Tensor) -> torch.Tensor:
        self.eval()
        # --- Call new encode ---
        _, z = self.encode(src)
        return z

    @torch.no_grad()
    def greedy_decode(self, z: torch.Tensor, sos_id: int, eos_id: int, max_len: int) -> torch.Tensor:
        """
        Solution A: Only optimize memory reuse, code remains almost unchanged
        """
        self.eval()
        batch_size = z.size(0)
        device = z.device

        # Pre-compute base memory to avoid repeated from_latent calculations
        base_memory = self.from_latent(z).unsqueeze(1)  # [B, 1, d_model]

        decoder_input = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_len - 1):
            current_len = decoder_input.size(1)

            # 🚀 Optimization 1: Direct broadcast, no need for repeat
            memory = base_memory.expand(-1, current_len, -1)  # broadcast is more efficient than repeat

            # Other parts remain unchanged
            trg_emb = self.embedding(decoder_input) * math.sqrt(self.d_model)
            trg_emb = self.pos_encoder(trg_emb)
            causal_mask = self._generate_square_subsequent_mask(current_len, device)

            output = self.transformer_decoder(
                tgt=trg_emb,
                memory=memory,
                tgt_mask=causal_mask
            )

            logits = self.fc_out(output[:, -1, :])
            next_token = logits.argmax(-1)

            next_token = torch.where(finished, torch.full_like(next_token, eos_id), next_token)
            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(1)], dim=1)

            finished = finished | (next_token == eos_id)
            if finished.all():
                break

        return decoder_input

    @torch.no_grad()
    def get_deterministic_binary_representation(self, src: torch.Tensor) -> torch.Tensor:
         """
         Compute noise-free, deterministic binary latent representation.
         Directly harden based on sign of encoder output logits.
         """
         self.eval()

         src_padding_mask = (src == self.pad_id)
         src_emb = self.embedding(src) * math.sqrt(self.d_model)
         src_emb = self.pos_encoder(src_emb)

         # ✅ Critical fix: Use exactly the same logic as encode method
         if self.encoder_type == 'transformer':
             memory = self.encoder(src=src_emb, src_key_padding_mask=src_padding_mask)
             pooled = self.attention_pooling(memory, src_padding_mask)
         else: # gru
             lengths = (~src_padding_mask).sum(dim=1).cpu()
             packed_input = nn.utils.rnn.pack_padded_sequence(
                 src_emb, lengths, batch_first=True, enforce_sorted=False
             )
             _, hidden = self.encoder(packed_input)
             last_hidden_forward, last_hidden_backward = hidden[-2,:,:], hidden[-1,:,:]
             pooled_raw = torch.cat([last_hidden_forward, last_hidden_backward], dim=1)
             pooled = torch.tanh(self.encoder_output_transform(pooled_raw))

         base_logits = self.to_latent(pooled)
         binary_code = (base_logits > 0).int()
         return binary_code
    
    @torch.no_grad()
    def calculate_reconstruction_loss(self, z_batch: torch.Tensor, sos_id: int, eos_id: int, max_len: int) -> torch.Tensor:
        """
        Calculate reconstruction loss for a batch of latent vectors z.
        Loss is defined as negative log probability of sequence generation during autoregressive decoding (-log p(S|z)).
        Higher loss values indicate worse compatibility between z and decoder.

        Args:
            z_batch (torch.Tensor): Batch of latent vectors, shape [B, latent_dim].
            sos_id (int): Start-of-sequence token ID.
            eos_id (int): End-of-sequence token ID.
            max_len (int): Maximum decoding length.

        Returns:
            torch.Tensor: Reconstruction loss for each z, shape [B].
        """
        self.eval()
        batch_size = z_batch.size(0)
        device = z_batch.device

        # Initialize decoder input and memory
        decoder_input = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)
        memory_z = self.from_latent(z_batch).unsqueeze(1)

        # Accumulate log probabilities for each sample
        total_log_probs = torch.zeros(batch_size, device=device)

        # Mark which sequences have generated <eos>, stop accumulating probabilities
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            # Prepare decoder input
            memory = memory_z.repeat(1, decoder_input.size(1), 1)
            trg_emb = self.embedding(decoder_input) * math.sqrt(self.d_model)
            trg_emb = self.pos_encoder(trg_emb)
            causal_mask = self._generate_square_subsequent_mask(decoder_input.size(1), device)

            # Decoder forward pass
            output = self.transformer_decoder(
                tgt=trg_emb,
                memory=memory,
                tgt_mask=causal_mask
            )

            # Only use last timestep output to predict next token
            logits = self.fc_out(output[:, -1, :])

            # Convert logits to log probabilities
            log_probs = F.log_softmax(logits, dim=-1)

            # Greedily select next token
            next_token = logits.argmax(-1)

            # Gather log probability of selected token from distribution
            # gather(dim, index)
            gathered_log_p = log_probs.gather(1, next_token.unsqueeze(-1)).squeeze(-1)

            # Core logic: Only accumulate probabilities for unfinished sequences
            # `~is_finished` is a boolean mask, True converts to 1, False to 0
            total_log_probs += gathered_log_p * (~is_finished)

            # Add newly generated token to input sequence for next step
            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(1)], dim=1)

            # Update completion marker
            # `|=` is in-place OR operation, once a sequence is completed, it stays True forever
            is_finished |= (next_token == eos_id)

            # Exit loop early if all sequences in batch are complete
            if is_finished.all():
                break

        # Loss is negative log probability
        return -total_log_probs
    