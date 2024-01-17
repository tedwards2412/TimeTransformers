import torch
import torch.nn as nn
import math
import torch.utils.checkpoint


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear layers for query, key, and value
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def custom_softmax(self, x, dim=-1):
        max_val = torch.max(x, dim=dim, keepdim=True).values
        max_val = torch.where(
            torch.isinf(max_val), torch.tensor(0.0, device=max_val.device), max_val
        )
        exp_x = torch.exp(x - max_val)
        sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
        softmax_x = exp_x / (sum_exp_x + 1e-9)
        return softmax_x

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            # attn_scores_temp = attn_scores.masked_fill(mask == 0, -1e10)
            attn_scores = attn_scores.masked_fill(mask == 0, -torch.inf)

        # Softmax is applied to obtain attention probabilities
        # attn_probs = torch.softmax(attn_scores_temp, dim=-1)
        attn_probs = self.custom_softmax(attn_scores, dim=-1)

        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        # Adjust to handle d_model = 1
        if self.d_model == 1:
            return x.unsqueeze(1)
        else:
            batch_size, seq_length, _ = x.size()
            return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(
                1, 2
            )

    def combine_heads(self, x):
        # Adjust to handle d_model = 1
        if self.d_model == 1:
            return x.squeeze(1)
        else:
            batch_size, _, seq_length, _ = x.size()
            return (
                x.transpose(1, 2)
                .contiguous()
                .view(batch_size, seq_length, self.d_model)
            )

    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        # attn_output2 = nn.functional.scaled_dot_product_attention(
        #     Q, K, V, attn_mask=mask
        # )

        # Combine heads and apply output transformation

        output = self.W_o(self.combine_heads(attn_output))

        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc2(self.relu(self.fc1(x))))


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        # Create a learnable positional embedding
        self.pe = nn.Parameter(torch.zeros(1, max_seq_length, d_model))

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.pe[:, : x.size(1)]
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        # self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        # Checkpoint the forward pass
        if self.training:  # Only apply checkpointing during training
            return torch.utils.checkpoint.checkpoint(
                self.forward_step, x, src_mask, use_reentrant=False
            )
        else:
            return self.forward_step(x, src_mask)

    def forward_step(self, x, src_mask):
        # def forward(self, x, src_mask):
        attn_output = self.self_attn(x, x, x, src_mask)
        # x = self.norm1(x + self.dropout(attn_output))
        x = x + self.dropout(attn_output)
        ff_output = self.feed_forward(x)
        # x = self.norm2(x + self.dropout(ff_output))
        x = x + self.dropout(ff_output)
        return x


class Decoder_Transformer(nn.Module):
    def __init__(
        self,
        output_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
        device=torch.device("cpu"),
    ):
        super(Decoder_Transformer, self).__init__()
        self.device = device
        self.max_seq_length = max_seq_length
        # self.positional_encoding = PositionalEncoding(d_model, max_seq_length).to(
        #     device
        # )
        self.embedding_layer = nn.Linear(1, d_model).to(device)
        self.positional_encoding = LearnedPositionalEncoding(
            d_model, max_seq_length
        ).to(device)
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(d_model, num_heads, d_ff, dropout).to(device)
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(d_model, output_size).to(device)
        self.dropout = nn.Dropout(dropout).to(device)

    def generate_mask(self, src, custom_mask=None):
        batch_size, _ = src.size(0), src.size(1)

        assert self.max_seq_length == src.size(
            1
        ), "Input needs to be the max sequence length"

        # Original mask: zeros above the diagonal, ones on and below
        mask = torch.tril(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        ).to(self.device)

        # Expand the mask for the batch size. Shape: [batch_size, 1, seq_length, seq_length]
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1, -1)
        # print(mask.shape)

        # Combine with custom mask if provided
        if custom_mask is not None:
            # Ensure custom_mask is a boolean tensor and has the same device as the original mask
            custom_mask = custom_mask.to(dtype=torch.bool, device=self.device)

            # # Check if custom_mask is of correct shape
            if custom_mask.size() == (batch_size, self.max_seq_length):
                # Add a dimension to custom_mask to match the shape of the original mask
                custom_mask = (
                    custom_mask.unsqueeze(1)
                    .expand(-1, self.max_seq_length, -1)
                    .unsqueeze(1)
                )
                mask = mask & custom_mask

            else:
                raise ValueError("Custom mask must have shape [batch_size, seq_length]")

        return mask

    def Gaussian_loss(
        self, transformer_pred, y_true, epsilon=torch.tensor(1e-6, dtype=torch.float32)
    ):
        epsilon = epsilon.to(self.device)
        # Splitting the output into mean and variance
        mean = transformer_pred[:, :, 0]
        var = torch.nn.functional.softplus(transformer_pred[:, :, 1]) + epsilon

        # Calculating the Gaussian negative log-likelihood loss
        # print(y_true, mean, torch.log(var))
        loss = torch.mean((y_true - mean) ** 2 / var + torch.log(var))

        return loss

    def forward(self, src, custom_mask=None):
        src_mask = self.generate_mask(src, custom_mask=custom_mask)
        src = self.embedding_layer(src)
        src = self.positional_encoding(src)
        # src = self.dropout(src)

        for dec_layer in self.decoder_layers:
            src = dec_layer(src, src_mask)

        output = self.fc(src)
        return output

    def generate(self, src, n_sequence):
        # Generate the next n_sequence elements
        generated_sequence = []

        # Initial input for the model
        current_input = src

        for _ in range(n_sequence):
            # Pass the current input through the model
            output = self.forward(current_input)

            # Get the last output (most recent forecast)
            next_output = torch.normal(
                mean=output[:, -1, 0].unsqueeze(1),
                std=torch.sqrt(
                    torch.nn.functional.softplus(output[:, -1, 1]).unsqueeze(1) + 1e-6,
                ),
            )

            # Append the predicted value to the generated sequence
            generated_sequence.append(next_output[:, 0])  # Assuming output_size is 1

            # Concatenate the new output to the current input for the next iteration
            current_input = torch.cat(
                (
                    current_input[:, -self.max_seq_length + 1 :, :],
                    next_output.unsqueeze(-1),
                ),
                dim=1,
            )

        # Convert the list of tensors to a single tensor
        sequence = torch.stack(
            generated_sequence, dim=1
        )  # Shape: [batch_size, n_sequence]

        return sequence


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

    def generate(self, start, n_sequence):
        # Generate the next n_sequence elements
        generated_sequence = []

        for _ in range(n_sequence):
            # Pass the current input through the model
            output = self.forward(start)

            # Append the predicted value to the generated sequence
            generated_sequence.append(output)

            # Concatenate the new output to the current input for the next iteration
            start = torch.cat((start[:, 1:], output), dim=1)

        return torch.stack(generated_sequence, dim=1)
