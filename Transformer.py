import torch
import torch.nn as nn
import math


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

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)

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
        return self.fc2(self.relu(self.fc1(x)))


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
        # print("x", x)
        attn_output = self.self_attn(x, x, x, src_mask)
        # print("attention", (x + self.dropout(attn_output)).shape)
        # x = self.norm1(x + self.dropout(attn_output))
        x = x + self.dropout(attn_output)
        # print("norm1", x)
        ff_output = self.feed_forward(x)
        # print("ff", ff_output)
        # x = self.norm2(x + self.dropout(ff_output))
        x = x + self.dropout(ff_output)
        # print("norm2", x)
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
    ):
        super(Decoder_Transformer, self).__init__()
        self.max_seq_length = max_seq_length
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src):
        batch_size, seq_length = src.size(0), src.size(1)

        # Create a mask of shape [seq_length, seq_length] with ones above
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)

        # Expand the mask for the batch size. Shape: [batch_size, 1, seq_length, seq_length]
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1, -1)

        return mask

    def forward(self, src):
        src_mask = self.generate_mask(src)
        # src = self.positional_encoding(src)
        # src = self.dropout(src)

        for dec_layer in self.decoder_layers:
            # print(src.shape, src)
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
                std=torch.abs(output[:, -1, 1]).unsqueeze(1) + 1e-8,
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
