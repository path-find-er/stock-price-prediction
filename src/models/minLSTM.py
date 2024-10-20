# This file was automatically generated# Import necessary libraries
import torch
import torch.nn as nn
from tqdm import tqdm
import torch
import torch.nn as nn

def safe_log(x, eps=1e-8):
    # Computes a numerically stable logarithm
    return torch.log(x + eps)

class minLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(minLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.forget_gate_layer = nn.Linear(input_dim, hidden_dim)
        self.input_gate_layer = nn.Linear(input_dim, hidden_dim)
        self.candidate_hidden_layer = nn.Linear(input_dim, hidden_dim)
                
    def forward(self, input_seq, initial_hidden=None):
        # input_seq: Input tensor of shape (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = input_seq.size()

        if initial_hidden is None:
            # Initialize initial_hidden to zeros if not provided
            initial_hidden = torch.zeros(batch_size, self.hidden_dim, device=input_seq.device)

        # Compute forget_gate and input_gate using sigmoid activations
        # As per the minLSTM equations in the paper (Section 3.2.4):
        # "forget_gate = σ(Linear_dh(x_t))"
        # "input_gate = σ(Linear_dh(x_t))"
        forget_gate = torch.sigmoid(self.forget_gate_layer(input_seq))  # Shape: (batch_size, seq_len, hidden_dim)
        input_gate = torch.sigmoid(self.input_gate_layer(input_seq))  # Shape: (batch_size, seq_len, hidden_dim)

        # Normalize the gates to ensure forget_gate_norm + input_gate_norm = 1
        # This corresponds to Step 3 in Section 3.2.3 of the paper:
        # "We ensure LSTM's output is time-independent in scale."
        # "We normalize the two gates: forget_gate_norm, input_gate_norm ← forget_gate / (forget_gate + input_gate), input_gate / (forget_gate + input_gate)"
        gate_sum = forget_gate + input_gate + 1e-8  # Add epsilon for numerical stability
        # print(torch.allclose(forget_gate + input_gate, torch.ones_like(forget_gate + input_gate), atol=1e-5))

        forget_gate_norm = forget_gate / gate_sum
        input_gate_norm = input_gate / gate_sum

        # Compute candidate_hidden as per the minLSTM equations:
        # "candidate_hidden = Linear_dh(x_t)"
        candidate_hidden = self.candidate_hidden_layer(input_seq)  # Shape: (batch_size, seq_len, hidden_dim)

        # Set a_t = forget_gate_norm, b_t = input_gate_norm ⊙ candidate_hidden
        # This corresponds to the recurrence h_t = a_t ⊙ h_{t−1} + b_t
        # As per Algorithm 4 in Section A.1.2 of the paper:
        # "h_{1:t} ← ParallelScan(forget_gate_norm_{1:t}, [initial_hidden, input_gate_norm_{1:t} ⊙ candidate_hidden_{1:t}])"
        # a_t = forget_gate_norm  # Shape: (batch_size, seq_len, hidden_dim)
        # b_t = input_gate_norm * candidate_hidden  # Shape: (batch_size, seq_len, hidden_dim)
        memory_retention = forget_gate_norm  # Shape: (batch_size, seq_len, hidden_dim)
        new_information = input_gate_norm * candidate_hidden  # Shape: (batch_size, seq_len, hidden_dim)

        # Compute cumulative products of a_t in log-space for numerical stability
        # This follows the implementation details in Section B.1 of the paper:
        # "We consider a log-space implementation, which takes as input log(a_{1:t}) and log(b_{0:t})"
        memory_retention_log = safe_log(memory_retention)
        log_prod_memory_retention = torch.cumsum(memory_retention_log, dim=1)  # Cumulative sum over time

        # Compute s_k = prod_{j=k+1}^t a_j
        # As per the implementation in the paper's Appendix B.1:
        # "Compute s_k = exp(log_prod_a_T - log_prod_a_k)"
        padded_log_prod_memory_retention = torch.cat(
            [torch.zeros((batch_size, 1, self.hidden_dim), device=input_seq.device), log_prod_memory_retention],
            dim=1
        )  # Shape: (batch_size, seq_len + 1, hidden_dim)

        # Compute s_k for each time step
        # s_k = exp(log_prod_a_T - log_prod_a_{k})
        scaling_factor = torch.exp(
            log_prod_memory_retention[:, -1:, :] - padded_log_prod_memory_retention[:, :-1, :]
        )  # s_k shape: (batch_size, seq_len, hidden_dim)

        # Adjust b_t using s_k
        # "Adjusted b_t = s_k ⊙ b_t" (from Section B)
        adjusted_new_information = scaling_factor * new_information

        # Compute hidden_states by summing over adjusted_b_t
        # As per the log-space implementation in Section B.1:
        # "Compute hidden_states = cumsum(adjusted_b_t, dim=1)"
        hidden_states = torch.cumsum(adjusted_new_information, dim=1)

        # Compute prod_{j=1}^t a_j
        prod_memory_retention = torch.exp(log_prod_memory_retention)

        # Add the term (prod_{j=1}^t a_j) ⊙ initial_hidden
        # This incorporates the initial hidden state initial_hidden into the computation
        hidden_states += prod_memory_retention * initial_hidden.unsqueeze(1)

        return hidden_states  # Output shape: (batch_size, seq_len, hidden_dim)

class PriceVolumePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PriceVolumePredictor, self).__init__()
        self.minlstm = minLSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
                
    def forward(self, x):
        h = self.minlstm(x)  # h shape: (batch_size, seq_len, hidden_size)
        out = self.fc(h[:, -1, :])  # Use the last hidden state for prediction
        return out