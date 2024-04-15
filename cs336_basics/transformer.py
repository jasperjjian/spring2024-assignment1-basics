import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float):
        super(RMSNorm, self).__init__()
        self.dim = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(self.dim))

    def forward(self, x):
        x_squared = torch.square(x)
        denominator = torch.sqrt(torch.mean(x_squared, dim=-1, keepdim=True) + self.eps)
        x_normalized = (x / denominator) * self.gain
        return x_normalized
    
class Gelu(nn.Module):
    def forward(self, x):
        return x * 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))
    
class Softmax(nn.Module):
    def forward(self, x, dim):
        max = torch.max(x, dim=dim, keepdim=True).values
        x = torch.exp(x - max)
        x = x / torch.sum(x, dim=dim, keepdim=True)
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Linear(self.d_model, self.d_ff, bias=False)
        self.w2 = nn.Linear(self.d_ff, self.d_model, bias=False)
        self.activation = Gelu()
    
    def forward(self, x):
        non_linearity_d_ff = self.activation(self.w1(x))
        w2_multiply_d_model = self.w2(non_linearity_d_ff)
        return w2_multiply_d_model
    
class Attention(nn.Module):
    def forward(self, k, q, v, mask=None, dpout=None):
        d_k = k.shape[-1]
        """print(k.shape)
        print(q.shape)
        print(v.shape)
        print(mask.shape)"""
        q_times_k = (q @ torch.transpose(k, -1, -2)) / torch.sqrt(torch.tensor(d_k))
        
        if mask != None:
            add_mask = torch.where(mask, torch.tensor(float('-inf')), torch.tensor(0))
            q_times_k += add_mask
        softmax = Softmax()
        softmaxxed = softmax(q_times_k, -1)

        if dpout != None:
            dropout = nn.Dropout(p=dpout, inplace=True)
            softmaxxed = dropout(softmaxxed)
        
        return softmaxxed @ v
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_pdrop=None):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.p_drop = attn_pdrop
        self.d_key = d_model // num_heads
        self.d_value = d_model // num_heads
        
        self.q = nn.Linear(self.d_model, self.d_key * self.num_heads)
        self.k = nn.Linear(self.d_model, self.d_key * self.num_heads)
        self.v = nn.Linear(self.d_model, self.d_value * self.num_heads)
        self.w0 = nn.Linear(self.d_model, self.d_value * self.num_heads)
        print(self.d_model)
        # Define linear transformation for output after concatenation
        


    def forward(self, x):
        print('hello')
        print(self.q.data.shape)
        seq_len = x.shape[-2]
        lower_triangular = torch.triu(torch.empty(seq_len, seq_len), diagonal = 1)
        lower_triangular = torch.where(lower_triangular == 0, False, True)
        print(lower_triangular)
        
        print(lower_triangular.shape)
        concat_kqv = torch.cat([self.k.data, self.q.data, self.v.data], dim=1)
        print(concat_kqv.shape)

        projected = x @ concat_kqv
        print(projected.shape)
        
        attention = Attention()
        k = projected[..., 0:(self.d_key * self.num_heads)]
        q = projected[..., (self.d_key * self.num_heads):(2*self.d_key * self.num_heads)]
        v = projected[..., (2*self.d_key * self.num_heads):]
        attn_output = attention(k, q, v, mask=lower_triangular, dpout=self.p_drop)

        return self.w0(attn_output)