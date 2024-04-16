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
        
        self.q = nn.Linear(self.d_key*self.num_heads, self.d_model, bias=False)
        self.k = nn.Linear(self.d_key*self.num_heads, self.d_model, bias=False)
        self.v = nn.Linear(self.d_value*self.num_heads, self.d_model, bias=False)
        """self.q1 = nn.Linear(self.d_model, self.d_key, bias=False)
        self.k1 = nn.Linear(self.d_model, self.d_key, bias=False)
        self.v1 = nn.Linear(self.d_model, self.d_value, bias=False)
        self.q2 = nn.Linear(self.d_model, self.d_key, bias=False)
        self.k2 = nn.Linear(self.d_model, self.d_key, bias=False)
        self.v2 = nn.Linear(self.d_model, self.d_value, bias=False)"""
        self.w0 = nn.Linear(self.d_model, self.d_value*self.num_heads, bias=False)
        #print(self.d_model)
        # Define linear transformation for output after concatenation
        
    def forward(self, x):
        #print('hello')
        #print(self.q.data.shape)
        seq_len = x.shape[-2]
        lower_triangular = torch.triu(torch.empty(seq_len, seq_len), diagonal = 1)
        lower_triangular = torch.where(lower_triangular == 0, False, True)
        """print(lower_triangular)
        print(lower_triangular.shape)"""
        
        attention = Attention()
        """k1 = x @ self.k1.data.t_()
        q1 = x @ self.q1.data.t_()
        v1 = x @ self.v1.data.t_()
        attn_output1 = attention(k1, q1, v1, mask=lower_triangular, dpout=self.p_drop)

        k2 = x @ self.k2.data.t_()
        q2 = x @ self.q2.data.t_()
        v2 = x @ self.v2.data.t_()
        attn_output2 = attention(k2, q2, v2, mask=lower_triangular, dpout=self.p_drop)"""

        k = x @ self.k.data.t_()
        q = x @ self.q.data.t_()
        v = x @ self.v.data.t_()

        """print(torch.allclose(k, torch.cat([k1, k2], dim=-1)))
        print(torch.allclose(q, torch.cat([q1, q2], dim=-1)))
        print(torch.allclose(v, torch.cat([v1, v2], dim=-1)))"""
        attn_output = []
        for n in range(self.num_heads):
            attn_output.append(attention(k[...,n*self.d_key:(n+1)*self.d_key], q[...,n*self.d_key:(n+1)*self.d_key], v[...,n*self.d_key:(n+1)*self.d_key], mask=lower_triangular, dpout=self.p_drop))
        #attn_output = torch.cat([attn_output1, attn_output2], dim=-1)
        attn_output = torch.cat(attn_output, dim=-1)
        return attn_output @ self.w0.data.t_()