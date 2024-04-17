import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
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
        #print(q_times_k == q_times_k)
        softmax = Softmax()
        softmaxxed = softmax(q_times_k, -1)
        #print(softmaxxed)
        if dpout != None:
            dropout = nn.Dropout(p=dpout, inplace=True)
            softmaxxed = dropout(softmaxxed)
        output = softmaxxed @ v
        #print(output)
        return output
    
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
        
        self.w0 = nn.Linear(self.d_model, self.d_value*self.num_heads, bias=False)

    def forward(self, x):
        seq_len = x.shape[-2]
        lower_triangular = torch.triu(torch.ones(seq_len, seq_len), diagonal = 1).bool()
        attention = Attention()
        
        k = x @ torch.t(self.k.weight.data)
        q = x @ torch.t(self.q.weight.data)
        v = x @ torch.t(self.v.weight.data)
        
        attn_output = []
        for n in range(self.num_heads):
            attn_output.append(attention(k[...,n*self.d_key:(n+1)*self.d_key], q[...,n*self.d_key:(n+1)*self.d_key], v[...,n*self.d_key:(n+1)*self.d_key], mask=lower_triangular, dpout=self.p_drop))
        attn_output = torch.cat(attn_output, dim=-1)
        #print(attn_output)
        return attn_output @ torch.t(self.w0.weight.data)
    
class PreNormBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, attn_pdrop=None, residual_pdrop=None):
        super(PreNormBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop
        
        self.norm1 = RMSNorm(self.d_model, 1e-05)
        self.norm2 = RMSNorm(self.d_model, 1e-05)
        self.multihead = MultiHeadAttention(self.d_model, self.num_heads, attn_pdrop=None)
        self.ffn = FeedForward(self.d_model, self.d_ff)
        if self.residual_pdrop != None:
            self.residual_dropout = nn.Dropout(p=self.residual_pdrop, inplace=True)

    def forward(self, x):
        residual_d_model = x

        norm1_d_model = self.norm1(residual_d_model)
        post_attention_d_model = self.multihead(norm1_d_model)
        if self.residual_dropout != None:
            post_attention_d_model = self.residual_dropout(post_attention_d_model)
        
        residual_d_model = residual_d_model + post_attention_d_model

        norm2_d_model = self.norm2(residual_d_model)
        post_ffn_d_model = self.ffn(norm2_d_model)
        if self.residual_dropout != None:
            post_ffn_d_model = self.residual_dropout(post_ffn_d_model)
       
        output = residual_d_model + post_ffn_d_model
        return output

class Transformer(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, attn_pdrop=None, residual_pdrop=None):
        super(Transformer, self).__init__()
        assert d_model % num_heads == 0
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop
        self.num_layers = num_layers

        self.final_norm = RMSNorm(self.d_model, 1e-05)
        self.blocks = nn.ModuleList([PreNormBlock(self.d_model, self.num_heads, self.d_ff, attn_pdrop=attn_pdrop, residual_pdrop=self.residual_pdrop) for _ in range(self.num_layers)])
        
        if self.residual_pdrop != None and self.residual_pdrop != 0:
            self.initial_dropout = nn.Dropout(p=self.residual_pdrop, inplace=True)

        self.vocab_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.positional_embedding = nn.Embedding(self.context_length, self.d_model)
        self.output_linear = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(self, x):
        tok_embedding_d_model = self.vocab_embedding(x)
        context_vector = torch.arange(x.size(-1)) % self.context_length
        pos_embedding_d_model = self.positional_embedding(context_vector)
        x_embedded_d_model = tok_embedding_d_model + pos_embedding_d_model

        if self.residual_pdrop != None and self.residual_pdrop != 0:
            x_embedded_d_model = self.initial_dropout(x_embedded_d_model)
        
        for attn in self.blocks:
            x_embedded_d_model = attn(x_embedded_d_model)

        output = self.final_norm(x_embedded_d_model)
        output = self.output_linear(output)
        
        return output
