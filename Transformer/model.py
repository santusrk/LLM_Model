import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        input_embedding = self.embedding(x) * math.sqrt(self.d_model)
        return input_embedding

class PositionalEmbedding(nn.Module):
    def __init__(self,d_model:int, seq_len:int,dropout:float) -> None:
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len,d_model)
        matrix = torch.zeros(seq_len,d_model)
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        div_term  = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        matrix[:,0::2] = torch.sin(position*div_term)
        matrix[:, 1::2] = torch.cos(position * div_term)
        matrix = matrix.unsqueeze(0)
        self.register_buffer('matrix',matrix)

    def forward(self,x):
        x=x+(self.matrix[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self,eps: float = 10**-6 ) -> None:
        super().__init__()
        self.eps=eps
        self.alpha =nn.Parameter(torch.ones(1)) # multiplicative
        self.bias = nn.Parameter(torch.zeros(1)) # additive

    def forward(self,x):
        mean =x.mean(dim= -1,keepdim = True)
        std = x.std(dim=-1,keepdim =True)
        return  self.alpha * (x-mean)/(std +self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self,d_model:int,dff:int,dropout:float):
        super().__init__()
        self.d_model = d_model
        self.dff =dff
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model,dff) # w1 , b1
        self.linear2 = nn.Linear(dff,d_model) # w2, b2

    def forward(self,x):
        # (batch, seq_len , d_model) ----> (batch, seq_len , dff) --> (batch, seq_len , d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,d_model:int,h:int,dropout:float):
        super().__init__()
        self.d_model = d_model
        self.h=h
        assert d_model % h == 0, "d_model not divisible by h"
        self.dropout= nn.Dropout(dropout)
        self.d_k = d_model//h
        self.w_q = nn.Linear(d_model,d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv
        self.w_o = nn.Linear(d_model, d_model) #Wo

    @staticmethod
    def attention(query,key,value,mask,dropout:nn.Dropout):
        d_k = query.shape[-1]
        attention_score = (query @ key.transpose(-2,-1))/math.sqrt(d_k)
        if mask is not None:
            attention_score.masked_fill_(mask==0,-1e9)
        if dropout is not None:
            attention_score = dropout(attention_score)
        return (attention_score @ value),attention_score

    def forward(self,q,k,v,mask):
        query =self.w_q(q)
        key= self.w_k(k)
        value = self.w_v(v)
        # (batch , seq_len,d_model) ---> (batch,seq_len,h,d_k) ---> (batch,h ,seq_len,d_k)
        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x,self.attention_score = MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)

        # concartinate all the head
        # (batch,h, seq_len, d_k) ---> (batch , seq_len, h, d_k) ----> (batch , seq_len, d_model)
        x = x.transponse(1,2).contiguous().view(x.shape[0],-1,self.h * self.d_k)
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self,dropout: float)->None:
        super().__init__()
        self.dropout= nn.Dropout(dropout)
        self.norm = LayerNormalization()
    def forward(self,x,sublayer):
        return x +self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self,self_attention_block : MultiHeadAttentionBlock,feed_forward_block: FeedForwardBlock,dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout= nn.Dropout(dropout)
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self,x,src_mask):
        x= self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
        x= self.residual_connection[1](x,self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers:nn.ModuleList) -> None:
        self.layers= layers
        self.norm = LayerNormalization()

    def forward(self,x,mask):
        for layer in self.layers:
            x= layer(x,mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self,self_attention_block : MultiHeadAttentionBlock ,cross_attention_block : MultiHeadAttentionBlock, feed_forward_block:FeedForwardBlock,dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = nn.Dropout(dropout)
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        x= self.residual_connection[0](x,lambda x : self.self_attention_block(x,x,x,tgt_mask))
        x= self.residual_connection[1](x,lambda x: self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x=self.residual_connection[2](x,self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self,layers : nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    def forward(self, x, encoder_out,src_mask,tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_out,src_mask,tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model= d_model
        self.vocab_size = vocab_size
        self.proj = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        return torch.log_softmax(self.prj(x),dim =-1)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding : InputEmbedding, tgt_embedding : InputEmbedding, src_position: PositionalEmbedding, tgt_position: PositionalEmbedding, projection_layer : ProjectionLayer):
        super().__init__()
        self.encoder= encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_pos = src_position
        self.tgt_pos = tgt_position
        self.projection_layer = projection_layer

    def encoder(self, src, src_mask):
        src = self.src_embedding(src)
        src= self.src_pos(src)
        return self.encode(src,src_mask)

    def decoder(self,encoder_out,src_mask,tgt,tgt_mask ):
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_out,src_mask,tgt_mask)

    def project(self,x):
        return self.projection_layer(x)




def build_transformer(input_vocab_size:int,out_vocab_size:int,input_seq_len:int,out_seq_len:int,d_model:int = 512,N:int = 6, h:int =8,dropout:float = 0.1, dff:int=2048):
    input_embedding = InputEmbedding(d_model , input_vocab_size)
    output_embedding = InputEmbedding(d_model, out_vocab_size)
    input_positional_embedding = PositionalEmbedding(d_model, input_seq_len,dropout)
    output_positional_embedding = PositionalEmbedding(d_model, out_seq_len,dropout)
    encoder_blocks=[]
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model , h ,dropout)
        feed_forward_block = FeedForwardBlock(d_model,dff,dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block,feed_forward_block,dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks=[]
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block = FeedForwardBlock(d_model, dff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block,decoder_cross_attention_block,feed_forward_block,dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    proection_layer = ProjectionLayer(d_model, out_vocab_size)

    transformer = Transformer(encoder,decoder,input_embedding,output_embedding,input_positional_embedding,output_positional_embedding,proection_layer)
    # initialize the parameter
    for p in transformer.parameters():
        if p.dim() >1 :
            nn.init.xavier_uniform_(p)

    return transformer




















sentence = 'My name is srikant'
tokens = sentence.lower().split()
vocab_size =len(set(tokens))
d_model = 512
token_to_index = {token: idx for idx, token in enumerate(set(tokens))}

indices = [token_to_index[token] for token in tokens]
indices_tensor = torch.tensor(indices, dtype=torch.long)

embedding =InputEmbedding(d_model,vocab_size)
input_embedding = embedding.forward(indices_tensor)


positional_embedding = PositionalEmbedding(d_model,vocab_size,0.3)
pos_embedding =positional_embedding.forward(input_embedding)

layer_norm=LayerNormalization()
layer_norm =(layer_norm.forward(pos_embedding))

ffn= FeedForwardBlock(512,2048,0.3)
feed_forward_nn = ffn.forward(layer_norm)
print(feed_forward_nn)







