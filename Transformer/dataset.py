import torch
import torch as nn
from torch.utils.data import DataLoader, Dataset
from typing import Any

class BilingualDataset(Dataset):
    def __init__(self,ds,tokenizer_src,tokenizer_tgt,src_lng,tgt_lng,seq_len)->None:
        super().__init__()
        self.ds=ds
        self.tokenizer_src=tokenizer_src
        self.tokenizer_tgt=tokenizer_tgt
        self.src_lng=src_lng
        self.tgt_lng=tgt_lng
        self.seq_len =seq_len

        self.sos_token = torch.Tensor([tokenizer_src.token_to_id(['[SOS]'])] , dtype= torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id(['[EOS]'])], dtype=torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id(['[PAD]'])], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)



    def __getitem(self,index:Any)-> Any:
        src_target_pair = self.ds[index]
        src_text=src_target_pair['translation'][self.src_lng]
        tgt_text = src_target_pair['translation'][self.tgt_lng]

        input_token = self.tokenizer_src.encode(src_text).ids
        output_token = self.tokenizer_tgt.encode(tgt_text).ids

        input_padding_token = self.seq_len - len(input_token)-2
        output_padding_token = self.seq_len - len(output_token) - 1

        if input_padding_token < 0 or output_padding_token <0 :
            raise ValueError('Sentence is too long')
        # add SOS and EOS to the source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(input_token,dtype= torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * input_padding_token ,dtype=torch.int64)
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(output_token, dtype=torch.int64),
                torch.tensor([self.pad_token] * output_padding_token, dtype=torch.int64)
            ]
        )

        label = torch.cat(
            [

                torch.tensor(output_token, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * output_padding_token, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0)==self.seq_len
        assert decoder_input.size(0)==self.seq_len
        assert label.size(0)==self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0))
        }


def causal_mask(size):
    mask = torch.triu(torch.ones(1,size,size),diagonal=1).type(torch.int)
    return mask==0


