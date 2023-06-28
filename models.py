import torch
import numpy as np

def VQ(enc_embed : torch.Tensor, codebook : torch.Tensor):
    #encoder embed: batch_size x Nz
    #codebook: KxNz
    similarity = enc_embed @ codebook.T
    codebook_idxs = torch.argmax(similarity, dim=1)
    return codebook_idxs

if __name__ == '__main__':
    cont_idxs = np.random.choice(5, 10)
    encoder_embed = torch.eye(5)[cont_idxs]
    print(encoder_embed.shape)
    print(cont_idxs)
    codebook = torch.eye(5)
    # print(codebook)

    idxs = VQ(encoder_embed, codebook)
    print(idxs)

