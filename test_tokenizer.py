import torch
from eisforge.ml.eis_gpt.tokenizer import EISTokenizer

tokenizer = EISTokenizer(d_model=128)
freq   = torch.logspace(-2, 5, 60).unsqueeze(0)
z_real = torch.rand(1, 60) * 100
z_imag = torch.rand(1, 60) * 50

output = tokenizer(freq, z_real, z_imag)
print(f'✅ Tokenizer OK! output shape: {output.shape}')
print(f'   انتظار: torch.Size([1, 60, 128])')