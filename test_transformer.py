import torch
from eisforge.ml.eis_gpt.transformer import EISForgeModel

model = EISForgeModel(d_model=128, n_heads=8, n_layers=6)

freq   = torch.logspace(-2, 5, 60).unsqueeze(0)
z_real = torch.rand(1, 60) * 100
z_imag = torch.rand(1, 60) * 50

result = model.predict(freq, z_real, z_imag)

print(f"✅ Transformer OK!")
print(f"   مدار پیش‌بینی‌شده: {result['predicted_circuit']}")
print(f"   اطمینان: {result['confidence']*100:.1f}%")
print(f"   سه کاندید برتر:")
for c in result['top3']:
    print(f"     {c['circuit']}: {c['probability']*100:.1f}%")