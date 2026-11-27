# Compatibility fixture

`legacy_conditioning.npz` contains CPU predictions and state dictionaries captured
from the original `src/loadit/models.py` at commit `8d589b3`, before the conditioning
fixes. It covers ternary labels with and without temperature, an unconditioned
model, the legacy binary layout, and the older multi-hot layout.

Models use `input_size=(4, 2)`, `patch_size=(1, 2)`, hidden size 16, depth 1, eight
attention heads, and four calendar variables. Seed 123 initializes the model and
inputs. The final projection and AdaLN modulation weights are randomized with
standard deviation 0.05 so the fixture detects conditioning changes rather than
comparing zero-initialized outputs. Captured with PyTorch 2.5.1 and timm 1.0.19.
