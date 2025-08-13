from safetensors.torch import safe_open, save_file
import torch
import math

FP4_VALUES = [
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]

def dequantize(blocks, scales, dtype=torch.bfloat16, rows_per_chunk=16384 * 512):
	scales = scales.to(torch.int32) - 127
	assert blocks.shape[:-1] == scales.shape, (
		f"{blocks.shape=} does not match {scales.shape=}"
	)
	lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

	*prefix_shape, G, B = blocks.shape
	rows_total   = math.prod(prefix_shape) * G

	blocks = blocks.reshape(rows_total, B)
	scales = scales.reshape(rows_total, 1)

	out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

	for r0 in range(0, rows_total, rows_per_chunk):
		r1 = min(r0 + rows_per_chunk, rows_total)

		blk = blocks[r0:r1]
		exp = scales[r0:r1]

		# nibble indices -> int64
		idx_lo = (blk & 0x0F).to(torch.long)
		idx_hi = (blk >> 4).to(torch.long)

		sub = out[r0:r1]
		sub[:, 0::2] = lut[idx_lo]
		sub[:, 1::2] = lut[idx_hi]

		torch.ldexp(sub, exp, out=sub)
		del idx_lo, idx_hi, blk, exp
	return out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)

input_path = "original_safetensors"
output_path = "model.safetensors"

new_tensors = {}

with safe_open(input_path, framework="pt", device="cpu") as f:
    keys = set(f.keys())

    for key in keys:
        if key.endswith(".blocks"):
            base_name = key[:-len(".blocks")]
            scales_name = base_name + ".scales"

            if scales_name in keys:
                # Dequantize
                blocks_tensor = f.get_tensor(key)
                scales_tensor = f.get_tensor(scales_name)
                new_tensors[base_name] = dequantize(blocks_tensor, scales_tensor)
            else:
                # If no scales, skip or handle error
                pass

        elif key.endswith(".scales"):
            # Skip scales because we already handled them
            continue
        else:
            # Non-quantized tensor, copy as-is
            new_tensors[key] = f.get_tensor(key)

save_file(new_tensors, output_path)
print(f"Saved dequantized file without .blocks/.scales to {output_path}")

