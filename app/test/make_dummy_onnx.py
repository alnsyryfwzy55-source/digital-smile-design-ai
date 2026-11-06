"""Hand-encodes a tiny ONNX model with the same interface as teeth_unet_v2.onnx (image 1x3x384x384 -> logits 1x7x384x384). For pipeline tests only."""
import struct, sys, numpy as np
def varint(n):
    out = b''
    while True:
        b = n & 0x7f; n >>= 7
        out += bytes([b | (0x80 if n else 0)])
        if not n: return out
def key(f, wt): return varint((f << 3) | wt)
def vint(f, n): return key(f, 0) + varint(n)
def ld(f, b): b = b if isinstance(b, bytes) else b.encode(); return key(f, 2) + varint(len(b)) + b
def shape(dims): return b''.join(ld(1, vint(1, d)) for d in dims)
def vinfo(name, dims): return ld(1, name) + ld(2, ld(1, vint(1, 1) + ld(2, shape(dims))))
W = np.zeros((7, 3, 1, 1), np.float32)
W[0, :, 0, 0] = [-0.6, -0.6, -0.6]; W[3, :, 0, 0] = [0.5, 0.5, 0.5]   # bright → class 3, dark → background
B = np.array([0.5, -3, -3, -0.2, -3, -3, -3], np.float32)
def attr_ints(name, vals): return ld(5, ld(1, name) + b''.join(vint(8, v) for v in vals) + vint(20, 7))
def attr_int(name, v): return ld(5, ld(1, name) + vint(3, v) + vint(20, 2))
node = (ld(1, 'image') + ld(1, 'W') + ld(1, 'B') + ld(2, 'logits') + ld(3, 'conv') + ld(4, 'Conv') + attr_ints('dilations', [1, 1]) + attr_int('group', 1)
        + attr_ints('kernel_shape', [1, 1]) + attr_ints('pads', [0, 0, 0, 0]) + attr_ints('strides', [1, 1]))
t_w = b''.join(vint(1, d) for d in W.shape) + vint(2, 1) + ld(8, 'W') + ld(9, W.tobytes())
t_b = vint(1, 7) + vint(2, 1) + ld(8, 'B') + ld(9, B.tobytes())
graph = ld(1, node) + ld(2, 'g') + ld(5, t_w) + ld(5, t_b) + ld(11, vinfo('image', [1, 3, 384, 384])) + ld(12, vinfo('logits', [1, 7, 384, 384]))
model = vint(1, 8) + ld(2, 'dummy') + ld(7, graph) + ld(8, ld(1, '') + vint(2, 17))
open(sys.argv[1], 'wb').write(model)
