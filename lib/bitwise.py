import numpy as np

# Flip the i-th bit of integer a
def bitflip(a, i):
    pos = 2**i
    return a-pos if a&pos else a+pos

# Convert an integer to a list of bits
def bitlist(num, nbit):
    return [int(bool(num & (1 << idx))) for idx in range(nbit)]

def bitlist2activation(lst):
    return np.array([1 if bit == 1 else -1 for bit in lst])