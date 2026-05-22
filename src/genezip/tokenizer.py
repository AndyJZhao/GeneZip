import numpy as np

RC_MAP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")

def revcomp_seq(s: str) -> str:
    return s.translate(RC_MAP)[::-1]

class SingleNucleotideTokenizer:
    pad = 0
    A, C, G, T = 65, 67, 71, 84
    a, c, g, t = 97, 99, 103, 116
    informative_min = 65
    informative_max = 90
    repetitive_offset = 32

    def __init__(self, RC_augmentation=False):
        self.RC_augmentation = RC_augmentation

    def __call__(self, text):
        if self.RC_augmentation and (np.random.rand() < 0.5):
            text = revcomp_seq(text)
        input_ids = np.frombuffer(text.encode("ascii"), dtype=np.uint8)
        return input_ids.astype(int)

    def save_pretrained(self, output_dir, **kwargs):
        pass

class FastSingleNucleotideTokenizer:
    pad = 0
    informative_min = 3
    informative_max = 6
    repetitive_offset = 4

    def __call__(self, text):
        input_ids = []
        for ch in text:
            if ch == "A":
                input_ids.append(3)
            elif ch == "C":
                input_ids.append(4)
            elif ch == "G":
                input_ids.append(5)
            elif ch == "T":
                input_ids.append(6)
            elif ch == "a":
                input_ids.append(7)
            elif ch == "c":
                input_ids.append(8)
            elif ch == "g":
                input_ids.append(9)
            elif ch == "t":
                input_ids.append(10)
            else:
                input_ids.append(11)
        return np.array(input_ids, dtype=int)

    def save_pretrained(self, output_dir, **kwargs):
        pass

class SixMerTokenizer:
    K = 6
    pad = 0
    bos = 1
    eos = 2
    K_limit = (4 ** K) - 1
    repetitive_offset = 6000
    ascii2digit = {65: 0, 67: 1, 71: 2, 84: 3}
    left_pad_offsets = [0]
    for i in range(K):
        left_pad_offsets.append(left_pad_offsets[-1] + 4 ** (K - i))
    informative_min = 3
    informative_max = left_pad_offsets[-1] + 2

    def __call__(self, seq):
        input_ids = []
        left_pad = self.K
        val = 0
        for ch in seq:
            left_pad = max(0, left_pad - 1)
            digit = ord(ch)
            if digit == 64:
                input_ids.append(self.eos)
                left_pad = self.K
                val = 0
            else:
                if digit > 96:
                    digit = digit - 32
                    repetitive_offset = self.repetitive_offset
                else:
                    repetitive_offset = 0
                if digit not in self.ascii2digit:
                    input_ids.append(self.eos)
                    left_pad = self.K
                    val = 0
                else:
                    digit = self.ascii2digit[digit]
                    val = ((val << 2) | digit) & self.K_limit
                    input_ids.append(val + self.left_pad_offsets[left_pad] + repetitive_offset + self.informative_min)
        return np.array(input_ids)

    def save_pretrained(self, output_dir, **kwargs):
        pass
