from io import BufferedWriter, FileIO
import sys
import os
import numpy as np
import jsonlines
import json
from transformers import PreTrainedTokenizerFast
import struct
from tqdm import tqdm

class file_header(object):
    def __init__(self, magic_header = 'xiaoyaowudi data'):
        self.magic_header = magic_header.encode('ascii')
    def set(self, version, total_len, num_sizes, dtype):
        self.version = version
        self.total_len = total_len
        self.num_sizes = num_sizes
        self.dtype = dtype
    def unpack(self, header): # 64 bytes
        if header[:16] != self.magic_header:
            raise Exception('Header not match!')
        header = header[16:]
        one = struct.unpack('<32B', header[:32])
        total_len, num_sizes = struct.unpack('<2Q', header[32:])
        if one[0] != 1:
            raise Exception('Version unsupported')
        if one[1] == 1:
            self.dtype = np.uint8
        elif one[1] == 2:
            self.dtype = np.uint16
        elif one[1] == 4:
            self.dtype = np.uint32
        elif one[1] == 8:
            self.dtype = np.uint32
        else:
            raise Exception('Integer size unsupported')
        self.version = one[0]
        self.total_len = total_len
        self.num_sizes = num_sizes
    def output(self):
        return self.magic_header + np.array([self.version, np.array([1], dtype = self.dtype).itemsize] + [0,] * 30, dtype ='<u1').tobytes() +\
                                   np.array([self.total_len, self.num_sizes], dtype = '<u8').tobytes()

def output_bin(lens, output_file_path, dtype = np.uint16):
    lens = lens.astype(np.uint64).flatten()
    header = file_header()
    header.set(1, np.sum(lens), lens.shape[0], dtype)
    with open(output_file_path, 'wb') as f:
        f.write(header.output() + lens.tobytes())

def refine_context(context):
    context = context.strip().split('\n')
    for c in range(len(context)):
        context[c] = context[c].strip().strip('\u3000').strip('\r')
    context = list(filter(lambda c: c != '', context))
    context = '\n' + ('\n'.join(context)).strip()
    if context == '':
        context = '\n'
    return context

if __name__ == '__main__':
    row_limit = int(sys.argv[1].strip())
    in_file = sys.argv[2].strip()
    out_path = sys.argv[3].strip()
    out_name = os.path.splitext(os.path.basename(in_file))[0]
    bin_name = os.path.join(out_path, out_name + '.bin')
    idx_name = os.path.join(out_path, out_name + '.idx')
    tmp_name = os.path.join(out_path, out_name + '.tmp')
    tokenizer = PreTrainedTokenizerFast(tokenizer_file = './20B_tokenizer.json')
    sizes = []
    with BufferedWriter(FileIO(tmp_name, "wb"), buffer_size = (2**20)) as writer:
        with jsonlines.open(in_file, mode = 'r') as reader:
            count = 0
            for row in reader:
                text = refine_context(row['text'])
                tokens = tokenizer.encode(text)
                tokens.append(0)
                writer.write(np.array(tokens, dtype = np.uint16).tobytes())
                sizes.append(len(tokens))
                count += 1
                if row_limit > 0 and count>=row_limit:
                    break
                if count%10 ==0:
                    print(count, end = ('\n' if count % 100 ==0 else ' '))
    print('\nEnd of tokenization')
    sizes = np.array(sizes)
    cum_sizes = np.cumsum(sizes)
    indices = np.arange(0, sizes.shape[0]);
    np.random.shuffle(indices)
    tmp_buffer = memoryview(np.memmap(tmp_name, mode = 'r'))
    with BufferedWriter(FileIO(bin_name, "wb"), buffer_size = (2**20)) as writer:
        for idx in tqdm(indices):
            passage = np.frombuffer(tmp_buffer, dtype = np.uint16, count = sizes[idx], offset = (cum_sizes[idx] - sizes[idx]) * 2)
            # print(tokenizer.decode(passage))
            writer.write(passage.tobytes())
    output_bin(sizes[indices], idx_name)
