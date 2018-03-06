from my_wavelet import *
from my_quantization import *
from img_util import *
import struct
import numpy as np
import gzip
import matplotlib.pyplot as plt
import os

from my_compression import *







# create a quantizer with num_bit (including 1 sign bit)
quantizer = lloyd_max_quantizer(num_bit=8)
# wavelet type, choice: 'D6','D4', 'Haar'(default)
g_wavelet_type = 'D6'

def compress(img, fpath, thres_percentile = 95, flg_write_file = False, wavelet_type = 'D4'):
    # wavelet transform
    transformed = TransformImg(img, wavelet_type=wavelet_type)
    # quantization:
    # sign: uint8 matrix(256X256) : -1 for -, 1 for +
    # code: uint8 matrix(256X256) : 0~127
    # book: float64 array(128,): map to avg of the bin
    sign, code, book = quantizer.quantize(transformed, thres_percentile)

    sign_zigzag = ''
    #zigzag traversal for sign and code
    code_zigzag = np.empty((256 * 256,), dtype=np.uint8)
    for m_idx, zig_idx in enumerate(zigzag_indexs(256)):
        code_zigzag[m_idx] = code[zig_idx[0], zig_idx[1]]
        sign_zigzag += '1' if sign[zig_idx[0], zig_idx[1]] > 0 else '0'

    # # without zigzag
    # code_zigzag = code.flatten()
    # for s in sign.flatten():
    #     sign_zigzag += '1' if s > 0 else '0'

    # serialize into one bytearray
    # format: code [:65536], sign[65536:65536+8192], book[65536+8192:]
    code_bytes = bytearray(code_zigzag.tobytes())
    # sign matrix: zigzag and convert to bit
    sign_bytes = str_to_byteArray(sign_zigzag)
    code_bytes.extend(sign_bytes)
    book_bytes = bytearray(book.tobytes())
    code_bytes.extend(book_bytes)

    # save file
    if flg_write_file:
        with gzip.open(fpath, 'wb') as f:
            f.write(code_bytes)

    return code_bytes

def decompress(fpath, wavelet_type = 'D4'):
    with gzip.open(fpath, 'rb') as f:
        file_content = f.read()

    return decompress_bytes(file_content, wavelet_type)

def decompress_bytes(file_content,wavelet_type = 'D4'):
    code_bytes_decompressed = file_content[:65536]  # 65536 bytes
    sign_bytes_decompressed = file_content[65536:65536 + 8192]  # 8192 bytes
    book_bytes_decompressed = file_content[65536 + 8192:]  # 1024 bytes

    sign_str = byteArray_to_str(sign_bytes_decompressed)

    code_decom = np.zeros((256, 256), dtype=np.uint8)
    sign_decom = np.zeros((256, 256), dtype=np.float32)

    # zigzag traversal
    for m_idx, zig_idx in enumerate(zigzag_indexs(256)):
        code_decom[zig_idx[0], zig_idx[1]] = code_bytes_decompressed[m_idx]
        sign_decom[zig_idx[0], zig_idx[1]] = -1. if sign_str[m_idx] == '0' else 1.

    # for idx in range(code_decom.size):
    #     code_decom[idx//256,idx%256] = code_bytes_decompressed[idx]
    #     sign_decom[idx//256, idx%256] = -1. if sign_str[idx] =='0' else 1.

    book_floats = []
    b_cnt = 0
    buf = bytearray()
    for byte in book_bytes_decompressed:

        buf.append(byte)
        b_cnt += 1

        if b_cnt == 8:
            # unpack into float64
            book_floats.append(struct.unpack('d', buf))
            b_cnt = 0
            buf = bytearray()

    book_decom = np.asarray(book_floats, dtype=np.float32)

    decompressed = quantizer.dequantize(sign_decom, code_decom, book_decom)
    return InverseTransform(decompressed, wavelet_type)

def demo_percentile(img,  thres_percent, flg_write = False, tgt_fpath=None, wavelet_type = 'D4'):
    byte_array = compress(img, tgt_fpath, thres_percent,wavelet_type=wavelet_type, flg_write_file=flg_write)
    dec_img = decompress_bytes(byte_array, wavelet_type=wavelet_type)
    dec_img = dec_img
    return dec_img

def binary_trial(img, start_thres_p = 10, pnr_target = 24, trial_time = 8):
    curr_thres = start_thres_p
    curr_min_thres = 1.      # the threshold is at least 1%
    curr_max_thres = 100.

    for trial_idx in range(trial_time):
        dec_img = demo_percentile(img, curr_thres, wavelet_type=g_wavelet_type)
        mse = np.mean(np.square(np.uint8(img) - dec_img))
        psnr = 10 * np.log10((255.) ** 2 / mse)

        print('###### thres_percentile = {} #####'.format(curr_thres))
        print('mse = {}'.format(mse))
        print('psnr = {}'.format(psnr))

        if psnr>pnr_target:
            curr_min_thres = curr_thres
            curr_thres = (curr_max_thres+curr_min_thres)/2
        else:
            curr_max_thres = curr_thres
            curr_thres = (curr_max_thres + curr_min_thres) / 2





def main():
    ls_thres=[50, 80, 98]
    img_path = './elephant.jpg'
    target_fpath ='./my_compress.gz'

    img= GetAndProcessImg(img_path)
    # img = np.uint8(img)
    img = np.uint8(ScaleImg(img))
    img = np.float32(img)

    PlotImg_NoBlock(img, 'uncompressed')
    uncompressed_fpath = './uncompressed.gz'
    with gzip.open(uncompressed_fpath, 'wb') as f:
        f.write(img)
    uncompressed_fsize = os.path.getsize(uncompressed_fpath)

    print("file size with entropy encoding only: {}".format(uncompressed_fsize))
    for thres in ls_thres:
        dec_img = demo_percentile(img,  thres, flg_write=True, tgt_fpath=target_fpath, wavelet_type=g_wavelet_type)
        PlotImg_NoBlock(dec_img, 'inversed from threshold {}%'.format(thres))

        mse = np.mean(np.square(np.uint8(img) - np.uint8(dec_img)))
        psnr = 10*np.log10((255.)**2/mse)
        compressed_fsize = os.path.getsize(target_fpath)

        comp_ratio = float(uncompressed_fsize)/compressed_fsize
        print ('compression ration={}'.format(comp_ratio))
        print('mse = {}'.format(mse))
        print('psnr = {}'.format(psnr))

    plt.show()

def main2():
    img_path = './elephant.jpg'
    img = GetAndProcessImg(img_path)
    binary_trial(img, trial_time=10)


if __name__ == '__main__':
    main()

