import numpy as np
import math
# create log quantizer with
class thres_log_quantizer:
    def __init__(self, num_bit):
        self.num_bit = num_bit
        self.num_bins = 2**(num_bit-1)
        # self.thres_percentile = thres_percentile
        # self.bin_bounds = None

    def _create_sgn_mat(self, in_img):
        return np.where(in_img>0, 1., -1.)

    def _create_bins(self,abs_img, thres_percentile):

        ascend_ord = np.sort(abs_img.flatten())
        percentile_idx = int(thres_percentile/100.*abs_img.size)
        thres = ascend_ord[percentile_idx]+1e-7
        max = ascend_ord[-1]

        return  [thres * ((max / thres) ** (n/(self.num_bins-1.))) for n in range(self.num_bins)]


    def quantize(self, in_img, thres_percentile):
        sgn_mat = self._create_sgn_mat(in_img)
        abs_img = np.abs(in_img)

        bin_maxs = self._create_bins(abs_img, thres_percentile)
        bin_count = np.zeros((len(bin_maxs),1), dtype = np.uint32)
        bin_sum = np.zeros((len(bin_maxs), 1), dtype=np.float32)
        quantized = np.empty(in_img.shape, dtype=np.uint32)
        for p_idx, val in enumerate(abs_img.flatten()):
            bin_choose = -1
            for b_idx, bin_bound in enumerate(bin_maxs):
                if val < bin_bound:
                    bin_choose = b_idx
                    bin_count[b_idx,0] +=1
                    bin_sum[b_idx,0]+=val
                    break

            else:
                bin_choose = len(bin_count)-1
                bin_count[-1,0]+=1
                bin_sum[-1,0]+=val

            quantized[p_idx//in_img.shape[0], p_idx%in_img.shape[0]]=bin_choose
            # handle divide by 0
            bin_count = np.where(bin_count==0,1,bin_count)

        return sgn_mat, quantized, bin_sum/bin_count

    def dequantize(self, sgn_mat, quantized, codebook):
        temp = np.empty(quantized.shape, dtype=np.float32)
        for i in range(quantized.shape[0]):
            for j in range(quantized.shape[1]):
                temp[i,j] = codebook[quantized[i, j]]

        return np.multiply(sgn_mat, temp)


class lloyd_max_quantizer:
    CONVERGE_THRES = 0.03

    def __init__(self, num_bit):
        self.num_bit = num_bit
        self.num_bins = 2**(num_bit-1)
        # self.thres_percentile = thres_percentile
        # self.bin_bounds = None

    def _create_sgn_mat(self, in_img):
        return np.where(in_img>0, 1., -1.)

    def _create_bins(self,abs_img, thres_percentile):
        ascend_ord = np.sort(abs_img.flatten())
        percentile_idx = int(thres_percentile/100.*abs_img.size)
        thres = ascend_ord[percentile_idx]+1e-7
        max = ascend_ord[-1]

        bin_maxs = np.asarray([thres * ((max / thres) ** (n/(self.num_bins-1.))) for n in range(self.num_bins)])[:, np.newaxis]
        bin_reps = self._calc_bin_rep(abs_img, bin_maxs)

        return  bin_maxs, bin_reps

    def _calc_bin_rep(self, abs_img, bin_maxs):
        bin_count = np.zeros((len(bin_maxs), 1), dtype=np.uint32)
        bin_sum = np.zeros((len(bin_maxs), 1), dtype=np.float32)

        for p_idx, val in enumerate(abs_img.flatten()):
            for b_idx, bin_bound in enumerate(bin_maxs):
                if val < bin_bound:
                    bin_count[b_idx,0] +=1
                    bin_sum[b_idx,0]+=val
                    break

            else:
                bin_count[-1,0]+=1
                bin_sum[-1,0]+=val

        bin_rep = bin_sum/bin_count
        for idx, f_isNAN in enumerate(np.isnan(bin_rep)):
            if f_isNAN:
                if idx ==0:
                    bin_rep[idx] = 0

                else:
                    bin_rep[idx] = bin_rep[idx-1]+1e-7

        return bin_rep

    def _calc_bin_max(self, bin_reps, thres,max_val):
        bin_maxs = []

        bin_maxs.append(thres)
        curr_bin_rep = None
        for bin_rep in bin_reps[1:]:
            if not curr_bin_rep is None:
                bin_maxs.append((curr_bin_rep+bin_rep)/2)
            curr_bin_rep=bin_rep

        bin_maxs.append(max_val)

        return np.asarray(bin_maxs)[:,np.newaxis]

    def quantize(self, in_img, thres_percentile):
        sgn_mat = self._create_sgn_mat(in_img)
        abs_img = np.abs(in_img)
        max_val = abs_img.max()+1e-7
        prev_bin_maxs, bin_reps = self._create_bins(abs_img, thres_percentile)
        curr_bin_maxs = self._calc_bin_max(bin_reps, prev_bin_maxs[0,0],max_val)
        diff = prev_bin_maxs-curr_bin_maxs
        cond = 1
        while(cond>lloyd_max_quantizer.CONVERGE_THRES):
            prev_bin_maxs = curr_bin_maxs
            bin_reps = self._calc_bin_rep(abs_img, curr_bin_maxs)
            curr_bin_maxs = self._calc_bin_max(bin_reps,prev_bin_maxs[0,0],max_val)
            diff = prev_bin_maxs - curr_bin_maxs
            cond = np.linalg.norm(diff)

        quantized = np.empty(in_img.shape, dtype=np.uint32)
        for p_idx, val in enumerate(abs_img.flatten()):
            bin_choose = -1
            for b_idx, bin_bound in enumerate(curr_bin_maxs):
                if val < bin_bound:
                    bin_choose = b_idx
                    break

            else:
                bin_choose = len(curr_bin_maxs)-1

            quantized[p_idx//in_img.shape[0], p_idx%in_img.shape[0]]=bin_choose


        return sgn_mat, quantized, bin_reps

    def dequantize(self, sgn_mat, quantized, codebook):
        temp = np.empty(quantized.shape, dtype=np.float32)
        for i in range(quantized.shape[0]):
            for j in range(quantized.shape[1]):
                temp[i,j] = codebook[quantized[i, j]]

        return np.multiply(sgn_mat, temp)