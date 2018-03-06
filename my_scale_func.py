import numpy as np
from scipy.ndimage.interpolation import shift
import matplotlib.pyplot as plt


sqrt2= 1.414213562

class phi_func:
    def __init__(self, list_s=None, j=0):
        if list_s is None:
            self.list_s = np.asarray([1.],dtype=np.float32)
        else:
            self.list_s = np.asarray(list_s,dtype=np.float32)

        self.scale = j


    def get_next_phi(self, list_h):
        num_h = len(list_h)
        next_scale = self.scale+1

        ret = np.zeros((num_h*2**(next_scale)), np.float32)
        samples = np.zeros((num_h*2**(next_scale)), np.float32)
        samples[:self.list_s.shape[0]] = self.list_s
        shift_unit = (2**self.scale)
        for idx, h in enumerate(list_h):
            a= h*shift(samples, idx*shift_unit, cval=0.)
            ret+=a

        ret *= sqrt2

        return phi_func(ret, next_scale)

    def get_psi(self, list_h):
        num_h = len(list_h)
        next_scale = self.scale + 1
        list_g = list(list_h)
        list_g.reverse()
        ret = np.zeros((num_h * 2 ** (next_scale)), np.float32)
        samples = np.zeros((num_h * 2 ** (next_scale)), np.float32)
        samples[:self.list_s.shape[0]] = self.list_s
        shift_unit = (2 ** (self.scale))
        for idx, h in enumerate(list_g):
            a =(-1**idx)* h * shift(samples, idx * shift_unit, cval=0.)
            ret += a

        ret *= sqrt2
        return phi_func(ret, self.scale)


    def plot(self, num_h = 4):
        x_axis_intervals = np.linspace(0,4,num=len(self.list_s)+1)
        x_axis_val = [(x_axis_intervals[i] + x_axis_intervals[i+1])/2 for i in range(len(x_axis_intervals)-1)]
        plt.plot(x_axis_val, self.list_s)
        # plt.axis([0, num_h-1, 0, 20])

class d4_func:
    def __init__(self, list_s=None, j=0, endpt = 1):
        if list_s is None:
            self.list_s = np.asarray([1.],dtype=np.float32)
        else:
            self.list_s = np.asarray(list_s,dtype=np.float32)

        self.scale = j
        self.endpt = endpt



    def get_next_phi(self, list_h):

        next_scale = self.scale+1
        next_endpt = 1.+(1-0.5**next_scale)*2

        ret = np.zeros(int(next_endpt/(2**(-1*next_scale))), np.float32)
        samples = np.zeros(int(next_endpt/(2**(-1*next_scale))), np.float32)
        samples[:self.list_s.shape[0]] = self.list_s
        shift_unit = (2**self.scale)
        for idx, h in enumerate(list_h):
            ret+=float(h)*shift(samples, idx*shift_unit, cval=0.)

        ret *= sqrt2

        return d4_func(ret, next_scale, next_endpt)

    def get_psi(self, list_h):

        next_scale = self.scale + 1
        next_endpt = 1. + (1 - 0.5 ** next_scale) * 2
        list_g = [list_h[3],-list_h[2],list_h[1],-list_h[0]]
        ret = np.zeros(int(next_endpt / (2 ** (-1 * next_scale))), np.float32)
        samples = np.zeros(int(next_endpt / (2 ** (-1 * next_scale))), np.float32)
        samples[:self.list_s.shape[0]] = self.list_s
        shift_unit = (2 ** self.scale)
        for idx, g in enumerate(list_g):
            ret += (-1**(idx))* float(g) * shift(samples, idx * shift_unit, cval=0.)

        ret *= sqrt2
        return d4_func(ret, next_scale, next_endpt)


    def plot(self, num_h = 4):
        x_axis_intervals = np.linspace(0,self.endpt,num=len(self.list_s)+1)
        x_axis_val = [(x_axis_intervals[i] + x_axis_intervals[i+1])/2 for i in range(len(x_axis_intervals)-1)]
        plt.plot(x_axis_val, self.list_s)

class dx_func:
    def __init__(self, x = 4, list_s=None, j=0, endpt = 1.):
        if list_s is None:
            self.list_s = np.asarray([1.],dtype=np.float32)
        else:
            self.list_s = np.asarray(list_s,dtype=np.float32)

        self.x = x
        self.scale = j
        self.endpt = endpt



    def get_next_phi(self, list_h):

        next_scale = self.scale+1
        next_endpt = self.endpt/2+0.5*(self.x-1)

        ret = np.zeros(int(next_endpt/(2**(-1*next_scale))), np.float32)
        samples = np.zeros(int(next_endpt/(2**(-1*next_scale))), np.float32)
        samples[:self.list_s.shape[0]] = self.list_s
        shift_unit = (2**self.scale)
        for idx, h in enumerate(list_h):
            ret+=float(h)*shift(samples, idx*shift_unit, cval=0.)

        ret *= sqrt2

        return dx_func(self.x, ret, next_scale, next_endpt)

    def get_psi(self, list_h):

        next_scale = self.scale + 1
        next_endpt = self.endpt/2+0.5*(self.x-1)
        list_g = [list_h[3],-list_h[2],list_h[1],-list_h[0]]
        ret = np.zeros(int(next_endpt / (2 ** (-1 * next_scale))), np.float32)
        samples = np.zeros(int(next_endpt / (2 ** (-1 * next_scale))), np.float32)
        samples[:self.list_s.shape[0]] = self.list_s
        shift_unit = (2 ** self.scale)
        for idx, g in enumerate(list_g):
            ret += (-1**(idx))* float(g) * shift(samples, idx * shift_unit, cval=0.)

        ret *= sqrt2
        return dx_func(self.x, ret, next_scale, next_endpt)


    def plot(self):
        x_axis_intervals = np.linspace(0,self.endpt,num=len(self.list_s)+1)
        x_axis_val = [(x_axis_intervals[i] + x_axis_intervals[i+1])/2 for i in range(len(x_axis_intervals)-1)]
        plt.plot(x_axis_val, self.list_s)


def test():
    list_h = [0.15774243,0.69950381,1.06226376,0.44583132,-0.3199866,-0.18351806,0.13788809,0.03892321,-0.04466375,0.0007832512,0.0067560624,-0.0015235338]
    # list_h = [0.47046721, 1.14111692, 0.650365, -0.19093442, -0.12083221, 0.0498175]
    # list_h = [0.33267055295008263, 0.8068915093110925, 0.45987750211849154, -0.13501102001025458, -0.08544127388202666, 0.03522629188570954]
    p = dx_func(x=len(list_h))
    for _ in range(10):
        p = p.get_next_phi(list_h)

    s = p.get_psi(list_h)
    p.plot()
    s.plot()
    #
    plt.show()

if __name__ == '__main__':
    test()