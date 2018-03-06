import numpy as np
from numpy import matlib
import math

def GetHaarMatrices(N):
    N = int(N)
    Q = np.matrix([[1,1],[1,-1]])
    M = int(N/2)
    T = np.kron(matlib.eye(M), Q)/np.sqrt(2.)
    P = np.vstack((matlib.eye(N)[::2,:], matlib.eye(N)[1::2,:]))
    return T,P

def GetD4Matrices(N):
    N = int(N)
    D4_coef = [ 0.4829629131445, 0.8365163037378, 0.224143868, -0.129409522552]

    Q1 = np.matrix([[D4_coef[0], D4_coef[1]], [D4_coef[3], -D4_coef[2]]])
    Q2 = np.matrix([[D4_coef[2],D4_coef[3]],[D4_coef[1],-D4_coef[0]]])
    M = int(N / 2)
    T1 = np.kron(matlib.eye(M), Q1)
    T2 = np.kron(matlib.eye(M), Q2)
    T2 = CircularRShift(T2, 2)
    T = (T1+T2)
    P = np.vstack((matlib.eye(N)[::2, :], matlib.eye(N)[1::2, :]))
    return T, P

def GetD6Matrices(N):
    N=int(N)
    D6_coef = [0.33267055295008263, 0.8068915093110925, 0.45987750211849154, -0.13501102001025458, -0.08544127388202666, 0.03522629188570954]
    Q1 = np.matrix([[D6_coef[0], D6_coef[1]], [D6_coef[5], -D6_coef[4]]])
    Q2 = np.matrix([[D6_coef[2], D6_coef[3]], [D6_coef[3], -D6_coef[2]]])
    Q3 = np.matrix([[D6_coef[4], D6_coef[5]], [D6_coef[1], -D6_coef[0]]])
    M = int(N / 2)
    T1 = np.kron(matlib.eye(M), Q1)
    T2 = np.kron(matlib.eye(M), Q2)
    T3 = np.kron(matlib.eye(M), Q3)
    T2 = CircularRShift(T2, 2)
    T3 = CircularRShift(T3, 4)
    T = (T1 + T2 + T3)
    P = np.vstack((matlib.eye(N)[::2, :], matlib.eye(N)[1::2, :]))
    return T, P

# Rightshif the T matrix by n to the right
def CircularRShift(T, n):
    if not T.ndim == 2:
        print ('Error: Matrix is not 2D')
        return None

    P1 = T[:,:-n]
    P2 = T[:,-n:]
    return np.hstack((P2,P1))

def IsPow2(num):
    return (math.log(num,2) - int(math.log(num,2))) == 0

def TransformImg(img, wavelet_type ='D4', flg_one_level = False):
    if not (IsPow2(img.shape[0]) and IsPow2(img.shape[1]) and img.shape[0]==img.shape[1]):
        print ('Error: input image size is not right')
        return None

    size = int(img.shape[0])
    transformed = img.copy()
    while size > 1:
        if wavelet_type == 'D6':
            T, P = GetD6Matrices(size)
        elif wavelet_type == 'D4':
            T, P = GetD4Matrices(size)
        else:
            T, P = GetHaarMatrices(size)
        transformed[:size, :size] = P * T * transformed[:size, :size] * T.T * P.T
        size //= 2
        if flg_one_level:
            break

    return transformed

def InverseTransform(img, wavelet_type ='D4', flg_one_level=False):
    if not (IsPow2(img.shape[0]) and IsPow2(img.shape[1]) and img.shape[0]==img.shape[1]):
        print ('Error: input image size is not right')
        return None

    size = 2
    inversed = img.copy()
    while size < img.shape[0]+1:
        if wavelet_type == 'D6':
            T, P = GetD6Matrices(size)
        elif wavelet_type == 'D4':
            T, P = GetD4Matrices(size)
        else:
            T, P = GetHaarMatrices(size)
        inversed[:size, :size] = T.T * P.T * inversed[:size, :size] *P *T
        size *= 2
        if flg_one_level:
            break

    return inversed

def main():
    print(GetD4Matrices(8))

if __name__ == '__main__':
    main()