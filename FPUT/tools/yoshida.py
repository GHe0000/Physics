import numpy as np
import numba as nb

def _calc_yoshida_coeffs(n):
    coeffs = np.array([0.5])
    for i in range(1, n//2):
        x1 = 1 / (2 - 2**(1/(2*i+1)))
        x0 = 1 - 2 * x1
        coeffs_ = np.concatenate((coeffs, np.flip(coeffs)))
        coeffs_ = np.concatenate((x1 * coeffs_, x0 * coeffs))
        coeffs = coeffs_.copy()
    coeffs = np.concatenate((coeffs, np.flip(coeffs)))
    return coeffs

def yoshida_step(q, p, gradT, gradV, dt, coeffs):
    func_select = np.tile([1,0], len(coeffs))
    for c, fs in zip(coeffs, func_select):
        if fs == 1:
            
    

    pass

if __name__ == '__main__':
    test_coeffs = _calc_yoshida_coeffs(6)
    print(test_coeffs)
    print(np.sum(test_coeffs))
    print(test_coeffs.shape)
    print(np.tile([1,0], len(test_coeffs)))


