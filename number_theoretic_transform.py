# coding=utf-8
"""
Computing convolutions with generalized fourier transforms.
"""

import math
import cmath
import numpy as np


def ceil_pow_2(n):
    """
    Returns the smallest power of 2 at least as large as the given n.

    >>> ceil_pow_2(7)
    8
    >>> ceil_pow_2(8)
    8
    >>> ceil_pow_2(9)
    16
    """
    return 1 << int(math.ceil(math.log(n, 2)))


def multiplicative_inverse(a, n):
    """
    Returns the multiplicative inverse of a modulo n.

    >>> multiplicative_inverse(3, 11)
    4
    """
    t, t2 = 0, 1
    r, r2 = n, a
    while r2 != 0:
        q = r // r2
        t, t2 = t2, t - q * t2
        r, r2 = r2, r - q * r2
    if r > 1:
        raise ValueError("not invertible")
    return t % n


def check_principal_root(p, g, n):
    """
    Checks if the given g is an n'th principal root of unity, for the integers modulo p.
    """
    if pow(g, n, p) != 1:
        raise ValueError("Bad pow")
    for k in range(1, n):
        if pow(g, k, p) == 1:
            raise ValueError("Pre pow")
        if sum(pow(g, i*k, p) for i in range(n)) % p != 0:
            raise ValueError("Bad sum pow")


def convolution_ref(x, y):
    """
    Returns the cyclic convolution of x and y, computed the slow-but-definitely-correct way.

    >>> convolution_ref([1], [1])
    [1]
    >>> convolution_ref([2], [-3])
    [-6]
    >>> convolution_ref([2, 3, 5], [7, 11, 13])
    [108, 108, 94]
    """
    if len(x) != len(y):
        raise ValueError("Different lengths")
    n = len(x)
    return [sum(x[i]*y[(j-i) % n] for i in range(n)) for j in range(n)]


def convolution_fft(x, y):
    """
    Returns the convolution of two vectors, computed via the convolution theorem using a fast fourier transform and a
    point-wise product.

    >>> convolution_fft([2, 3, 5, 7], [11, 13, 17, 19])
    [255, 273, 261, 231]
    """
    if len(x) != len(y):
        raise ValueError("Different lengths")
    a, b = cooley_tukey_fft(x), cooley_tukey_fft(y)
    c = cooley_tukey_ifft(zip_dot(a, b))
    return [int(round(abs(e))) for e in c]


def zip_dot(x, y):
    """
    Computes the point-wise product of two vectors.
    """
    if len(x) != len(y):
        raise ValueError("Different lengths")
    return [a * b for a, b in zip(x, y)]


def prime_rev(x, normalize):
    return [normalize(-e) for e in [x[0]] + x[-1:0:-1]]


def different_lists(x, y):
    """
    Determines if two vectors differ.
    """
    return not np.all(np.array(x) == y)


def cooley_tukey(x, root, normalize):
    """
    The Cooley-Tukey FFT algorithm, generalized so it can also be used for number-theoretic transforms.
    """
    if len(x) <= 1:
        return [normalize(e) for e in x]
    if len(x) % 2 != 0:
        raise ValueError("Not a multiple of 2")
    n = len(x)

    u = cooley_tukey(x[0::2], root, normalize)
    v = cooley_tukey(x[1::2], root, normalize)

    y = [x[i]*root(i, 2) for i in range(n)]
    # For the FFT w and t would be just u and -v respectively.
    # For other cases that's not true... and unfortunately these extra recursions make the algorithm O(n^2).
    # It should be possible to get back down to O(n lg^2 n) due to the recursions overlapping (i.e. dynamic programming
    # would help), but ideally I want to figure out how to avoid them altogether. For example, the results are *almost*
    # a rotation of u and v, except that the rotation has a longer period.
    w = cooley_tukey(y[0::2], root, normalize)
    t = cooley_tukey(y[1::2], root, normalize)

    for k in range(n/2):
        f = root(k, n)
        u[k] = u[k] + v[k]*f
        v[k] = w[k] + t[k]*f
    return [normalize(e) for e in u + v]


def cooley_tukey_prime(x, p, g):
    """
    Computes the number-theoretic transform of x using field of integers modulo p and the (p-1)'th principal root g.
    Zero-pads the vector up to a multiple of 2, so Cooley-Tukey can be used.

    >>> cooley_tukey_prime([2, 3, 1, 3], 5, 2)
    [4, 1, 2, 1]
    >>> cooley_tukey_prime([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 11, 2)
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    >>> cooley_tukey_prime([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 11, 2)
    [1, 2, 4, 8, 5, 10, 9, 7, 3, 6]
    >>> cooley_tukey_prime([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 11, 2)
    [1, 4, 5, 9, 3, 1, 4, 5, 9, 3]
    >>> cooley_tukey_prime([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 11, 2)
    [1, 8, 9, 6, 4, 10, 3, 2, 5, 7]
    >>> cooley_tukey_prime([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 11, 2)
    [1, 5, 3, 4, 9, 1, 5, 3, 4, 9]
    >>> cooley_tukey_prime([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 11, 2)
    [1, 10, 1, 10, 1, 10, 1, 10, 1, 10]
    """
    m = ceil_pow_2(min(len(x), p))
    x = x + [0]*(m - len(x))
    y = cooley_tukey(
        x,
        lambda n, d: pow(g, len(x)*n/d, p),
        lambda v: int(v % p))
    return y[:p-1]


def cooley_tukey_prime_i(x, p, g):
    """
    Computes the inverse number-theoretic transform of x using field of integers modulo p and the (p-1)'th principal
    root g.

    >>> cooley_tukey_prime_i([4, 1, 2, 1], 5, 2)
    [2, 3, 1, 3]
    >>> cooley_tukey_prime_i([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 11, 2)
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    >>> cooley_tukey_prime_i([1, 2, 4, 8, 5, 10, 9, 7, 3, 6], 11, 2)
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    >>> cooley_tukey_prime_i([1, 4, 5, 9, 3, 1, 4, 5, 9, 3], 11, 2)
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    >>> cooley_tukey_prime_i([1, 8, 9, 6, 4, 10, 3, 2, 5, 7], 11, 2)
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    >>> cooley_tukey_prime_i([1, 5, 3, 4, 9, 1, 5, 3, 4, 9], 11, 2)
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    >>> cooley_tukey_prime_i([1, 10, 1, 10, 1, 10, 1, 10, 1, 10], 11, 2)
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    """
    return prime_rev(cooley_tukey_prime(x, p, g), lambda v: v % p)


def cooley_tukey_fft(x):
    """
    Computes the fourier transform of x.

    >>> np.all(abs(cooley_tukey_fft([2, 3, 5, 7]) - np.array([17, -3 + 4j, -3, -3 - 4j])) <= 0.000001)
    True
    >>> np.all(abs(cooley_tukey_fft([2, -3, 5, 7, 11+13j, 17j, 19, 23]) - \
            np.array([64+30j, -11.828427+2.414214j, 6+46j, -30.213203-1.544156j, 10-4j, -6.171573-0.414214j, -28-20j, \
                      12.213203-52.455844j])) <= 0.000001)
    True
    """
    return cooley_tukey(x, lambda n, d: cmath.exp(-2j*math.pi*n/d), lambda v: v)


def cooley_tukey_ifft(x):
    """
    Computes the inverse fourier transform of x.
    """

    return [e/len(x) for e in cooley_tukey(x, lambda n, d: cmath.exp(2j*math.pi*n/d), lambda v: v)]


def prime_ref(x, p, g):
    """
    Computes the number-theoretic transform of x using field of integers modulo p and the (p-1)'th principal
    root g. Done the slow-but-obviously-correct way, for comparison when verifying other methods.
    """
    n = len(x)
    return [sum(pow(g, i*j, p) * x[i] for i in range(n)) % p for j in range(n)]


def prime_ref_i(x, p, g):
    """
    Computes the inverse number-theoretic transform of x using field of integers modulo p and the (p-1)'th principal
    root g. Done the slow-but-obviously-correct way, for comparison when verifying other methods.
    """
    n = len(x)
    return [-sum(pow(g, -i*j % (p-1), p) * x[i] for i in range(n)) % p for j in range(n)]


def test_run():
    p=29
    g=2
    # check_generator(p, g)
    # for i in range(p):
    #     v = [1] * i + [2] + [0]*(p-i-2)
    #     print
    #     print i, v
    #     # print "rf", prime_ref(v, p, g)
    #     # print "ct", cooley_tukey_prime(v, p, g)
    #     # print "un", cooley_tukey_prime_i(cooley_tukey_prime(v, p, g), p, g)
    #     xx = [-e % p for e in cooley_tukey_prime(cooley_tukey_prime(v, p, g), p, g)]
    #     xx = [xx[0]] + list(reversed(xx[1:]))
    #     print "xx", xx
    #     print "conv", cooley_tukey_prime_i([e*e % p for e in cooley_tukey_prime(v, p, g)], p, g)
    #     print "conv", cooley_tukey_prime([-e*e % p for e in cooley_tukey_prime_i(v, p, g)], p, g)

    # print prime_ref([pow(2, i, 101) for i in range(64)], 101, 2)
    # print prime_ref([pow(2, i, 101) for i in range(128)], 101, 2)
    # print prime_ref([pow(4, i, 101) for i in range(128)], 101, 2)
    # print prime_ref([pow(8, i, 101) for i in range(128)], 101, 2)
    # print prime_ref([pow(16, i, 101) for i in range(128)], 101, 2)


    n = 8
    fw1 = [0 if i >= 8 else (i*i+2)%p for i in range(n)]

    for j in range(n):
        fwj = [pow(g, i*j, p) for i in range(n)]
        print j, prime_ref(zip_dot(fw1, fwj), p, g)
    # fv1 = cooley_tukey_prime(fw1, p, g)
    # fv2 = cooley_tukey_prime(fw2, p, g)
    # print [-e%p for e in convolution_ref(fv1, fv2)]

test_run()
