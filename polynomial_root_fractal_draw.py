# coding=utf-8
"""
Renders a polar density plot of all of the roots of polynomials up to a certain degree and with the allowed coefficients
limited to a finite set.
"""

import math
import random
import numpy as np
import cmath
import itertools
import cv2


amplitude_resolution = 300
angle_resolution = 1200
amplitude_max = 0
amplitude_min = 1
angle_min = math.pi*1
angle_max = math.pi*0
deg = 20
coefficients = [1, 1j]
print "Upper bound on total roots to be sampled:", (len(coefficients)**deg) * deg


def sample_coefs_random(degree):
    while True:
        yield [cmath.exp(1j * random.random() * 2 * math.pi)
               for _ in range(degree + 1)]


def sample_coefs_exhaustive(allowed, degree):
    return itertools.product(allowed, repeat=degree + 1)


def sample_roots(degree, allowed_coefficients):
    for coefs in sample_coefs_exhaustive(allowed_coefficients, degree):
        for root in np.roots(coefs):
            yield root


def dither(c):
    if isinstance(c, tuple):
        return dither(c[0]), dither(c[1])
    if isinstance(c, complex):
        return dither(c.real), dither(c.imag)
    return math.ceil(c) if random.random() < c % 1 else math.floor(c)


def show_density_plot(buf):
    adjusted = 1 - (np.log(buf + 1)*40)/255
    rgb = cv2.merge([adjusted, adjusted, adjusted])
    cv2.imshow("result", rgb)


def run_sampling_while_previewing(roots):
    sample_densities = np.zeros((amplitude_resolution, angle_resolution), np.float64)
    processed_count = 0
    for r in roots:
        ap = cmath.polar(r)
        pt = (ap[0] - amplitude_min)/(amplitude_max-amplitude_min)*amplitude_resolution,\
             ((ap[1] % (2*math.pi)) - angle_min)/(angle_max - angle_min)*angle_resolution
        px = dither(pt)
        if px[0] < 0 or px[0] >= amplitude_resolution or px[1] < 0 or px[1] >= angle_resolution:
            continue
        sample_densities[px[0], px[1]] += 1

        processed_count += 1
        if processed_count % 1000 == 0:
            print "Sampled:", processed_count
            show_density_plot(sample_densities)
            cv2.waitKey(1)  # otherwise the continuing work hangs the window
    return sample_densities


def run():
    roots = sample_roots(deg, coefficients)
    sample_densities = run_sampling_while_previewing(roots)
    show_density_plot(sample_densities)
    print "done"
    cv2.waitKey()


run()
