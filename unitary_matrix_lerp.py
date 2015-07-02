# coding=utf-8
"""
Interpolating between 2x2 unitary matrices.
"""

import numpy as np
import math


def unitary_breakdown(m):
    """
    Breaks a 2x2 unitary matrix into quaternion and phase components.
    """
    # Extract rotation components
    a, b, c, d = m[0, 0], m[0, 1], m[1, 0], m[1, 1]
    t = (a + d)/2j
    x = (b + c)/2
    y = (b - c)/-2j
    z = (a - d)/2

    # Extract common phase factor
    p = max([t, x, y, z], key=lambda e: abs(e))
    p /= abs(p)
    pt, px, py, pz = t/p, x/p, y/p, z/p

    q = [pt.real, px.real, py.real, pz.real]
    return q, p


def sin_scale_ratio(theta, factor):
    """
    Returns sin(theta * factor) / sin(theta), with care around the origin to avoid dividing by zero.
    """
    # Near zero, switch to a Taylor series based approximation to avoid floating point error blowup.
    if abs(theta) < 0.0001:
        d = theta * theta / 6
        return factor * (1 - d * factor * factor) / (1 - d)
    return math.sin(theta * factor) / math.sin(theta)


def unitary_lerp(u1, u2, t):
    """
    Interpolates between two 2x2 unitary numpy matrices.
    """
    # Split into rotation and phase parts
    q1, p1 = unitary_breakdown(u1)
    q2, p2 = unitary_breakdown(u2)

    # Spherical interpolation of rotation
    dot = sum(v1*v2 for v1,v2 in zip(q1, q2))
    if dot < 0:
        # Don't go the long way around...
        q2 *= -1
        p2 *= -1
        dot *= -1
    theta = math.acos(min(dot, 1))
    c1 = sin_scale_ratio(theta, 1-t)
    c2 = sin_scale_ratio(theta, t)
    u3 = (u1 * c1 / p1 + u2 * c2 / p2)

    # Angular interpolation of phase
    a1 = np.angle(p1)
    a2 = np.angle(p2)
    da = (a2 - a1 + math.pi) % (math.pi * 2) - math.pi  # smallest signed angle distance (mod 2pi)
    a3 = a1 + da * t
    p3 = math.cos(a3) + 1j * math.sin(a3)
    return u3 * p3


print unitary_lerp(np.mat([[1, 1], [1, -1]])/math.sqrt(2), np.mat([[1j, 1], [1, 1j]])/math.sqrt(2), 0.25)