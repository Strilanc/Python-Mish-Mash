# coding=utf-8
"""
Constructs large increment gates and large controlled-not gates out of a linear number of Toffoli-or-smaller gates.
"""

import math
import itertools


def evaluate_circuit(gates, initial_bits=None):
    """
    :param gates: [( [controls], [targets] )]
    :param initial_bits: set(on_bits)

    >>> evaluate_circuit([])
    set([])
    >>> evaluate_circuit([([], [])])
    set([])
    >>> evaluate_circuit([([1], [])])
    set([])
    >>> evaluate_circuit([([], [1])])
    set([1])
    >>> evaluate_circuit([([], [1])], {1})
    set([])

    >>> evaluate_circuit([([1], [2]), ([1, 2], [3, 4])], {1}) == {1, 2, 3, 4}
    True
    """
    bits = set(initial_bits or set())
    for controls, targets in gates:
        if all(c in bits for c in controls):
            for t in targets:
                if t in bits:
                    bits.remove(t)
                else:
                    bits.add(t)
    return bits


def borrowed_cnot_1(gate, borrow):
    controls, targets = gate
    n = len(controls)
    c1 = controls[:(n + 1) // 2]
    c2 = controls[(n + 1) // 2:] + [borrow]
    return [
        (c1, [borrow]),
        (c2, targets),
        (c1, [borrow]),
        (c2, targets)
    ]


def borrowed_cnot_n(controls, target, borrows):
    n = len(controls) - 3
    top = [(controls[:2], [borrows[0]])]
    sweep = [([controls[i + 2], borrows[i]], [borrows[i + 1]]) for i in range(n)]
    tree = flatten([
        reversed(sweep),
        top,
        sweep,
    ])
    bot = [([controls[-1], borrows[n]], [target])]
    return flatten([
        bot,
        tree,
        bot,
        tree,
    ])


def flatten(lists):
    return [e for r in lists for e in r]


def naive_increment(bits):
    return [(bits[:i], [bits[i]]) for i in reversed(range(len(bits)))]


def reduce_cnot(controls, target, borrows):
    if len(controls) <= 2:
        return [(controls, [target])]
    if len(borrows) < len(controls) - 2:
        reduced = borrowed_cnot_1((controls, [target]), borrows[0])
        return flatten([reduce_cnot(c, t[0], list(set(controls + borrows + [target]) - set(c + t)))
                        for c, t in reduced])

    return borrowed_cnot_n(controls, target, borrows)


def share_controls(controls, targets, borrows):
    if len(targets) == 0:
        return []
    if len(targets) == 1:
        return reduce_cnot(controls, targets[0], borrows)
    if len(controls) == 0:
        return [([], targets)]
    if len(controls) <= 2:
        return [(controls, [t]) for t in targets]

    propagate = [([targets[i]], [targets[i + 1]]) for i in range(len(targets) - 1)]
    return flatten([
        reversed(propagate),
        reduce_cnot(controls, targets[0], borrows + targets[1:]),
        propagate
    ])


def borrowed_increment_1(bits, borrows, continue_reduce=False):
    n = (len(bits) + 1) // 2
    top = bits[:n]
    bot = bits[n:]
    bot_inc = reduce_increment([borrows[0]] + bot, top + borrows[1:]) if continue_reduce \
        else naive_increment([borrows[0]] + bot)
    top_inc = reduce_increment(top, borrows + bot) if continue_reduce \
        else naive_increment(top)
    tog = share_controls(top, bot + [borrows[0]], borrows[1:])
    return flatten([
        [
            ([], bot),
        ],
        tog,
        bot_inc,
        [
            ([], [borrows[0]]),
        ],
        tog,
        [
            ([], bot),
        ],
        bot_inc,
        [
            ([], [borrows[0]]),
        ],
        top_inc,
    ])


def interleave(a, b):
    return [e for p in zip(a, b) for e in p]


def borrowed_increment_n(bits, borrows):
    field = interleave(borrows, bits)
    n = len(bits)
    sweep_down = reversed(flatten([
        [
            ([field[i*2+2]], [field[i*2+1]]),
            ([field[i*2+1], field[i*2+2]], [field[i*2]]),
            ([field[i*2+1], field[i*2]], [field[i*2+2]]),
        ] for i in range(n - 1)]))
    sweep_up = flatten([
        [
            ([field[i*2]], [field[i*2+1]]),
            ([field[i*2+1], field[i*2+2]], [field[i*2]]),
            ([field[i*2+1], field[i*2]], [field[i*2+2]]),
        ] for i in range(n - 1)])
    tree = flatten([
        sweep_up,
        [([field[-2]], [field[-1]])],
        sweep_down,
    ])
    carry = borrows[0]
    garbage = borrows[1:]
    carry_toggles = [([carry], [b]) for b in bits[:-1]]
    garbage_toggles = [([], garbage)]

    return flatten([
        [([], [field[-1]])],
        carry_toggles,
        garbage_toggles,
        tree,
        garbage_toggles,
        tree,
        carry_toggles,
    ])


def reduce_increment(bits, borrows):
    if len(bits) <= 2:
        return naive_increment(bits)
    if len(borrows) == 0:
        raise ValueError("Need a borrowed bit")
    if len(borrows) == len(bits) - 1:
        cnot = reduce_cnot(bits[:-1], bits[-1], borrows)
        inc = reduce_increment(bits[:-1], borrows)
        return flatten([cnot, inc])
    if len(borrows) < len(bits):
        return borrowed_increment_1(bits, borrows, continue_reduce=True)
    return borrowed_increment_n(bits, borrows)


def circuit_repr(gates, show_tof=True):
    col_mins = [min(min(t + [float("inf")]), min(c + [float("inf")])) for t, c in gates]
    col_maxes = [max(max(t + [0]), max(c + [0])) for c, t in gates]
    has_controls = [len(c) > 0 for c, _ in gates]
    n = max(col_maxes) + 1
    if any(not set(c).isdisjoint(set(t)) for c, t in gates):
        raise ValueError("Control toggle overlap")
    max_controls = max(len(c) for c, _ in gates)
    max_targets = max(len(t) for c, t in gates if len(c) > 1)

    col_padding = 0
    row_padding = 1
    grid = [["│" if r2 != 0 and col_mins[c] <= r < col_maxes[c] and c2 == 0 and has_controls[c]
             else " " if r2 != 0
             else "─" if c2 != 0
             else "X" if r in gates[c][1]
             else "•" if r in gates[c][0]
             else "┼" if col_mins[c] <= r <= col_maxes[c] and has_controls[c]
             else "─"
             for c in range(len(gates))
             for c2 in range(col_padding + 1)]
            for r in range(n)
            for r2 in (range(row_padding + 1) if r < n - 1 else [0])]
    return '\n'.join(''.join(row) for row in grid) + '\n' +\
           ('Toffoli`d' if max_controls <= 2 and max_targets <= 1 and show_tof else '')


def binary_set(n):
    m = int(math.ceil(math.log(n + 1)/math.log(2))) + 1
    return set(i for i in range(m) if ((1 << i) & n) != 0)


def from_binary_set(n):
    return sum(1 << i for i in n)


def validate_increment(circuit, targets, borrows):
    print circuit_repr(circuit)
    for c in power_set(range(len(targets))):
        for b in power_set(borrows):
            v = from_binary_set(c)
            c2 = binary_set((v + 1) % (1 << len(targets)))
            input = set(flatten([[targets[i] for i in c], b]))
            expected = set(flatten([[targets[i] for i in c2], b]))
            actual = evaluate_circuit(circuit, input)
            if actual != expected:
                print "NO", v, actual - expected, expected - actual, actual, expected, input
    print "done"


def power_set(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))


def validate_cnot(circuit, controls, target, borrows):
    print circuit_repr(circuit)
    for c in power_set(controls):
        for b in power_set(borrows):
            for t in power_set([target]):
                input = set(flatten([c, b, t]))
                toggle = len(controls) == len(c)
                expected = set(flatten([c, b, [target] if bool(t) != toggle else []]))
                actual = evaluate_circuit(circuit, input)
                if actual != expected:
                    print "NO", actual - expected, expected - actual, actual, expected, input
    print "done"


n_bits = 5
increments = range(n_bits)
borrows = [n_bits]
validate_increment(reduce_increment(increments, borrows), increments, borrows)
