"""Reference implementation of the aggregation engine (Algorithm 1 of the paper).

Input : five dimension scores d1..d5, each an integer 1..5 or None (missing)
Output: (level, baseline_level, override_flag)

Band edges 1.8, 2.6, 3.4 and 4.2 are compared in whole numbers: the test "mean >= 1.8" is
evaluated as 5*S >= 9*n, where S is the sum of the n available scores, so no floating-point
rounding can move a student across a boundary.
"""

def risk_level(scores):
    avail = [s for s in scores if s is not None]
    n = len(avail)
    if n == 0:
        return None, None, False
    S = sum(avail)
    base = 1 + sum(5 * S >= e * n for e in (9, 13, 17, 21))   # e = 5 x band edge
    override = scores[1] == 5 or scores[2] == 5               # D2 attendance or D3 behaviour at the maximum
    return (5 if override else base), base, bool(override and base < 5)


if __name__ == '__main__':
    cases = {
        (2, 2, 3, 1, 2): (2, 2, False),      # mean 2.0
        (1, 2, 2, 2, 2): (2, 2, False),      # mean exactly 1.8 -> level 2
        (3, 3, 2, 3, 2): (3, 3, False),      # mean exactly 2.6 -> level 3
        (2, 5, 1, 2, 2): (5, 2, True),       # attendance override
        (3, 2, 5, 1, 1): (5, 2, True),       # behaviour override
        (4, 5, 4, 4, 4): (5, 5, False),      # level 5 through the mean
        (2, 2, None, 2, 1): (1, 1, False),   # one score missing: mean of four = 1.75
    }
    for k, v in cases.items():
        assert risk_level(list(k)) == v, (k, risk_level(list(k)), v)
    print('risk_engine: all', len(cases), 'worked examples pass')
