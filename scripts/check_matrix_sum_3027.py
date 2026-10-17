#!/usr/bin/env python3
"""Exact, reproducible checks for catalogue 3027, using only Python's stdlib.

The oracle is the Leibniz determinant summed over principal minors. It does
not use Newton identities or the primitive-word product. The checks support
the manuscript's proof; they are not a formal or independent human verification.

Run: python3 scripts/check_matrix_sum_3027.py
Optional: --output /absolute/path/report.json (writes only that requested file).
"""

import argparse
from collections import Counter
from fractions import Fraction
from functools import lru_cache
import hashlib
from itertools import combinations, permutations, product
import json
from math import factorial
from pathlib import Path
import platform
import random


SEED = 3027


class Polynomial:
    """Sparse integral polynomials; monomials are sorted variable-index tuples."""

    def __init__(self, terms):
        self.terms = {m: c for m, c in terms.items() if c}

    @classmethod
    def variable(cls, i):
        return cls({(i,): 1})

    @staticmethod
    def coerce(value):
        return value if isinstance(value, Polynomial) else Polynomial({(): value})

    def __add__(self, other):
        result = dict(self.terms)
        for monomial, coefficient in self.coerce(other).terms.items():
            result[monomial] = result.get(monomial, 0) + coefficient
        return Polynomial(result)

    __radd__ = __add__

    def __neg__(self):
        return Polynomial({m: -c for m, c in self.terms.items()})

    def __sub__(self, other):
        return self + -self.coerce(other)

    def __rsub__(self, other):
        return self.coerce(other) + -self

    def __mul__(self, other):
        result = {}
        for m, c in self.terms.items():
            for n, d in self.coerce(other).terms.items():
                monomial = tuple(sorted(m + n))
                result[monomial] = result.get(monomial, 0) + c * d
        return Polynomial(result)

    __rmul__ = __mul__

    def __eq__(self, other):
        return self.terms == self.coerce(other).terms

    def __bool__(self):
        return bool(self.terms)

    def __repr__(self):
        return f"Polynomial({len(self.terms)} monomials)"


def normalize(value, modulus):
    return value if modulus is None else value % modulus


def determinant(matrix):
    """Leibniz formula, including det(empty) = 1; arbitrary commutative ring."""
    n = len(matrix)
    result = 0
    for permutation in permutations(range(n)):
        inversions = sum(permutation[i] > permutation[j]
                         for i in range(n) for j in range(i + 1, n))
        term = (-1) ** inversions
        for i in range(n):
            term *= matrix[i][permutation[i]]
        result += term
    return result


def elementary(matrix, j, modulus=None):
    n = len(matrix)
    if j == 0:
        return 1
    if j > n:
        return 0
    result = sum(determinant(tuple(tuple(matrix[a][b] for b in indices)
                                   for a in indices))
                 for indices in combinations(range(n), j))
    return normalize(result, modulus)


def matrix_sum(matrices, n, modulus=None):
    return tuple(tuple(normalize(sum(a[i][j] for a in matrices), modulus)
                       for j in range(n)) for i in range(n))


def matrix_multiply(a, b, modulus=None):
    n = len(a)
    return tuple(tuple(normalize(sum(a[i][h] * b[h][j] for h in range(n)),
                                modulus)
                       for j in range(n)) for i in range(n))


def identity(n):
    return tuple(tuple(int(i == j) for j in range(n)) for i in range(n))


def word_matrix(word, matrices, n, modulus=None):
    result = identity(n)
    for letter in word:
        result = matrix_multiply(result, matrices[letter], modulus)
    return result


def trace(matrix):
    return sum(matrix[i][i] for i in range(len(matrix)))


def rotations(word):
    return tuple(word[i:] + word[:i] for i in range(len(word)))


def primitive(word):
    length = len(word)
    return not any(length % d == 0 and word == word[:d] * (length // d)
                   for d in range(1, length))


@lru_cache(maxsize=None)
def necklaces(alphabet_size, max_degree):
    return tuple(word for d in range(1, max_degree + 1)
                 for word in product(range(alphabet_size), repeat=d)
                 if primitive(word) and word == min(rotations(word)))


def partitions(k, maximum=None):
    if k == 0:
        yield ()
        return
    maximum = k if maximum is None else min(k, maximum)
    for first in range(maximum, 0, -1):
        for rest in partitions(k - first, first):
            yield (first,) + rest


def invariant_table(matrices, n, max_degree, modulus=None):
    return {word: tuple(elementary(word_matrix(word, matrices, n, modulus),
                                  j, modulus)
                        for j in range(min(n, max_degree // len(word)) + 1))
            for word in necklaces(len(matrices), max_degree)}


def decorated_partition_formula(invariants, n, k, modulus=None):
    """Enumerate the finite displayed formula, without coefficient extraction."""
    choices = {s: tuple((word, s // len(word)) for word in invariants
                        if s % len(word) == 0 and s // len(word) <= n)
               for s in range(1, k + 1)}
    total = 0
    term_count = 0

    def visit(parts, position, used, previous_word, r_sum, value):
        nonlocal total, term_count
        if position == len(parts):
            total += (-1) ** (k + r_sum) * value
            total = normalize(total, modulus)
            term_count += 1
            return
        for word, r in choices[parts[position]]:
            if word in used:
                continue
            if (position and parts[position] == parts[position - 1]
                    and not previous_word < word):
                continue
            visit(parts, position + 1, used | {word}, word, r_sum + r,
                  normalize(value * invariants[word][r], modulus))

    for parts in partitions(k):
        visit(parts, 0, set(), (), 0, 1)
    return total, term_count


def primitive_product_coefficients(invariants, max_degree, modulus=None):
    result = [1] + [0] * max_degree
    for word, values in invariants.items():
        new = [0] * (max_degree + 1)
        for old_degree, coefficient in enumerate(result):
            for r, value in enumerate(values):
                degree = old_degree + r * len(word)
                if degree <= max_degree:
                    new[degree] += coefficient * (-1) ** r * value
                    new[degree] = normalize(new[degree], modulus)
        result = new
    return result


def trace_partition_formula(matrices, n, k):
    """Newton--Girard closed formula over Q, expanding every ordered word."""
    powers = {r: sum(trace(word_matrix(word, matrices, n))
                     for word in product(range(len(matrices)), repeat=r))
              for r in range(1, k + 1)}
    answer = Fraction(0)
    for parts in partitions(k):
        denominator = 1
        numerator = (-1) ** (k - len(parts))
        for r, multiplicity in Counter(parts).items():
            denominator *= r ** multiplicity * factorial(multiplicity)
            numerator *= powers[r] ** multiplicity
        answer += Fraction(numerator, denominator)
    return answer


def mobius(number):
    sign = 1
    prime = 2
    while prime * prime <= number:
        if number % prime == 0:
            number //= prime
            sign *= -1
            if number % prime == 0:
                return 0
        prime += 1
    return -sign if number > 1 else sign


def run_checks():
    rng = random.Random(SEED)
    counts = Counter()
    noncommuting = Counter()

    def verify(matrices, n, label, modulus=None, degree=None, trace_check=False):
        degree = n if degree is None else degree
        invariants = invariant_table(matrices, n, degree, modulus)
        coefficients = primitive_product_coefficients(invariants, degree, modulus)
        oracle_matrix = matrix_sum(matrices, n, modulus)
        for k in range(degree + 1):
            oracle = elementary(oracle_matrix, k, modulus)
            actual, number_of_terms = decorated_partition_formula(
                invariants, n, k, modulus)
            if actual != oracle or normalize((-1) ** k * coefficients[k],
                                             modulus) != oracle:
                raise AssertionError((label, n, len(matrices), k, modulus,
                                      actual, oracle, coefficients[k]))
            if k <= n and number_of_terms != len(matrices) ** k:
                raise AssertionError(("decoration multiplicity", label, k,
                                      number_of_terms, len(matrices) ** k))
            if trace_check and trace_partition_formula(matrices, n, k) != oracle:
                raise AssertionError(("trace partition", label, k))
            counts[label + "_coefficient_checks"] += 1
        counts[label + "_matrix_tuples"] += 1
        if any(matrix_multiply(a, b, modulus) != matrix_multiply(b, a, modulus)
               for a, b in combinations(matrices, 2)):
            noncommuting[label] += 1

    # All alphabet-size and empty-sum boundaries up to n = 6, over Z.
    for n in range(7):
        for alphabet_size in range(n + 1):
            matrices = tuple(tuple(tuple(rng.randrange(-3, 4) for _ in range(n))
                                   for _ in range(n)) for _ in range(alphabet_size))
            verify(matrices, n, "integer", trace_check=True)

    # Finite fields include characteristics <= k; direct modular arithmetic.
    for p in (2, 3, 5, 7):
        for n in range(1, 6):
            for alphabet_size in range(n + 1):
                for _ in range(2):
                    matrices = tuple(tuple(tuple(rng.randrange(p) for _ in range(n))
                                           for _ in range(n))
                                     for _ in range(alphabet_size))
                    verify(matrices, n, f"F{p}", p)

    # Every ordered pair of 2-by-2 matrices over F_2, also above matrix degree.
    binary_matrices = tuple((entries[:2], entries[2:])
                            for entries in product(range(2), repeat=4))
    for a, b in product(binary_matrices, repeat=2):
        verify((a, b), 2, "exhaustive_F2_2x2_pairs", 2, degree=4)

    # Ring specialization is stronger than the target; include zero divisors.
    for modulus in (4, 6, 8):
        for _ in range(5):
            matrices = tuple(tuple(tuple(rng.randrange(modulus) for _ in range(3))
                                   for _ in range(3)) for _ in range(3))
            verify(matrices, 3, f"Zmod{modulus}", modulus, degree=5)

    for n in range(1, 5):
        matrices = tuple(tuple(tuple(Fraction(rng.randrange(-3, 4),
                                               rng.randrange(1, 5))
                                     for _ in range(n)) for _ in range(n))
                         for _ in range(n))
        verify(matrices, n, "rational", trace_check=True)

    # Generic integral polynomial checks with 8 and 18 independent variables.
    for n in (2, 3):
        matrices = tuple(tuple(tuple(Polynomial.variable(a * n * n + i * n + j)
                                     for j in range(n)) for i in range(n))
                         for a in range(2))
        verify(matrices, n, f"generic_{n}x{n}_pair", degree=n + 1)

    # Primitive counts and periodic-word orbit multiplicities are separate checks.
    for alphabet_size in range(7):
        words = necklaces(alphabet_size, 6)
        for d in range(1, 7):
            numerator = sum(mobius(e) * alphabet_size ** (d // e)
                            for e in range(1, d + 1) if d % e == 0)
            expected = numerator // d
            if numerator % d or sum(len(w) == d for w in words) != expected:
                raise AssertionError(("primitive count", alphabet_size, d))
            counts["primitive_count_checks"] += 1

    matrices = tuple(tuple(tuple(rng.randrange(-2, 3) for _ in range(3))
                           for _ in range(3)) for _ in range(3))
    total_matrix = matrix_sum(matrices, 3)
    for m in range(1, 7):
        left = trace(word_matrix((0,) * m, (total_matrix,), 3))
        right = sum(len(w) * trace(word_matrix(w * (m // len(w)), matrices, 3))
                    for w in necklaces(3, m) if m % len(w) == 0)
        if left != right:
            raise AssertionError(("periodic trace multiplicity", m, left, right))
        counts["periodic_orbit_checks"] += 1
    for word in necklaces(3, 4):
        original = word_matrix(word, matrices, 3)
        for rotation in rotations(word):
            rotated = word_matrix(rotation, matrices, 3)
            for j in range(4):
                if elementary(original, j) != elementary(rotated, j):
                    raise AssertionError(("cyclic invariance", word, rotation, j))
                counts["cyclic_invariance_checks"] += 1

    # Explicit noncommuting examples used in the manuscript.
    a = ((0, 1), (0, 0))
    b = ((0, 0), (1, 0))
    verify((a, b), 2, "matrix_unit_pair", degree=4, trace_check=True)
    verify((a, b), 2, "matrix_unit_pair_F2", 2, degree=4)
    if matrix_multiply(a, b) == matrix_multiply(b, a):
        raise AssertionError("Pair must not commute")
    if elementary(matrix_sum((a, b), 2), 2) != -1:
        raise AssertionError("Wrong sign in matrix-unit example")
    cycle = (((0, 1, 0), (0, 0, 0), (0, 0, 0)),
             ((0, 0, 0), (0, 0, 1), (0, 0, 0)),
             ((0, 0, 0), (0, 0, 0), (1, 0, 0)))
    verify(cycle, 3, "oriented_three_cycle", degree=5, trace_check=True)
    for p in (2, 3, 5, 7):
        verify(cycle, 3, "oriented_three_cycle_Fp", p, degree=5)
    orientation = [trace(word_matrix(w, cycle, 3)) for w in ((0, 1, 2), (0, 2, 1))]
    if orientation != [1, 0] or elementary(matrix_sum(cycle, 3), 3) != 1:
        raise AssertionError("Reversal must not be identified with cyclic rotation")
    if any(trace(word_matrix((0,) * r, (identity(2),), 2, 2)) % 2
           for r in range(1, 9)) or elementary(identity(2), 2, 2) != 1:
        raise AssertionError("Characteristic-two trace obstruction")
    counts["explicit_trace_obstruction_checks"] += 1

    return {
        "status": "PASS",
        "problem_id": "3027",
        "seed": SEED,
        "python_version": platform.python_version(),
        "dependencies": "Python standard library only",
        "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "counts": dict(sorted(counts.items())),
        "noncommuting_tuples": dict(sorted(noncommuting.items())),
        "examples": {"S2_E12_plus_E21_over_Z": -1,
                     "S2_E12_plus_E21_over_F2": 1,
                     "tr_E12_E23_E31": orientation[0],
                     "tr_E12_E31_E23": orientation[1],
                     "S3_oriented_cycle": 1,
                     "F2_I2_all_positive_power_traces": 0,
                     "F2_I2_S2": 1},
        "limits": "Finite exact checks and small generic polynomial identities; "
                  "not a formal proof or independent human verification.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional exact report path")
    args = parser.parse_args()
    report = run_checks()
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
