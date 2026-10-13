#!/usr/bin/env python3
"""Exact, reproducible checks for catalogue 3030; not a formal proof.

OpenAI GPT 6.1 Sol (Max reasoning) generated this code on 2026-10-12.
Research directed and overseen by Wentao Li. Uses only Python's standard library.
Run from the repository root with optional --output /absolute/path/results.json.
"""

import argparse
from collections import deque
from itertools import combinations, permutations, product
import json
from math import gcd, prod
from pathlib import Path
import platform
import random


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def identity(size):
    return [[int(i == j) for j in range(size)] for i in range(size)]


def matmul(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)]
            for row in a]


def determinant(a):
    """Bareiss elimination with exact division and row pivoting; det(empty)=1."""
    size = len(a)
    if size == 0:
        return 1
    b = [row[:] for row in a]
    require(all(len(row) == size for row in b), "determinant requires square input")
    previous, sign = 1, 1
    for k in range(size - 1):
        pivot_row = next((i for i in range(k, size) if b[i][k]), None)
        if pivot_row is None:
            return 0
        if pivot_row != k:
            b[k], b[pivot_row] = b[pivot_row], b[k]
            sign = -sign
        pivot = b[k][k]
        for i in range(k + 1, size):
            for j in range(k + 1, size):
                numerator = pivot * b[i][j] - b[i][k] * b[k][j]
                require(numerator % previous == 0, "nonexact Bareiss division")
                b[i][j] = numerator // previous
            b[i][k] = 0
        previous = pivot
    return sign * b[-1][-1]


def determinant_leibniz(a):
    """Independent determinant evaluator for small implementation checks."""
    size = len(a)
    total = 0
    for perm in permutations(range(size)):
        inversions = sum(perm[i] > perm[j]
                         for i in range(size) for j in range(i + 1, size))
        total += (-1) ** inversions * prod(a[i][perm[i]] for i in range(size))
    return total


def adjugate(a):
    size = len(a)
    return [[(-1) ** (i + j) * determinant(
        [[a[r][c] for c in range(size) if c != i]
         for r in range(size) if r != j])
        for j in range(size)] for i in range(size)]


def extended_gcd(a, b):
    old_r, remainder = abs(a), abs(b)
    old_s, s, old_t, t = 1, 0, 0, 1
    while remainder:
        quotient = old_r // remainder
        old_r, remainder = remainder, old_r - quotient * remainder
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t
    return old_r, old_s * (-1 if a < 0 else 1), old_t * (-1 if b < 0 else 1)


def maximal_minors(a):
    return [(columns, determinant([[row[j] for j in columns] for row in a]))
            for columns in combinations(range(len(a[0])), len(a))]


def minor_gcd(minors):
    return gcd(*(value for _, value in minors))


def right_inverse(a, minors):
    """Construct a Bezout/adjugate witness; also return its signed certificate."""
    size, width = len(a), len(a[0])
    divisor, coefficients = 0, []
    for _, value in minors:
        divisor, scale, coefficient = extended_gcd(divisor, value)
        coefficients = [scale * c for c in coefficients] + [coefficient]
        if divisor == 1:
            break
    if divisor != 1:
        return None, []
    witness = [[0] * size for _ in range(width)]
    certificate = []
    for coefficient, (columns, value) in zip(coefficients, minors):
        if coefficient == 0:
            continue
        square = [[row[j] for j in columns] for row in a]
        adj = adjugate(square)
        require(matmul(square, adj) == [[value * int(i == j) for j in range(size)]
                                      for i in range(size)], "adjugate identity")
        for i, column in enumerate(columns):
            for j in range(size):
                witness[column][j] += coefficient * adj[i][j]
        certificate.append({"columns_zero_based": list(columns),
                            "minor": value, "coefficient": coefficient})
    require(sum(t["minor"] * t["coefficient"] for t in certificate) == 1,
            "Bezout certificate")
    require(matmul(a, witness) == identity(size), "constructed MY != I")
    return witness, certificate


def rref_mod(a, prime, width):
    b, pivots = [[value % prime for value in row] for row in a], []
    for column in range(width):
        pivot_row = next((i for i in range(len(pivots), len(b)) if b[i][column]), None)
        if pivot_row is None:
            continue
        k = len(pivots)
        b[k], b[pivot_row] = b[pivot_row], b[k]
        scale = pow(b[k][column], -1, prime)
        b[k] = [value * scale % prime for value in b[k]]
        for i in range(len(b)):
            if i != k:
                factor = b[i][column]
                b[i] = [(x - factor * y) % prime for x, y in zip(b[i], b[k])]
        pivots.append(column)
        if len(pivots) == len(b):
            break
    return b, pivots


def prime_divisors(value):
    divisors, candidate = [], 2
    while candidate * candidate <= value:
        if value % candidate == 0:
            divisors.append(candidate)
            while value % candidate == 0:
                value //= candidate
        candidate += 1
    if value > 1:
        divisors.append(value)
    return divisors


def block_matrix(diagonal, x):
    return [[diagonal[i] * int(i == j) for j in range(len(diagonal))] + row[:]
            for i, row in enumerate(x)]


def prime_local(diagonal, x):
    """Return the criterion and a checked obstruction vector when it fails."""
    size, n = len(diagonal), len(x[0])
    for prime in prime_divisors(prod(diagonal)):
        selected = [i for i, value in enumerate(diagonal) if value % prime == 0]
        rows = [x[i] for i in selected]
        _, pivots = rref_mod(rows, prime, n)
        full_rank = len(rref_mod(block_matrix(diagonal, x), prime, size + n)[1])
        require(full_rank == size - len(selected) + len(pivots), "rank decomposition")
        if len(pivots) < len(selected):
            transposed = [list(col) for col in zip(*rows)]
            reduced, trans_pivots = rref_mod(transposed, prime, len(selected))
            free = next(j for j in range(len(selected)) if j not in trans_pivots)
            local = [int(j == free) for j in range(len(selected))]
            for i, column in enumerate(trans_pivots):
                local[column] = -reduced[i][free] % prime
            vector = [0] * size
            for i, entry in zip(selected, local):
                vector[i] = entry
            require(any(vector), "zero obstruction")
            a = block_matrix(diagonal, x)
            require(all(sum(vector[i] * a[i][j] for i in range(size)) % prime == 0
                        for j in range(size + n)), "invalid obstruction")
            return False, {"prime": prime, "rows_zero_based": selected,
                           "rank": len(pivots), "left_null_vector": vector}
    return True, None


def local_minor_inverse(diagonal, x):
    """Use det D and one selected maximal minor for each prime dividing det D."""
    a, size, n = block_matrix(diagonal, x), len(diagonal), len(x[0])
    chosen = [(tuple(range(size)), prod(diagonal))]
    for prime in prime_divisors(prod(diagonal)):
        selected = [i for i, value in enumerate(diagonal) if value % prime == 0]
        _, pivots = rref_mod([x[i] for i in selected], prime, n)
        require(len(pivots) == len(selected), "local construction needs full rank")
        columns = tuple([i for i in range(size) if i not in selected]
                        + [size + j for j in pivots])
        value = determinant([[row[j] for j in columns] for row in a])
        require(value % prime != 0, "chosen maximal minor is divisible by its prime")
        chosen.append((columns, value))
    witness, certificate = right_inverse(a, chosen)
    require(witness is not None, "prime-selected minors do not produce an inverse")
    return witness, certificate


def finite_quotient_image_size(diagonal, x):
    """Enumerate the subgroup generated by X in product_i Z/d_i Z."""
    zero = (0,) * len(diagonal)
    generators = [tuple(column) for column in zip(*x)]
    seen, queue = {zero}, deque([zero])
    while queue:
        current = queue.popleft()
        for generator in generators:
            following = tuple((a + b) % d for a, b, d
                              in zip(current, generator, diagonal))
            if following not in seen:
                seen.add(following)
                queue.append(following)
    return len(seen)


def rowwise_coprime(diagonal, x):
    return all(gcd(d, *row) == 1 for d, row in zip(diagonal, x))


def check_case(diagonal, x, finite_quotient=False):
    a = block_matrix(diagonal, x)
    minors = maximal_minors(a)
    divisor = minor_gcd(minors)
    success, obstruction = prime_local(diagonal, x)
    require((divisor == 1) == success, "minor and local criteria disagree")
    require(prod(diagonal) % divisor == 0, "minor gcd does not divide det D")
    if finite_quotient:
        reached = finite_quotient_image_size(diagonal, x)
        require(prod(diagonal) % reached == 0, "quotient image size")
        require(divisor == prod(diagonal) // reached, "minor gcd versus quotient index")
    witness, certificate = right_inverse(a, minors)
    require((witness is not None) == success, "construction availability")
    if success:
        require(rowwise_coprime(diagonal, x), "rowwise necessity")
        local_minor_inverse(diagonal, x)
    pairwise = all(gcd(diagonal[i], diagonal[j]) == 1
                   for i in range(len(diagonal)) for j in range(i + 1, len(diagonal)))
    if pairwise:
        require(rowwise_coprime(diagonal, x) == success, "pairwise-coprime corollary")
    return {"diagonal": list(diagonal), "X": x, "minor_gcd": divisor,
            "has_integer_right_inverse": success, "right_inverse": witness,
            "bezout_certificate": certificate, "obstruction": obstruction}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rng = random.Random(3030)
    for _ in range(2000):
        size = rng.randrange(6)
        a = [[rng.randrange(-5, 6) for _ in range(size)] for _ in range(size)]
        if size >= 2 and rng.randrange(4) == 0:
            a[-1] = a[0][:]
        require(determinant(a) == determinant_leibniz(a), "determinant cross-check")
    for _ in range(300):
        size = rng.randrange(1, 5)
        width = size + 3
        a = [[rng.randrange(-4, 5) for _ in range(width)] for _ in range(size)]
        b = [[rng.randrange(-4, 5) for _ in range(size)] for _ in range(width)]
        expanded = sum(value * determinant([b[j] for j in columns])
                       for columns, value in maximal_minors(a))
        require(determinant(matmul(a, b)) == expanded, "Cauchy-Binet cross-check")

    specs = [((d,), n) for d in range(2, 9) for n in range(1, 4)]
    specs += [(d, 2) for d in [(2, 2), (2, 3), (2, 4), (3, 3),
                             (3, 4), (4, 4), (4, 6), (3, 9)]]
    specs += [(d, 3) for d in [(2, 2), (2, 3), (2, 4), (3, 3),
                             (2, 2, 2), (2, 2, 3), (2, 3, 4)]]
    exhaustive = []
    for diagonal, n in specs:
        counts = {"cases": 0, "right_inverses_checked": 0, "obstructions_checked": 0}
        residues = [list(product(range(d), repeat=n)) for d in diagonal]
        for rows in product(*residues):
            result = check_case(diagonal, [list(row) for row in rows], True)
            counts["cases"] += 1
            counts["right_inverses_checked" if result["has_integer_right_inverse"]
                   else "obstructions_checked"] += 1
        exhaustive.append({"diagonal": list(diagonal), "n": n, **counts})

    random_counts = {"cases": 0, "right_inverses_checked": 0,
                     "obstructions_checked": 0, "finite_quotient_checks": 0,
                     "residue_invariance_checks": 0}
    choices = (2, 3, 4, 5, 6, 8, 9, 10, 12, 25, 27, 49)
    for _ in range(500):
        size = rng.randrange(1, 6)
        n = size + rng.randrange(3)
        diagonal = tuple(rng.choice(choices) for _ in range(size))
        x = [[rng.randrange(-8, 9) for _ in range(n)] for _ in range(size)]
        quotient = prod(diagonal) <= 500
        result = check_case(diagonal, x, quotient)
        random_counts["cases"] += 1
        random_counts["right_inverses_checked" if result["has_integer_right_inverse"]
                      else "obstructions_checked"] += 1
        random_counts["finite_quotient_checks"] += int(quotient)
        shifted = [[entry + diagonal[i] * rng.randrange(-5, 6) for entry in row]
                   for i, row in enumerate(x)]
        require(prime_local(diagonal, shifted)[0] == result["has_integer_right_inverse"],
                "row residue invariance")
        require(minor_gcd(maximal_minors(block_matrix(diagonal, shifted)))
                == result["minor_gcd"], "lattice invariance under X -> X + D K")
        random_counts["residue_invariance_checks"] += 1

    named = {
        "rowwise_counterexample": check_case((2, 2), [[1, 0], [1, 0]], True),
        "no_unimodular_minor_but_inverse": check_case((2, 3), [[1, 0], [1, 0]], True),
        "prime_power_diagonal_and_identity": check_case((4, 8, 16), identity(3), True),
        "zero_X": check_case((2, 3), [[0, 0], [0, 0]], True),
    }
    require(rowwise_coprime((2, 2), [[1, 0], [1, 0]]), "counterexample row gcd")
    require(not named["rowwise_counterexample"]["has_integer_right_inverse"],
            "counterexample incorrectly accepted")
    require(all(abs(value) != 1 for _, value in maximal_minors(
        block_matrix((2, 3), [[1, 0], [1, 0]]))), "example has a unimodular minor")
    require(named["no_unimodular_minor_but_inverse"]["has_integer_right_inverse"],
            "nonunimodular-minor example incorrectly rejected")

    report = {"status": "PASS", "python": platform.python_version(),
              "dependencies": "Python standard library only", "seed": 3030,
              "determinant_cross_checks": 2000, "Cauchy_Binet_cross_checks": 300,
              "exhaustive_total": sum(t["cases"] for t in exhaustive),
              "exhaustive_specs": exhaustive, "random": random_counts,
              "named_examples": named,
              "limitations": "Finite exact checks support the handwritten proof; "
                             "not formal verification or independent human review."}
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
