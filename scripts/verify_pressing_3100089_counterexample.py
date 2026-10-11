#!/usr/bin/env python3
"""Verify the six-vertex counterexample to FOUR TOTAL insertion/deletion edits.

OpenAI GPT 6.1 Sol, Max reasoning, 11 October 2026.
The manuscript gives a self-contained proof; this is supporting computation.
No independent human or formal proof verification is claimed.

Vertices 1..5 form a cycle, 6 is pendant at 2; initially only 4 and 5 are black.
Enumerate the complete legal pressing tree directly with edge/color sets.
Cross-check all 1,957 distinct-symbol words using dense principal determinants.
Check insertion/deletion distance both by dynamic programming and subsequences.
Standard library only. Run without -O, because assertions are verification checks.
"""
import argparse
import hashlib
import itertools
import json
import platform
from pathlib import Path

VERTICES = tuple(range(1, 7))
INITIAL_EDGES = frozenset(frozenset(e) for e in ((1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (2, 6)))
INITIAL_BLACK = frozenset((4, 5))
EXPECTED_SUCCESS = ((4, 3, 2, 6, 5, 1), (5, 1, 2, 6, 4, 3))
EXPECTED_STALLED = ((4, 3, 2, 1), (4, 3, 5, 1), (5, 1, 2, 3), (5, 1, 4, 3))


def press_graph(edges, black, vertex):
    if vertex not in black:
        raise ValueError('Pressing a white vertex is illegal')
    closed = {vertex}
    for edge in edges:
        if vertex in edge:
            closed.update(edge)
    toggled = {frozenset(pair) for pair in itertools.combinations(closed, 2)}
    return edges ^ toggled, black ^ closed


def complete_tree():
    states = {}
    successful = []
    stalled = []

    def visit(edges, black, prefix):
        assert len(prefix) == len(set(prefix))
        assert all(not any(v in edge for edge in edges) and v not in black for v in prefix)
        states[prefix] = (edges, black)
        if not black:
            (stalled if edges else successful).append(prefix)
            return
        for v in sorted(black):
            e, b = press_graph(edges, black, v)
            visit(e, b, prefix + (v,))

    visit(INITIAL_EDGES, INITIAL_BLACK, ())
    assert tuple(sorted(successful)) == EXPECTED_SUCCESS
    assert tuple(sorted(stalled)) == EXPECTED_STALLED
    assert len(states) == 19
    return states, successful, stalled


def determinant(matrix):
    """Dense Gaussian elimination, no pressing update or atlas code."""
    a = [row[:] for row in matrix]
    for c in range(len(a)):
        pivot = next((r for r in range(c, len(a)) if a[r][c]), None)
        if pivot is None:
            return 0
        a[c], a[pivot] = a[pivot], a[c]
        for r in range(c + 1, len(a)):
            if a[r][c]:
                a[r] = [x ^ y for x, y in zip(a[r], a[c])]
    return 1


def principal_flag_check(states):
    matrix = [[int((u == v and u in INITIAL_BLACK) or frozenset((u, v)) in INITIAL_EDGES)
               for v in VERTICES] for u in VERTICES]
    assert determinant(matrix) == 1
    feasible = {}
    for k in range(7):
        for vertices in itertools.combinations(VERTICES, k):
            feasible[frozenset(vertices)] = determinant([[matrix[u - 1][v - 1] for v in vertices]
                                                        for u in vertices])
    legal_flags = set()
    distinct_words = 0
    for k in range(7):
        for word in itertools.permutations(VERTICES, k):
            distinct_words += 1
            if all(feasible[frozenset(word[:j])] for j in range(1, k + 1)):
                legal_flags.add(word)
    assert distinct_words == 1957
    assert legal_flags == set(states)
    assert {w for w in legal_flags if len(w) == 6} == set(EXPECTED_SUCCESS)
    return matrix, distinct_words, len(feasible)


def insertion_deletion_distance(a, b):
    """Direct cost DP. A substitution is not an allowed unit-cost move."""
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            value = min(previous[j] + 1, current[j - 1] + 1)
            if x == y:
                value = min(value, previous[j - 1])
            current.append(value)
        previous = current
    return previous[-1]


def subsequences(word):
    return {tuple(word[i] for i in positions)
            for k in range(len(word) + 1)
            for positions in itertools.combinations(range(len(word)), k)}


def main():
    if not __debug__:
        raise RuntimeError('Verification requires Python assertions; do not use -O')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    states, successful, stalled = complete_tree()
    matrix, distinct_words, subsets = principal_flag_check(states)
    first, second = EXPECTED_SUCCESS
    common = subsequences(first) & subsequences(second)
    lcs = max(map(len, common))
    distance = insertion_deletion_distance(first, second)
    assert lcs == 2 and distance == 8
    assert sorted(w for w in common if len(w) == 2) == [(2, 6), (4, 3), (5, 1)]
    # These are all valid states in the half-tree beginning with 4.
    expected_half = {
        (): ({(1,2),(1,5),(2,3),(2,6),(3,4),(4,5)}, {4,5}),
        (4,): ({(1,2),(1,5),(2,3),(2,6),(3,5)}, {3}),
        (4,3): ({(1,2),(1,5),(2,5),(2,6)}, {2,5}),
        (4,3,5): ({(2,6)}, {1}),
        (4,3,5,1): ({(2,6)}, set()),
        (4,3,2): ({(1,6),(5,6)}, {1,6}),
        (4,3,2,1): ({(5,6)}, set()),
        (4,3,2,6): ({(1,5)}, {5}),
        (4,3,2,6,5): (set(), {1}),
        (4,3,2,6,5,1): (set(), set()),
    }
    for prefix, (edges, black) in expected_half.items():
        actual_edges, actual_black = states[prefix]
        assert actual_edges == {frozenset(e) for e in edges}
        assert actual_black == black
    reflection = {1:3, 3:1, 4:5, 5:4, 2:2, 6:6}
    assert {frozenset(reflection[v] for v in e) for e in INITIAL_EDGES} == INITIAL_EDGES
    assert {reflection[v] for v in INITIAL_BLACK} == INITIAL_BLACK
    assert tuple(reflection[v] for v in first) == second
    result = {
        'model':'OpenAI GPT 6.1 Sol, Max reasoning',
        'python':platform.python_version(), 'total_edit_bound_in_original':4,
        'vertices':VERTICES, 'edges':sorted(sorted(e) for e in INITIAL_EDGES),
        'initial_black':sorted(INITIAL_BLACK), 'successful_words':sorted(successful),
        'stalled_maximal_words':sorted(stalled), 'legal_prefix_count':len(states),
        'principal_subsets_checked':subsets, 'distinct_words_checked':distinct_words,
        'full_permutations_in_distinct_words':720, 'augmented_matrix':matrix,
        'longest_common_subsequence_length':lcs, 'total_insertion_deletion_distance':distance,
        'h4_vertices':2, 'h4_edges':0, 'h4_components':2,
        'smallest_connected_threshold_for_this_graph':8,
        'hand_proof_half_tree_checked':True, 'reflection_checked':True,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'states':[{'prefix':prefix, 'edges':sorted(sorted(e) for e in edges), 'black':sorted(black)}
                  for prefix,(edges,black) in sorted(states.items())],
        'limits':'Two same-model computational methods support the self-contained hand proof; no independent human or theorem-prover verification.'
    }
    if args.output:
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('states','augmented_matrix')}))


if __name__ == '__main__':
    main()
