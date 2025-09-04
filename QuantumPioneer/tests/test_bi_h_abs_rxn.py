"""
Unit and regression test for the BimolecularHydrogenAbstractionReaction class.
"""

# Import package, test suite, and other packages as needed

import QuantumPioneer as qp


def test_instantiation():
    """
    Test instantiation of BimolecularHydrogenAbstractionReaction class
    """
    rnx_smi = (
        "[H:1][O:3][O:2].[H:4][N:8]([H:5])[C:6]#[N:7]"
        ">>"
        "[H:5][N:8][C:6]#[N:7].[H:1][O:3][O:2][H:4]"
    )
    rxn = qp.BimolecularHydrogenAbstractionReaction(rnx_smi)

    assert rxn.formed_bond == [(1, 3)]
    assert rxn.broken_bond == [(3, 7)]
    assert rxn.ts_h_index == 3
    assert rxn.ts_pivot_indices == [7, 1]
    assert rxn.r_fragment_indices == ((3, 4, 5, 6, 7), (0, 1, 2))
    assert rxn.p_fragment_indices == ((4, 5, 6, 7), (0, 1, 2, 3))
    assert rxn.r1_neighbour_indices == [5, 4]
    assert rxn.r2_neighbour_indices == [2]


if __name__ == "__main__":
    test_instantiation()
