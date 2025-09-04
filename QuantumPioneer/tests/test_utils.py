from rdkit import Chem

from QuantumPioneer import utils


def test_adjust_atom_map_smi_indexing():
    zero_idx_smi = (
        "[H:10][C:15]([H:11])([H:12])[C:16]([H:13])([H:14])[F:17]"
        "."
        "[H:0][C:5]([H:1])([H:2])[S:9][C:6]([H:3])([H:4])[O:8][O:7]"
        ">>"
        "[H:11][C:15]([H:12])[C:16]([H:13])([H:14])[F:17]"
        "."
        "[H:0][C:5]([H:1])([H:2])[S:9][C:6]([H:3])([H:4])[O:8][O:7][H:10]"
    )
    one_idx_smi = (
        "[H:11][C:16]([H:12])([H:13])[C:17]([H:14])([H:15])[F:18]"
        "."
        "[H:1][C:6]([H:2])([H:3])[S:10][C:7]([H:4])([H:5])[O:9][O:8]"
        ">>"
        "[H:12][C:16]([H:13])[C:17]([H:14])([H:15])[F:18]"
        "."
        "[H:1][C:6]([H:2])([H:3])[S:10][C:7]([H:4])([H:5])[O:9][O:8][H:11]"
    )
    assert (
        utils.adjust_atom_map_smi_indexing(rxn_smi=zero_idx_smi, mode="plus_one")
        == one_idx_smi
    )
    assert (
        utils.adjust_atom_map_smi_indexing(rxn_smi=one_idx_smi, mode="minus_one")
        == zero_idx_smi
    )

    zero_idx_smi = (
        "[C:0]([C:1]([C:2]([C:3]([Br:4])([H:22])[H:23])([H:20])[H:21])([H:18])[H:19])([H:15])([H:16])[H:17]"
        "."
        "[c:5]1([H:24])[c:6]([H:25])[c:7]([S-:8])[c:9]([H:26])[c:10]([H:27])[c:11]1[N+:12](=[O:13])[O-:14]"
    )
    one_idx_smi = (
        "[C:1]([C:2]([C:3]([C:4]([Br:5])([H:23])[H:24])([H:21])[H:22])([H:19])[H:20])([H:16])([H:17])[H:18]"
        "."
        "[c:6]1([H:25])[c:7]([H:26])[c:8]([S-:9])[c:10]([H:27])[c:11]([H:28])[c:12]1[N+:13](=[O:14])[O-:15]"
    )
    assert (
        utils.adjust_atom_map_smi_indexing(rxn_smi=zero_idx_smi, mode="plus_one")
        == one_idx_smi
    )
    assert (
        utils.adjust_atom_map_smi_indexing(rxn_smi=one_idx_smi, mode="minus_one")
        == zero_idx_smi
    )


def test_determine_atom_map_smi_indexing():
    zero_idx_smi = (
        "[H:10][C:15]([H:11])([H:12])[C:16]([H:13])([H:14])[F:17]"
        "."
        "[H:0][C:5]([H:1])([H:2])[S:9][C:6]([H:3])([H:4])[O:8][O:7]"
        ">>"
        "[H:11][C:15]([H:12])[C:16]([H:13])([H:14])[F:17]"
        "."
        "[H:0][C:5]([H:1])([H:2])[S:9][C:6]([H:3])([H:4])[O:8][O:7][H:10]"
    )
    one_idx_smi = (
        "[H:11][C:16]([H:12])([H:13])[C:17]([H:14])([H:15])[F:18]"
        "."
        "[H:1][C:6]([H:2])([H:3])[S:10][C:7]([H:4])([H:5])[O:9][O:8]"
        ">>"
        "[H:12][C:16]([H:13])[C:17]([H:14])([H:15])[F:18]"
        "."
        "[H:1][C:6]([H:2])([H:3])[S:10][C:7]([H:4])([H:5])[O:9][O:8][H:11]"
    )

    assert utils.determine_atom_map_smi_indexing(rxn_smi=zero_idx_smi) == 0
    assert utils.determine_atom_map_smi_indexing(rxn_smi=one_idx_smi) == 1

    two_idx_smi = (
        "[C:2]([C:3]([C:4]([C:5]([Br:6])([H:24])[H:25])([H:22])[H:23])([H:20])[H:21])([H:17])([H:18])[H:19]"
        "."
        "[c:7]1([H:26])[c:8]([H:27])[c:9]([S-:10])[c:11]([H:28])[c:12]([H:29])[c:13]1[N+:14](=[O:15])[O-:16]"
    )
    assert utils.determine_atom_map_smi_indexing(rxn_smi=two_idx_smi) == 2


def test_reorder_reaction_smile():
    input_rxn_smi = (
        "[H:1][C:6]([H:2])([H:3])[S:10][C:7]([H:4])([H:5])[O:9][O:8]"
        "."
        "[H:11][C:16]([H:12])([H:13])[C:17]([H:14])([H:15])[F:18]"
        ">>"
        "[H:1][C:6]([H:2])([H:3])[S:10][C:7]([H:4])([H:5])[O:9][O:8][H:11]"
        "."
        "[H:12][C:16]([H:13])[C:17]([H:14])([H:15])[F:18]"
    )
    expected_rxn_smi = (
        "[H:11][C:16]([H:12])([H:13])[C:17]([H:14])([H:15])[F:18]"
        "."
        "[H:1][C:6]([H:2])([H:3])[S:10][C:7]([H:4])([H:5])[O:9][O:8]"
        ">>"
        "[H:12][C:16]([H:13])[C:17]([H:14])([H:15])[F:18]"
        "."
        "[H:1][C:6]([H:2])([H:3])[S:10][C:7]([H:4])([H:5])[O:9][O:8][H:11]"
    )

    r_pattern = [utils.NONE_GROUP, "[#1,C,N,O]-[O;X2]-[O;X1+0]"]
    p_pattern = ["[CX3+0,NX2+0,OX1+0]", "[*]-[O;X2]-[OH]"]

    assert (
        utils.reorder_reaction_smile(
            rxn_smi=input_rxn_smi, r_pattern=r_pattern, p_pattern=p_pattern
        )
        == expected_rxn_smi
    )

    input_rxn_smi = (
        "[H:1][O:3][O:2].[H:4][N:8]([H:5])[C:6]#[N:7]"
        ">>"
        "[H:5][N:8][C:6]#[N:7].[H:1][O:3][O:2][H:4]"
    )
    expected_rxn_smi = (
        "[H:4][N:8]([H:5])[C:6]#[N:7].[H:1][O:3][O:2]"
        ">>"
        "[H:5][N:8][C:6]#[N:7].[H:1][O:3][O:2][H:4]"
    )
    assert (
        utils.reorder_reaction_smile(
            rxn_smi=input_rxn_smi, r_pattern=r_pattern, p_pattern=p_pattern
        )
        == expected_rxn_smi
    )


def test_isomorphic_check():
    smi1 = "c1ccc2c(c1)c3ccccc3[nH]2"
    smi2 = "C1=CC=C2C(=C1)C3=CC=CC=C3N2"
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)

    assert utils.isomorphic_check(mol1, mol2)

    smi1 = "CCC"
    smi2 = "C1=CC=C2C(=C1)C3=CC=CC=C3N2"
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)

    assert not utils.isomorphic_check(mol1, mol2)

    smi1 = "[H:10][C:15]([H:11])([H:12])[C:16]([H:13])([H:14])[F:17]"
    smi2 = "[H:20][C:45]([H:11])([H:12])[C:16]([H:13])([H:14])[F:17]"
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)

    assert utils.isomorphic_check(mol1, mol2)


def test_get_ordered_integers():
    smi1 = "[H:10][C:15]([H:11])([H:12])[C:16]([H:13])([H:14])[F:17]"
    assert utils.get_ordered_integers(smi1) == [10, 15, 11, 12, 16, 13, 14, 17]

    smi2 = "[c:5]1([H:24])[c:6]([H:25])[c:7]([S-:8])[c:9]([H:26])[c:10]([H:27])[c:11]1[N+:12](=[O:13])[O-:14]"  # noqa E501
    assert utils.get_ordered_integers(smi2) == [
        5,
        24,
        6,
        25,
        7,
        8,
        9,
        26,
        10,
        27,
        11,
        12,
        13,
        14,
    ]
