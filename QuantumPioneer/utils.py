import re
import typing as t

from rdkit import Chem
from rdmc import RDKitMol

NONE_GROUP = "[Xe]"  # using Xe as a None group that will match nothing;
# use when you do not care what the species is


def adjust_atom_map_smi_indexing(
    rxn_smi: str,
    mode: str,
) -> str:
    """
    Helper function to adjust the indexing of atom mapped smiles. Useful for converting
    the smi to zero- or one- indexing.

    Parameters
    ----------
    rxn_smi : str
        Input atom-mapped reaction smiles.
    mode : str
        plus_one: increase all indexing by 1
        minus_one: reduce all indexing by 1

    Returns
    -------
    str
        Re-numbered smi string.
    """

    def decrement(m):
        return str(int(m.group().rstrip("]")) - 1) + "]"

    def increment(m):
        return str(int(m.group().rstrip("]")) + 1) + "]"

    # Use a regular expression to find all the integers and decrement or increment each
    # one. Note we search for "{number}]" to ensure that the number corresponds to an
    # atom index. Sometimes, SMILES strings include numbers to indicate connectivity
    # i.e. in rings.
    if mode == "plus_one":
        new_smi = re.sub(r"\d+]", increment, rxn_smi)
    elif mode == "minus_one":
        new_smi = re.sub(r"\d+]", decrement, rxn_smi)
    else:
        raise ValueError(
            f"Specificed mode {mode} not recognized. "
            "Must be either plus_one or minus_one"
        )

    return new_smi


def determine_atom_map_smi_indexing(
    rxn_smi: str,
) -> int:
    """
    Helper function to determine the starting index of an atom mapped reaction smile.
    Expect either zero- or one- indexing.

    Parameters
    ----------
    rxn_smi : str
        Input atom-mapped reaction smiles.

    Returns
    -------
    int
        Starting index.
    """

    # Extract all integers from the string
    integers = [int(match.rstrip("]")) for match in re.findall(r"\d+]", rxn_smi)]

    # Return the smallest integer if there's any integer in the string
    return min(integers) if integers else None


def split_rxn_smi(
    rxn_smi: str,
) -> t.Tuple[t.List[str], t.List[str]]:
    """
    Split a given reaction smile into reactant and product smiles in lists.

    Parameters
    ----------
    rxn_smi : str
        Input reaction smiles.

    Returns
    -------
    Union[t.List[str], t.List[str]]
        Smiles of individual reactants and products, in separate lists.
    """

    reactants = rxn_smi.split(">>")[0].split(".")
    products = rxn_smi.split(">>")[1].split(".")

    return reactants, products


def reorder_reaction_smile(
    rxn_smi: str,
    r_pattern: t.Sequence[str] = (),
    p_pattern: t.Sequence[str] = (),
) -> str:
    """
    Preprocess a reaction smile to a specified order by user.

    This function will separate a reaction smile into species, then reorder them into
    reactants and products based on SMARTS patterns specified by user, and finally
    return the ordered new reaction smile.

    Note: use NONE_GROUP if you do not care about which species to be matched in a
    particular position. Do not use a wild card [*] for this purpose, for it will match
    any species and the logic in this function does not work well with it.
    Make sure to test this function. It is common to make mistakes in SMARTS.

    Parameters
    ----------
    rxn_smi : str
        Atom-mapped reaction smiles.
    r_pattern : Optional[t.List[str]], optional
        SMARTS pattern for matching reactants, length must match number of reactants in
        rxn_smi.
    p_pattern : Optional[t.List[str]], optional
        SMARTS pattern for matching products, length must match number of products in
        rxn_smi.

    Returns
    -------
    str
        New reaction smiles with species match the order of the pattern specified.
    """

    # return original smile if no pattern to match
    if not (r_pattern or p_pattern):
        return rxn_smi

    # get reactants and products from provided smi
    _reactants, _products = split_rxn_smi(rxn_smi)

    # raise if provided number of patterns mismatch given smi
    if len(_reactants) != len(r_pattern) or len(_products) != len(p_pattern):
        raise ValueError(
            "Provided number of patterns does not match number of species "
            "in the given reaction."
        )

    ps = Chem.SmilesParserParams()
    ps.removeHs = False

    def match_patterns(smis, patterns):
        smis = smis[:]
        # turn smi into rdkit molecule for matching

        mols = [Chem.MolFromSmiles(smi, ps) for smi in smis]
        matched = []
        for pattern in patterns:
            # turn pattern into rdkit molecule for substructure matching
            patt = Chem.MolFromSmarts(pattern)
            try:
                # return species index of the first match
                matched_idx = next(
                    i for i, x in enumerate(mols) if x.GetSubstructMatch(patt)
                )
            except StopIteration as e:
                if NONE_GROUP in patterns:
                    # NONE_group means the user does not care about which species get
                    # matched in the current index
                    # place holder for index/order keeping, will be replaced by leftover
                    # species later
                    matched.append(None)
                    continue
                raise ValueError(
                    f"Pattern {pattern} not found in provided reaction species."
                ) from e

            # add matched species to new reactants list
            matched.append(smis[matched_idx])
            mols.pop(matched_idx)
            smis.pop(matched_idx)

        # this means the user does not care about species order
        if matched == [None, None]:
            # so we leave order unchanged
            matched = smis
        # one of the species can be any left over species
        elif None in matched:
            # replace the place holder with left over species
            matched[matched.index(None)] = smis[0]

        return matched

    reactants = match_patterns(_reactants, r_pattern)
    products = match_patterns(_products, p_pattern)

    return ".".join(reactants) + ">>" + ".".join(products)


def isomorphic_check(
    mol1: RDKitMol,
    mol2: RDKitMol,
) -> bool:
    """
    Compare if two rdkit molecules are the same.

    Parameters
    ----------
    mol1 : RDKitMol
    mol2 : RDKitMol

    Returns
    -------
    bool
        True if the same.
    """

    return mol1.HasSubstructMatch(mol2) and mol2.HasSubstructMatch(mol1)


def get_ordered_integers(
    rxn_smi: str,
    sorted: bool = False,
) -> t.List[int]:
    """
    Get integers ordered by occurrence from a string. Useful to extract atom mapping
    from smiles.

    Parameters
    ----------
    rxn_smi : str
        Atom-mapped reaction smiles.
    sorted : bool, optional
        Whether to sort the integers, by default False.

    Returns
    -------
    t.List[int]
        Extracted integer list.
    """

    # Extract all integers from the string
    int_list = [int(match.rstrip("]")) for match in re.findall(r"\d+\]", rxn_smi)]
    if sorted:
        int_list.sort()

    return int_list


def get_neighbour_atom(mol, center_atom_idx, exlude_atom_idx_list=None):
    """
    Get neighboring atoms of a given atom in a molecule.

    Parameters
    ----------
    mol : RDKitMol
        The molecule.
    center_atom_idx : int
        Index of the center atom.
    exlude_atom_idx_list : list, optional
        t.List of atom indices to exclude, by default None.

    Returns
    -------
    dict
        Dictionary of neighboring atom indices and their symbols.
    """
    neighbour = {
        nb.GetIdx(): mol.GetAtomWithIdx(nb.GetIdx()).GetSymbol()
        for nb in mol.GetAtomWithIdx(center_atom_idx).GetNeighbors()
    }

    if exlude_atom_idx_list:
        for x in exlude_atom_idx_list:
            if x in neighbour:
                del neighbour[x]

    return neighbour


def find_fragment(mol, center_atom_idx, exlude_atom_idx_list):
    """
    Find the fragment of a molecule centered around a given atom.

    Parameters
    ----------
    mol : RDKitMol
        The molecule.
    center_atom_idx : int
        Index of the center atom.
    exlude_atom_idx_list : list
        t.List of atom indices to exclude.

    Returns
    -------
    dict
        Dictionary of fragment atom indices and their symbols.
    """
    frag = {}
    max_frag_size = len(mol.GetAtomicNumbers())
    center_atom = {center_atom_idx: mol.GetAtomWithIdx(center_atom_idx).GetSymbol()}

    frag = get_neighbour_atom(mol, center_atom_idx, exlude_atom_idx_list)

    search_list = list(frag.keys())
    to_exlude = []
    to_exlude.append(center_atom_idx)

    while search_list:
        for k in search_list:
            if mol.GetAtomWithIdx(k).GetSymbol() == "H":
                to_exlude.append(k)
                search_list = list({x for x in search_list if x not in to_exlude})
            neighbour = get_neighbour_atom(mol, k, to_exlude)
            search_list.extend(list(neighbour.keys()))
            to_exlude.append(k)
            search_list = list({x for x in search_list if x not in to_exlude})
            frag.update(neighbour)

            if len(frag.keys()) >= max_frag_size:
                frag.update(center_atom)
                return frag
    frag.update(center_atom)
    return frag
