import itertools as it

import numpy as np
from rdkit.Chem import AllChem, Draw, rdChemReactions
from rdmc import RDKitMol
from rdmc import ts as rdmc_ts
from rdmc.forcefield import RDKitFF
import py3Dmol
from rdmc.view import ts_viewer
import IPython

from QuantumPioneer import utils

FF = AllChem.ETKDGv3()
# this make sure we get different embedding each time
FF.randomSeed = np.random.randint(1, 10000000)

# Empirical parameters
# "angle_X_H_Y": initial angle for TS pivot. Do not make it too close to 180.
# "dihedral_r1": initial dihedral angle for TS, defined using r1, does not really
#     matter most of the time for bi-molecular H abstraction.
# "dihedral_r2": the other diehdral angle, defined using reactant 2 or r2
# "bond_length_scale_factor_r1": how much to scale the TS bond length based on reactant
#     bond length for r1; can be modified to scale each bond differently, but usually
#     1.2-1.25 is good guess for bi-molecular H abstraction with C-H and 1.15 - 1.2
#     for O-H.
# "bond_length_scale_factor_r2": how much to scale the TS bond length based on reactant
#     bond length for r1; can be modified to scale each bond differently, but usually
#     1.2-1.25 is good guess for bi-molecular H abstraction with C-H and 1.15 - 1.2
#     for O-H.
# "bond_length_X_H": directly specifiy the TS bond length for r1, useful if another
#     bond length estimator e.g., TS-EGNN/Chemprop/Kinbot can provide this information
# "bond_length_H_Y": directly specifiy the TS bond length for r2
_DEFAULT_TS_GUESS_PARAMETERS = {
    "angle_X_H_Y": 160,
    "dihedral_r1": 0,
    "dihedral_r2": 0,
    "bond_length_scale_factor_r1": 1.22,
    "bond_length_scale_factor_r2": 1.19,
    "bond_length_X_H": None,
    "bond_length_H_Y": None,
}

ROO_GROUP = "[#1,C,N,O]-[O;X2]-[O;X1+0]"  # match ROO radical, with R = H, C, O, N only
ROOH_GROUP = "[*]-[O;X2]-[OH]"  # match any ROOH group
RADICAL_GROUP = "[CX3+0,NX2+0,OX1+0]"  # match radical, with R = C, O, N only


class BimolecularHydrogenAbstractionReaction:
    """
    A bi-molecular hydrogen abstraction reaction.

    Parameters
    ----------
    rxn_smi : str
        Atom-mapped reaction smiles.
    num_ts_conformers : int, optional
        Number of transition state conformers to generate. Default is 1.
    max_attemps_per_conformer : int, optional
        Maximum number of attempts per conformer. Default is 5.

    Keyword Arguments
    ----------------
    **ts_guess_parameters : dict
        Keyword arguments for transition state guess parameters. The available
        parameters are:

        - `angle_X_H_Y` (default: 160): initial angle for the transition state pivot. Do
        not make it too close to 180.
        - `dihedral_r1` (default: 0): initial dihedral angle for the transition state,
        defined using reactant 1.
        - `dihedral_r2` (default: 0): the other diehdral angle for the transition state,
        defined using reactant 2.
        - `bond_length_scale_factor_r1` (default: 1.22): how much to scale the
        transition state bond length based on reactant bond length for reactant 1.
        - `bond_length_scale_factor_r2` (default: 1.19): how much to scale the
        transition state bond length based on reactant bond length for reactant 2.
        - `bond_length_X_H` (default: None): directly specify the transition state bond
        length for reactant 1.
        - `bond_length_H_Y` (default: None): directly specifiy the transition state bond
        length for reactant 2.

    Attributes
    ----------
    r_complex : RDKitMol
        RDKitMol for reactant complex.
    p_complex : RDKitMol
        RDKitMol for product complex.
    ts_complex : RDKitMol
        RDKitMol for transition state complex.
    pivot_atoms : list
        Pivot atoms indices for the transition state.
    r_fragment_indices : tuple
        Atom indices for reactant fragments.
    p_fragment_indices : tuple
        Atom indices for product fragments.
    r1_neighbour_indices : list
        Atom indices for reactant fragment 1 neighbors.
    r2_neighbour_indices : list
        Atom indices for reactant fragment 2 neighbors.
    p1_neighbour_indices : list
        Atom indices for product fragment 1 neighbors.
    p2_neighbour_indices : list
        Atom indices for product fragment 2 neighbors.
    """

    def __init__(
        self,
        rxn_smi: str,
        num_ts_conformers: int = 1,
        max_attemps_per_conformer: int = 5,
        **ts_guess_parameters,
    ):
        self._ts_guess_parameters = _DEFAULT_TS_GUESS_PARAMETERS.copy()
        self._ts_guess_parameters.update(ts_guess_parameters)

        rxn_smi = self._reorder_reaction_smile(rxn_smi)

        # generate reactant and product complex RDkitMOl from smiles, atoms are always
        # zero-indexed, use mol.GetAtomMapNumbers() to get atom map specified in the
        # smiles
        r_complex, p_complex = map(RDKitMol.FromSmiles, rxn_smi.split(">>"))

        # perceive reaction center
        # formed, broken bonds indices e.g., fbond = [(1, 3)] means a bond forms between
        # atom with index 1 and 3, notice that atoms are zero-indexed and reaction is
        # analyzed in the forward direction
        fbond, bbond = rdmc_ts.get_formed_and_broken_bonds(r_complex, p_complex)

        # the H atom index in the TS
        the_h_atom = set(it.chain(*fbond)).intersection(it.chain(*bbond)).pop()

        # TS pivot atom indices, not yet sorted by given reactants order
        _pivot_atoms = list(set(it.chain(*(fbond + bbond))).difference({the_h_atom}))
        pivot_atoms = [None] * len(_pivot_atoms)

        # get atom indices in each molecule fragments, re-arranged to match the
        # reactants and products order specified in the reaction smile
        _frags_r = r_complex.GetMolFrags(asMols=False)  # reactants complex
        _frags_p = p_complex.GetMolFrags(asMols=False)  # products complex

        frags_r = [None] * len(_frags_r)
        frags_p = [None] * len(_frags_p)

        _reactants, _products = utils.split_rxn_smi(rxn_smi=rxn_smi)

        _reactants_atom_map = [
            utils.get_ordered_integers(x, sorted=True) for x in _reactants
        ]
        _products_atom_map = [
            utils.get_ordered_integers(x, sorted=True) for x in _products
        ]

        r_complex_atom_map = r_complex.GetAtomMapNumbers()
        _frags_r_atom_map = [[r_complex_atom_map[i] for i in x] for x in _frags_r]

        for i, x in enumerate(_frags_r_atom_map):
            idx = _reactants_atom_map.index(x)
            frags_r[idx] = _frags_r[i]
        frags_r = tuple(frags_r)

        p_complex_atom_map = p_complex.GetAtomMapNumbers()
        _frags_p_atom_map = [[p_complex_atom_map[i] for i in x] for x in _frags_p]

        for i, x in enumerate(_frags_p_atom_map):
            idx = _products_atom_map.index(x)
            frags_p[idx] = _frags_p[i]
        frags_p = tuple(frags_p)

        # re-arrange pivot_atoms to match the reactants ordered in the smi;
        # pivot = R1 -- H(TS) -- R2; e.g., pivot = [15, 7] means atom with index 15 in
        # R1 and index 7 in R2 are atoms in the reaction coordinate
        for x in _pivot_atoms:
            idx = [x in f for f in frags_r].index(True)
            pivot_atoms[idx] = x

        # get indices for neighbouring atoms of pivot atoms
        # ordered by how "bulky" the molecular fragment that the atom is connected to
        _r1_neighbour_indices = list(
            utils.get_neighbour_atom(r_complex, pivot_atoms[0], [the_h_atom]).keys()
        )
        _r1_neighbour_indices_by_size = [
            (x, len(utils.find_fragment(r_complex, x, [pivot_atoms[0]])))
            for x in _r1_neighbour_indices
        ]
        _r1_neighbour_indices_by_size.sort(key=lambda x: x[1], reverse=True)
        r1_neighbour_indices = [x[0] for x in _r1_neighbour_indices_by_size]

        _r2_neighbour_indices = list(
            utils.get_neighbour_atom(r_complex, pivot_atoms[1], [the_h_atom]).keys()
        )
        _r2_neighbour_indices_by_size = [
            (x, len(utils.find_fragment(r_complex, x, [pivot_atoms[1]])))
            for x in _r2_neighbour_indices
        ]
        _r2_neighbour_indices_by_size.sort(key=lambda x: x[1], reverse=True)
        r2_neighbour_indices = [x[0] for x in _r2_neighbour_indices_by_size]

        self.rxn_smi = rxn_smi
        self.r_complex = r_complex
        self.p_complex = p_complex
        self.formed_bond = fbond  # R2OO
        self.broken_bond = bbond  # R1H
        self.ts_h_index = the_h_atom
        self.ts_pivot_indices = pivot_atoms
        self.r_fragment_indices = frags_r
        self.p_fragment_indices = frags_p
        self.r1_neighbour_indices = r1_neighbour_indices
        self.r2_neighbour_indices = r2_neighbour_indices

        self.ts_complexes = {}
        self.ts_relax_scores = {}
        self._generate_conformers(num_ts_conformers, max_attemps_per_conformer)

    def __repr__(self) -> str:
        return self.rxn_smi

    def _reorder_reaction_smile(self, rxn_smi: str) -> str:
        return utils.reorder_reaction_smile(
            rxn_smi, [utils.NONE_GROUP, ROO_GROUP], [RADICAL_GROUP, ROOH_GROUP]
        )

    def _generate_conformers(self, num_ts_conformers, max_attemps_per_conformer):
        for index in range(num_ts_conformers):
            for _ in range(max_attemps_per_conformer):
                try:
                    # Embed 3D geometry for reactants and products. Here we use
                    # ETKDGv3() defined on top of the notebook, but can be changed to
                    # others if needed
                    self.r_complex.EmbedConformer(FF)
                    self.p_complex.EmbedConformer(FF)
                    # We need to add redundant bond to the reactant complex graph to
                    # represent the TS geometry
                    ts_complex = self.r_complex.AddRedundantBonds(self.formed_bond)
                    relax_score = self._generate_ts_guess(ts_complex)
                    break
                except Exception as e:
                    raise ValueError("Failed to generate TS conformer.") from e
            self.ts_complexes[index] = ts_complex
            self.ts_relax_scores[index] = relax_score

    @staticmethod
    def _return_opt_spc_bond_distance(
        spc_smi,
        pivot_atom,
        the_h_atom,
        atom_map_idx,
        averaged=5,
    ):
        """
        Return the optimized bond distance of reacting R-H from reactant/product
        species.
        Only use on non-radical RH species.

        This function calcualte R-H bond distances for closed-shell reactant/product
        species based on averaging of N conformers embedded from force field a helper
        function used in gen_ts
        """

        bd_list = []

        for _ in range(averaged):
            r = RDKitMol.FromSmiles(spc_smi)
            ff = RDKitFF("mmff94s")
            r.EmbedConformer(FF)
            ff.setup(r)
            ff.optimize()
            m = ff.get_optimized_mol()

            r_conf = r.GetConformer()
            r_conf.SetPositions(m.GetPositions())

            h_idx = atom_map_idx.index(the_h_atom)
            pivot_idx = atom_map_idx.index(pivot_atom)

            bd = r_conf.GetBondLength([h_idx, pivot_idx])
            bd_list.append(bd)

        return sum(bd_list) / len(bd_list)

    def _initialize_ts_guess_geometry(self, ts_conformer):
        _reactants, _products = utils.split_rxn_smi(self.rxn_smi)

        # set TS bond distances
        # update bond distance of r1 -- H(ts)
        # step 1: find the reacting bond distance of r1 when it's not in the TS based
        # on averaging of 10 conformers
        r1_dist = self._return_opt_spc_bond_distance(
            spc_smi=_reactants[0],
            pivot_atom=self.ts_pivot_indices[0],
            the_h_atom=self.ts_h_index,
            atom_map_idx=self.r_fragment_indices[0],
        )

        # step 2: scale the reacting bond with a scaling factor, the factor is provided
        # by user currently, but it can be learned since it depends on reactant/product
        # type (usually 25% is good starting guess for C-H, 15% for O-H etc)
        bond_length_X_H = (
            r1_dist * self._ts_guess_parameters["bond_length_scale_factor_r1"]
        )

        # step 3: set bond length for the TS complex
        ts_conformer.SetBondLength(
            [self.ts_pivot_indices[0], self.ts_h_index], bond_length_X_H
        )

        # update bond distance of H(ts) -- r2
        r2_dist = self._return_opt_spc_bond_distance(
            spc_smi=_products[1],
            pivot_atom=self.ts_pivot_indices[1],
            the_h_atom=self.ts_h_index,
            atom_map_idx=self.p_fragment_indices[1],
        )

        bond_length_H_Y = (
            r2_dist * self._ts_guess_parameters["bond_length_scale_factor_r2"]
        )

        ts_conformer.SetBondLength(
            [self.ts_pivot_indices[1], self.ts_h_index], bond_length_H_Y
        )

        # set TS angle
        # angle_X_H_Y can be learned, but usually 160 deg is good for most bi-mol
        # H abs family
        ts_conformer.SetAngleDeg(
            [
                self.ts_pivot_indices[0],
                self.ts_h_index,
                self.ts_pivot_indices[1],
            ],
            self._ts_guess_parameters["angle_X_H_Y"],
        )

        # set TS dihedral
        # note: there are two sets of dihedrals for the TS, since a dihedral is defined
        # by 4 connecting atoms, and there are 5 atoms connected in the TS region, thus
        # 2 sets of dihedrals

        # set dihedrals
        # note: the dihedrals can actually be searched or learned, but not implemented
        # yet, currently we set to user provided value
        ts_conformer.SetTorsionDeg(
            [
                self.r1_neighbour_indices[0],
                self.ts_pivot_indices[0],
                self.ts_h_index,
                self.ts_pivot_indices[1],
            ],
            self._ts_guess_parameters["dihedral_r1"],
        )

        ts_conformer.SetTorsionDeg(
            [
                self.ts_pivot_indices[0],
                self.ts_h_index,
                self.ts_pivot_indices[1],
                self.r2_neighbour_indices[0],
            ],
            self._ts_guess_parameters["dihedral_r2"],
        )

    def _optimize_ts_using_force_field(self, ts_conformer):
        """
        A helper function to optimize the transition state using force field.

        This function performs a constrained optimization that first fixes the two TS
        bond distances, then optimizes the geometry to a stable point to relax it (not
        to a saddle point as in typical ts search). The idea behind this is to relax
        each fragment of the TS to make the geometry more reasonable.

        Parameters
        ----------
        species_complex_dict : dict
            A dictionary containing information about the species complex.
        """
        bond_length_X_H = ts_conformer.GetBondLength(self.broken_bond[0])
        bond_length_H_Y = ts_conformer.GetBondLength(self.formed_bond[0])

        ff = RDKitFF("mmff94s")
        fake_ts = RDKitMol.FromSmiles(self.rxn_smi.split(">>")[0])
        fake_ts.EmbedConformer()
        fake_ts.SetPositions(ts_conformer.GetPositions())
        ff.setup(fake_ts)

        the_h_atom = self.ts_h_index
        pivot_atom_r1 = self.ts_pivot_indices[0]
        pivot_atom_r2 = self.ts_pivot_indices[1]

        ff.add_distance_constraint(
            atoms=[the_h_atom, pivot_atom_r1], value=bond_length_X_H
        )
        ff.add_distance_constraint(
            atoms=[the_h_atom, pivot_atom_r2], value=bond_length_H_Y
        )

        ff.optimize()
        m = ff.get_optimized_mol()
        ts_conformer.SetPositions(m.GetPositions())

    @staticmethod
    def _check_fragment_collision(ts, h_idx, threshold=1.3):
        """
        This function checks if two parts of the TS are too close to each other.
        Since the TS bond distance is about 1.3 anstrom for C-H, if two fragments are
        closer than this, additional bonds may form (collision).
        """
        distance_matrix = np.triu(ts.GetDistanceMatrix())
        index = h_idx  # atom index of the ts h
        fragment_bond_distance_matrix = distance_matrix[
            0:index, index:
        ]  # check if two fragments are too close to each other
        if np.any(fragment_bond_distance_matrix < threshold):
            raise ValueError("Reactants collision detected.")
        else:
            small_distances = fragment_bond_distance_matrix[
                fragment_bond_distance_matrix < 2
            ]
            relax_score = (
                np.sum((small_distances - 2) ** 2)
                + np.min(small_distances)
                + np.max(fragment_bond_distance_matrix)
                + np.median(fragment_bond_distance_matrix)
                + np.average(fragment_bond_distance_matrix)
            )
            relax_score = relax_score / (5 * np.max(fragment_bond_distance_matrix))

            # notice that we return a score where the larger the score, the more
            # seperated the two fragments of the TS are (ideal for initial guess)
            return relax_score

    def _generate_ts_guess(self, ts_complex):
        """
        generate_ts_guess _summary_

        _extended_summary_

        Args:
            rxn_smi (_type_): _description_
            angle_X_H_Y (_type_): _description_
            dihedral_r1 (_type_): _description_
            dihedral_r2 (_type_): _description_
            bond_length_scale_factor_r1 (_type_): _description_
            bond_length_scale_factor_r2 (_type_): _description_
            bond_length_X_H (_type_, optional): _description_. Defaults to None.
            bond_length_H_Y (_type_, optional): _description_. Defaults to None.
        """

        ts_conformer = ts_complex.GetConformer()
        # initialize ts guess geometry
        self._initialize_ts_guess_geometry(ts_conformer)

        # optimize ts guess
        self._optimize_ts_using_force_field(ts_conformer)

        if ts_complex.HasCollidingAtoms(threshold=0.4):
            raise ValueError("Atom collision detected.")

        bond_length_X_H = ts_conformer.GetBondLength(self.broken_bond[0])
        bond_length_H_Y = ts_conformer.GetBondLength(self.formed_bond[0])

        threshold = min([bond_length_X_H, bond_length_H_Y]) * 0.98
        relax_score = self._check_fragment_collision(
            ts_complex, self.ts_h_index, threshold=threshold
        )

        return relax_score

    def to_dict(self):
        conformers = []
        for index, ts_complex in self.ts_complexes.items():
            conformers.append(
                (
                    self.ts_relax_scores[index],
                    "\n".join(ts_complex.ToXYZ().splitlines()[2:]) + "\n\n",
                )
            )
        conformers.sort(key=lambda y: y[0])
        return {
            "rxn_smi": self.rxn_smi,
            "formed_bond": self.formed_bond,
            "broken_bond": self.broken_bond,
            "ts_h_index": self.ts_h_index,
            "ts_pivot_indices": self.ts_pivot_indices,
            "r_fragment_indices": self.r_fragment_indices,
            "p_fragment_indices": self.p_fragment_indices,
            "r1_neighbour_indices": self.r1_neighbour_indices,
            "r2_neighbour_indices": self.r2_neighbour_indices,
            "ts_conformers_coord": tuple(conformers),
        }

    def _repr_html_(self):
        return Draw.ReactionToImage(
            rdChemReactions.ReactionFromSmarts(self.rxn_smi, useSmiles=True),
            useSVG=True,
        )

    def display_conformer(self, index: int = 0) -> py3Dmol.view:
        """
        Draw the conformer as an interactive 3D image.

        Parameters
        ----------
        index : int, optional
            Index of the conformer, by default 0.

        Returns
        -------
        py3Dmol.view
            The interactive 3D image.
        """
        return ts_viewer(
            self.r_complex, self.p_complex, self.ts_complexes[index], only_ts=True
        )
