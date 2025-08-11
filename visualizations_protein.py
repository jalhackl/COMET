import nglview as nv
import MDAnalysis as mda 


def create_protein_visualization(clustering, protein_structure, representation_type="cartoon", atom_selection="all"):
    
    #reasonable parameters for representation_type: "cartoon", "licorice"
    #reasonable parameters for atom_selection: "all", "backbone"

    
    pdb = mda.Universe(protein_structure)
    ca = pdb.select_atoms('name CA')

    ca.tempfactors = clustering
    pdb2 = mda.Universe(protein_structure)

    u = pdb2
    u.add_TopologyAttr('tempfactors', pdb.atoms.tempfactors)
    u.atoms.tempfactors = pdb.atoms.tempfactors


    for residue in u.residues:
        maxfact = max(residue.atoms.tempfactors)
        for atom in residue.atoms:
            atom.tempfactor = maxfact

    w = nv.show_mdanalysis(u, height='800px')


    w.representations = [
            {"type": representation_type, "params": {
                "sele": atom_selection, "color": "bfactor"
            }}
        ]


    w.display()

    return w