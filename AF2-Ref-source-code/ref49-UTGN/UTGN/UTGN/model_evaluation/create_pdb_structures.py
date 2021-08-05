"""Create PDB file for predicted and actual 3D structure.
"""

import os
import subprocess
from ast import literal_eval
import argparse
import numpy as np
import tensorflow as tf
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure
from Bio.PDB import PDBIO

NUM_DIMENSIONS = 3
AA_LETTERS = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'E': 'GLU',
    'Q': 'GLN',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL'
}


class Protein:
    """Stores amino acid sequence and structure.

    Attributes:
        id_: name of protein
        primary: string of amino acid sequence
        actual_tertiary: np array
        mask: np array of 1 and 0s
        pred_tertiary: np array
    """

    _aa_dict = {
        'A': '0',
        'C': '1',
        'D': '2',
        'E': '3',
        'F': '4',
        'G': '5',
        'H': '6',
        'I': '7',
        'K': '8',
        'L': '9',
        'M': '10',
        'N': '11',
        'P': '12',
        'Q': '13',
        'R': '14',
        'S': '15',
        'T': '16',
        'V': '17',
        'W': '18',
        'Y': '19'
    }
    _mask_dict = {'-': '0', '+': '1'}
    _aa_dict = dict((v, k) for k, v in _aa_dict.items())

    def __init__(self, id_, primary, actual_tertiary, mask, pred_tertiary):
        self.id_ = id_
        self.primary = primary
        self.actual_tertiary = actual_tertiary
        self.pred_tertiary = pred_tertiary
        self.mask = mask
        self.int_to_aa()

    def int_to_aa(self):
        integers = list(self.primary.astype('str'))
        aa = "".join([self._aa_dict[integer] for integer in integers])
        self.primary = aa


def read_tertiary_file(path):
    """Read .tertiary file"""

    coords = np.transpose(np.loadtxt(path))
    # coords = np.fliplr(coords)
    return coords


def read_and_decode(filename_queue):
    """Parse a single instance of a tf record.

    Args:
        filename_queue: tf queue

    Returns:
        id_:tf tensor
        primary: tf tensor
        tertiary: tf tensor
        mask: tf tensor
    """

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    context, features = tf.parse_single_sequence_example(
        serialized_example,
        context_features={'id': tf.FixedLenFeature((1, ), tf.string)},
        sequence_features={
            'primary':
            tf.FixedLenSequenceFeature((1, ), tf.int64),
            'tertiary':
            tf.FixedLenSequenceFeature((NUM_DIMENSIONS, ),
                                       tf.float32,
                                       allow_missing=True),
            'mask':
            tf.FixedLenSequenceFeature((1, ), tf.float32, allow_missing=True)
        })

    id_ = context['id'][0]
    primary = tf.to_int32(features['primary'][:, 0])
    tertiary = features['tertiary']
    mask = features['mask'][:, 0]
    return id_, primary, tertiary, mask


def tf_record_to_dict(tf_path, tertiary_dir):
    """Convert tfrecord to a list of Proteins.

    Args:
        tf_path: path to tf record
        tertiary_dir: directory that holds .tertiary files

    Returns:
        list of Proteins
    """

    tf.reset_default_graph()
    with tf.Session() as sess:
        proteins = []
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

        filename_queue = tf.train.string_input_producer([tf_path],
                                                        shuffle=False)

        attributes = read_and_decode(filename_queue)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        size = sum(1 for _ in tf.python_io.tf_record_iterator(tf_path))
        for i in range(size):

            id_, primary, tertiary, mask = sess.run(list(attributes))
            id_ = id_.decode("utf-8")
            try:
                pred_coords = read_tertiary_file(tertiary_dir + id_ +
                                                 '.tertiary')
            except (FileNotFoundError, OSError):
                pred_coords = np.array([])
            protein = Protein(id_, primary, tertiary, mask, pred_coords)
            proteins.append(protein)
        coord.request_stop()
        coord.join(threads)

        return proteins


def create_pdb_file(protein, save_dir):
    """Create a PDB file from a Protein and return the structure.

    Args:
        protein: Protein
        save_dir: directory to save pdb files

    Returns:
        None
    """

    def create_structure(coords, pdb_type, remove_masked):
        """Create the structure.

        Args:
            coords: 3D coordinates of structure
            pdb_type: predict or actual structure
            remove_masked: whether to include masked atoms. If false,
                           the masked atoms have coordinates of [0,0,0].

        Returns:
            structure
        """

        name = protein.id_
        structure = Structure(name)
        model = Model(0)
        chain = Chain('A')
        for i, residue in enumerate(protein.primary):
            residue = AA_LETTERS[residue]
            if int(protein.mask[i]) == 1 or remove_masked == False:
                new_residue = Residue((' ', i + 1, ' '), residue, '    ')
                j = 3 * i
                atom_list = ['N', 'CA', 'CB']
                for k, atom in enumerate(atom_list):
                    new_atom = Atom(name=atom,
                                    coord=coords[j + k, :],
                                    bfactor=0,
                                    occupancy=1,
                                    altloc=' ',
                                    fullname=" {} ".format(atom),
                                    serial_number=0)
                    new_residue.add(new_atom)
                chain.add(new_residue)
        model.add(chain)
        structure.add(model)
        io = PDBIO()
        io.set_structure(structure)
        io.save(save_dir + name + '_' + pdb_type + '.pdb')
        return structure

    coords = np.around(protein.actual_tertiary, 1)
    coords = coords / 100
    create_structure(coords, "actual", remove_masked=True)

    if protein.pred_tertiary.size != 0:
        coords = np.around(protein.pred_tertiary, 1)
        coords = coords / 100
        create_structure(coords, "pred", remove_masked=False)


def create_pdb_files(proteins, save_dir):
    """Create multiple pdb files

    Args:
        proteins: list of Proteins
        save_dir: directory to save pdb files

    Returns:
        None
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for protein in proteins:
        create_pdb_file(protein, save_dir)

def create_TM_files(proteins, pdb_dir, tm_dir):
    """Run the TMScore script for each protein.

    Args:
        proteins: list of Proteins
        pdb_dir: directory that contains pdb files
        tm_dir: directory to save TM files

    Returns:
        None
    """

    if not os.path.exists(tm_dir):
        os.makedirs(tm_dir)

    tm_data = []
    tm_scores = []
    rmsds = []

    for protein in proteins:
        name = protein.id_
        pdb_actual = pdb_dir + name + '_actual.pdb'
        pdb_pred = pdb_dir + name + '_pred.pdb'
        tm_file = tm_dir + 'tm_score_' + name + '.txt'
        subprocess.call(
            './TMScore ' + pdb_pred + ' ' + pdb_actual + ' > ' + tm_file, 
            shell=True)

        if os.path.getsize(tm_file) > 0:
            f = open(tm_file, 'r').read().split('\n')
            rmsd = f[14].split()[-1]
            tm_score = f[16].split()[2]
            data = name + '\t' + tm_score + '\t' + rmsd + '\n'
            tm_data.append(data)
            rmsds.append(float(rmsd))
            tm_scores.append(float(tm_score))

    f = open(tm_dir + 'summary.txt', "w")
    f.write("Name\tTM Score\tRMSD\n")
    for data in tm_data:
        f.write(data)
    
    def avg(list_):
        avg = sum(list_) / len(list_)
        return str(round(avg, 4))

    tm_score_avg = avg(tm_scores)
    rmsd_avg = avg(rmsds)
    f.write("Average" + '\t' + tm_score_avg + '\t' + rmsd_avg + '\n')
    f.close()
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create PDB structure and calculate TM Score.")

    parser.add_argument('tf_record', default='.', help='Path to tf record')

    parser.add_argument('tertiary_dir',
                        default='.',
                        help='Directory that contains .tertiary files')

    parser.add_argument('pdb_dir',
                        default='.',
                        help='Directory to save PDB files')

    parser.add_argument(
        'tm_dir',
        default='.',
        help='Directory to save TM score files')

    parser.add_argument(
        '-t', '--tm_scores',
        action='store_true', 
        help='If set, create tm scores.')

    args = parser.parse_args()

    proteins = tf_record_to_dict(args.tf_record, args.tertiary_dir)

    create_pdb_files(proteins, args.pdb_dir)

    if args.tm_scores:
        create_TM_files(proteins, args.pdb_dir, args.tm_dir)
