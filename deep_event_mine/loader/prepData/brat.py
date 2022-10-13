"""Read brat format input files."""

import glob
import collections
from collections import OrderedDict


def brat_loader(files_fold, params):

    # list of txt files containing the text of the documents
    file_list = glob.glob(files_fold + '*' + '.txt')

    entities = OrderedDict()

    sentences = OrderedDict()

    for filef in sorted(file_list):
        # skip if file starts with "."
        if filef.split("/")[-1].startswith("."):
            continue

        filename = filef.split('/')[-1].split('.txt')[0]
        ffolder = '/'.join(filef.split('/')[:-1]) + '/'

        fentities = OrderedDict()

        # list of all entities ids e.g ['T1', 'T2', 'T3', 'T4',.....,'TR51', 'TR52']
        idsT = []

        # list of all entities label name
        typesT = []

        # dictionary of all entities information stored in a dictionary
        # e.g. for one entity
        # >>> infoT["T1"]
        # OrderedDict([('id', 'T1'), ('type', 'Gene_or_gene_product'), ('pos1', '0'), ('pos2', '7'), ('text', 'Ras p21')])
        infoT = OrderedDict()

        # same information as above but stored as list
        # >>> termsT[0]
        # ['T1', 'Gene_or_gene_product', '0', '7', 'Ras p21']
        termsT = []

        # open file with annotations for the current document
        with open(ffolder + filename + '.ann', encoding="UTF-8") as infile:
            # loop over all lines in the .ann file
            for line in infile:

                # Check if line starts with 'T' meaning that it's an entity or trigger
                if line.startswith('T'):

                    # e.g. ['T19', 'Gene_or_gene_product 480 483', 'ras']
                    line = line.rstrip().split('\t')

                    # entity id e.g T19
                    eid = line[0]

                    # list containing entity label, start and ending indexes
                    # e.g ['Gene_or_gene_product', '480', '483']
                    e1 = line[1].split()

                    # entity label e.g. Gene_or_gene_product
                    etype = e1[0]

                    # starting index e.g 480
                    pos1 = e1[1]

                    # ending index e.g 483
                    pos2 = e1[2]

                    # actual entity text e.g ras
                    text = line[2]

                    idsT.append(eid)
                    typesT.append(etype)
                    ent_info = OrderedDict()
                    ent_info['id'] = eid
                    ent_info['type'] = etype
                    ent_info['pos1'] = pos1
                    ent_info['pos2'] = pos2
                    ent_info['text'] = text
                    infoT[eid] = ent_info
                    termsT.append([eid, etype, pos1, pos2, text])

            # frequency of each entity in the document
            typesT2 = dict(collections.Counter(typesT))

            fentities['data'] = infoT
            fentities['types'] = typesT
            fentities['counted_types'] = typesT2
            fentities['ids'] = idsT
            fentities['terms'] = termsT

        # check empty entities
        if len(idsT) == 0 and not params['raw_text']:
            continue

        else:
            entities[filename] = fentities

            lowerc = params['lowercase']
            with open(ffolder + filename + '.txt', encoding="UTF-8") as infile:
                lines = []
                for line in infile:
                    line = line.strip()
                    if len(line) > 0:
                        if lowerc:
                            line = line.lower()
                        lines.append(line)
                sentences[filename] = lines

    return entities, sentences


