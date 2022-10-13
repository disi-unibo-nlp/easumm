# coding: utf-8

from typing import List, Set, Tuple, Dict, Union
import os
from glob import glob


class DataFile:
    def __init__(self, root, filename):
        self.path = os.path.join(root, filename)
    @property
    def a1(self):
        return self.path + '.a1'
    @property
    def a2(self):
        return self.path + '.a2'
    @property
    def txt(self):
        return self.path + '.txt'
    @property
    def embeddings(self):
        return self.path + '-EMB.json'
    @property
    def base_name(self):
        return os.path.basename(self.path)

    def __repr__(self):
        return self.path


def data_files(dataset_root: str) -> List[DataFile]:
    def base_root_name(filename: str) -> str:
        return os.path.splitext(os.path.basename(filename))[0]
    def find_by_ext(ext: str) -> Set[str]:
        return set(map(base_root_name, glob(os.path.join(dataset_root, f'*.{ext}'))))

    a2_files = find_by_ext('a2')
    return [DataFile(dataset_root, f) for f in a2_files]


class StandoffEntity:
    def __init__(self, _id: str, _type: str, _span: Tuple[int, int], _name: str):
        self.id = _id
        self.type = _type
        self.span = _span
        self.name = _name

    def from_line(line: str) -> 'StandoffEntity':
        assert line[0] == 'T'

        if len(line.split('\t')) == 3:
            _id, _args, _name = line.split('\t')
        else:
            _id, _args, _name, _ = line.split('\t')

        _type, _span_start, _span_end = _args.split(' ')
        return StandoffEntity(_id, _type, (int(_span_start), int(_span_end)), _name)


class StandoffEvent:
    def __init__(self, _id: str, _trigger: StandoffEntity, _arguments: List[Tuple[Union[StandoffEntity, 'StandoffEvent'], str]]):
        self.id = _id
        self.trigger = _trigger
        self.arguments = _arguments

    def from_line(line: str, entities: Dict[str, StandoffEntity], events: Dict[str, 'StandoffEvent']):
        _id, _others = line.split('\t')
        [_, _trigger], *_arguments = [a.split(':') for a in _others.split()]
        resolved_args = [(entities[a[1]] if a[1] in entities else events[a[1]], a[0]) for a in _arguments]
        return StandoffEvent(_id, entities[_trigger], resolved_args)


def load_document(doc):

    with open(doc.a2) as a2_file:
        annotations = list(a2_file)

    entities = {}
    events = {}
    for line in annotations:
        if line[0] == 'T':
            ent = StandoffEntity.from_line(line)
            entities[ent.id] = ent
    
    repeat = True
    while repeat == True:
        repeat = False
        for line in annotations:
            if line[0] == 'E':
                try:
                    event = StandoffEvent.from_line(line, entities, events)
                    if event.id not in events:
                        events[event.id] = event
                except Exception as e:
                    repeat = True

    return entities, events


def load_a2_file(a2_file_path):
    with open(a2_file_path) as a2_file:
        annotations = list(a2_file)

    entities = {}
    events = {}
    for line in annotations:
        if line[0] == 'T':
            ent = StandoffEntity.from_line(line)
            entities[ent.id] = ent

    repeat = True
    while repeat == True:
        repeat = False
        for line in annotations:
            if line[0] == 'E':
                try:
                    event = StandoffEvent.from_line(line, entities, events)
                    if event.id not in events:
                        events[event.id] = event
                except Exception as e:
                    repeat = True

    return entities, events
