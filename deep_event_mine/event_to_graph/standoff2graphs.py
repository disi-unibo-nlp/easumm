# coding: utf-8

import pickle
import os
import networkx
import argparse
from deep_event_mine.event_to_graph import datautils


def base_root_name(filename: str) -> str:
    o, _ = os.path.splitext(os.path.basename(filename))
    return o


def events_folder(dataset: str) -> str:
    return f'experiments/{dataset}/results/ev-last/ev-tok-ann'


def entities_folder(dataset: str) -> str:
    return f'experiments/{dataset}/results/rel-last/rel-ann'


def a2_file_path(dataset: str, filename: str) -> str:
    return os.path.join(events_folder(dataset), filename + '.a2')


def emb_file_path(dataset: str, filename: str) -> str:
    return os.path.join(entities_folder(dataset), filename + '-EMB.json')


def preprocessed_text_path(dataset: str, filename: str) -> str:
    return f'data/{dataset}/processed-text/text/{filename}.txt'


def find_span_line(text: str, span_start: int, span_end: int) -> int:
    # Find the line containing the given span. If the spans is on multiple lines,
    # or it is outside the given text, returns None.
    if span_start >= len(text):
        return None

    def line_for(idx: int) -> int:
        return text.count('\n', 0, idx)

    start_line = line_for(span_start)
    end_line = line_for(span_end)
    if start_line != end_line:
        return None
    return start_line


def get_graphs(files_path, articles_ids):

    graphs_doc = {}
    datafiles = datautils.data_files(files_path)

    for datafile in datafiles:
        if datafile.base_name in articles_ids:
            entities, events = datautils.load_document(datafile)

            evs_graph = {}
            arguments_list = []
            for event in events.values():
                evs_graph[event.id] = {}
                args_dict = {}
                for argument, role in event.arguments:
                    if type(argument) is not datautils.StandoffEntity:
                        args_dict[argument.trigger.id] = {}
                        args_dict[argument.trigger.id]['role'] = role
                        args_dict[argument.trigger.id]['arguments'] = evs_graph[argument.id]['arguments']
                    else:
                        args_dict[argument.id] = {}
                        args_dict[argument.id]['role'] = role

                    arguments_list.append(argument.id)

                evs_graph[event.id]['arguments'] = args_dict
                evs_graph[event.id]['trigger'] = event.trigger.id

            arguments_set = set(arguments_list)

            all_graphs = []
            for ev_id, ev in evs_graph.items():

                if ev_id not in arguments_set:
                    graph = networkx.DiGraph(source_doc=datafile.base_name, dataset=files_path)

                    trig = entities[ev['trigger']]
                    graph.add_node(trig.id, type=trig.type, name=trig.name)

                    for argg_id, argg in ev['arguments'].items():
                        ent = entities[argg_id]

                        graph.add_node(ent.id, type=ent.type, name=ent.name)
                        graph.add_edge(trig.id, ent.id, key=argg['role'])

                        if 'arguments' in argg:
                            arg_nest1 = argg['arguments']

                            for arg_nest1_id, arg_nest1_obj in arg_nest1.items():
                                ent_nest1 = entities[arg_nest1_id]
                                graph.add_node(ent_nest1.id, type=ent_nest1.type, name=ent_nest1.name)
                                graph.add_edge(ent.id, ent_nest1.id, key=arg_nest1_obj['role'])

                                if 'arguments' in arg_nest1_obj:
                                    arg_nest2 = arg_nest1_obj['arguments']
                                    for arg_nest2_id, arg_nest2_obj in arg_nest2.items():

                                        ent_nest2 = entities[arg_nest2_id]
                                        graph.add_node(ent_nest2.id, type=ent_nest2.type, name=ent_nest2.name)
                                        graph.add_edge(ent_nest1.id, ent_nest2.id, key=arg_nest2_obj['role'])

                                    if 'arguments' in arg_nest2_obj:
                                        arg_nest3 = arg_nest2_obj['arguments']
                                        for arg_nest3_id, arg_nest3_obj in arg_nest3.items():
                                            ent_nest3 = entities[arg_nest3_id]
                                            graph.add_node(ent_nest3.id, type=ent_nest3.type, name=ent_nest3.name)
                                            graph.add_edge(ent_nest2.id, ent_nest3.id, key=arg_nest3_obj['role'])

                    graph.add_node('master_node')
                    for node in graph.nodes:
                        graph.add_edge('master_node', node, key='Other')
                    all_graphs.append(graph)

            graphs_doc[datafile.base_name] = all_graphs

    # for documents where we fail to extract graphs
    for artid in articles_ids:
        if artid not in graphs_doc:
            graphs_doc[artid] = []

    return graphs_doc


def get_single_graph(a2_file_path):

    entities, events = datautils.load_a2_file(a2_file_path)

    evs_graph = {}
    arguments_list = []
    for event in events.values():
        evs_graph[event.id] = {}
        args_dict = {}
        for argument, role in event.arguments:
            if type(argument) is not datautils.StandoffEntity:
                args_dict[argument.trigger.id] = {}
                args_dict[argument.trigger.id]['role'] = role
                args_dict[argument.trigger.id]['arguments'] = evs_graph[argument.id]['arguments']
            else:
                args_dict[argument.id] = {}
                args_dict[argument.id]['role'] = role

            arguments_list.append(argument.id)

        evs_graph[event.id]['arguments'] = args_dict
        evs_graph[event.id]['trigger'] = event.trigger.id

    arguments_set = set(arguments_list)

    all_graphs = []
    for ev_id, ev in evs_graph.items():

        if ev_id not in arguments_set:
            graph = networkx.DiGraph()

            trig = entities[ev['trigger']]
            graph.add_node(trig.id, type=trig.type, name=trig.name)

            for argg_id, argg in ev['arguments'].items():
                ent = entities[argg_id]

                graph.add_node(ent.id, type=ent.type, name=ent.name)
                graph.add_edge(trig.id, ent.id, key=argg['role'])

                if 'arguments' in argg:
                    arg_nest1 = argg['arguments']

                    for arg_nest1_id, arg_nest1_obj in arg_nest1.items():
                        ent_nest1 = entities[arg_nest1_id]
                        graph.add_node(ent_nest1.id, type=ent_nest1.type, name=ent_nest1.name)
                        graph.add_edge(ent.id, ent_nest1.id, key=arg_nest1_obj['role'])

                        if 'arguments' in arg_nest1_obj:
                            arg_nest2 = arg_nest1_obj['arguments']
                            for arg_nest2_id, arg_nest2_obj in arg_nest2.items():
                                ent_nest2 = entities[arg_nest2_id]
                                graph.add_node(ent_nest2.id, type=ent_nest2.type, name=ent_nest2.name)
                                graph.add_edge(ent_nest1.id, ent_nest2.id, key=arg_nest2_obj['role'])

                            if 'arguments' in arg_nest2_obj:
                                arg_nest3 = arg_nest2_obj['arguments']
                                for arg_nest3_id, arg_nest3_obj in arg_nest3.items():
                                    ent_nest3 = entities[arg_nest3_id]
                                    graph.add_node(ent_nest3.id, type=ent_nest3.type, name=ent_nest3.name)
                                    graph.add_edge(ent_nest2.id, ent_nest3.id, key=arg_nest3_obj['role'])

            graph.add_node('master_node')
            for node in graph.nodes:
                graph.add_edge('master_node', node, key='Other')
            all_graphs.append(graph)

    return all_graphs


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--files_path', type=str, required=True)
    args = parser.parse_args()

    graphs_doc = {}
    datafiles = datautils.data_files(args.files_path)

    for datafile in datafiles:
        print(datafile)
        entities, events = datautils.load_document(datafile)

        evs_graph = {}
        arguments_list = []
        for event in events.values():
            evs_graph[event.id] = {}
            args_dict = {}
            for argument, role in event.arguments:
                if type(argument) is not datautils.StandoffEntity:
                    args_dict[argument.trigger.id] = {}
                    args_dict[argument.trigger.id]['role'] = role
                    args_dict[argument.trigger.id]['arguments'] = evs_graph[argument.id]['arguments']
                else:
                    args_dict[argument.id] = {}
                    args_dict[argument.id]['role'] = role

                arguments_list.append(argument.id)

            evs_graph[event.id]['arguments'] = args_dict
            evs_graph[event.id]['trigger'] = event.trigger.id

        arguments_set = set(arguments_list)

        all_graphs = []
        for ev_id, ev in evs_graph.items():

            if ev_id not in arguments_set:
                graph = networkx.DiGraph(source_doc=datafile.base_name, dataset=args.files_path)

                trig = entities[ev['trigger']]
                graph.add_node(trig.id, type=trig.type, name=trig.name)

                for argg_id, argg in ev['arguments'].items():
                    ent = entities[argg_id]

                    graph.add_node(ent.id, type=ent.type, name=ent.name)
                    graph.add_edge(trig.id, ent.id, key=argg['role'])

                    if 'arguments' in argg:
                        arg_nest1 = argg['arguments']

                        for arg_nest1_id, arg_nest1_obj in arg_nest1.items():
                            ent_nest1 = entities[arg_nest1_id]
                            graph.add_node(ent_nest1.id, type=ent_nest1.type, name=ent_nest1.name)
                            graph.add_edge(ent.id, ent_nest1.id, key=arg_nest1_obj['role'])

                            if 'arguments' in arg_nest1_obj:
                                arg_nest2 = arg_nest1_obj['arguments']
                                for arg_nest2_id, arg_nest2_obj in arg_nest2.items():
                                    ent_nest2 = entities[arg_nest2_id]
                                    graph.add_node(ent_nest2.id, type=ent_nest2.type, name=ent_nest2.name)
                                    graph.add_edge(ent_nest1.id, ent_nest2.id, key=arg_nest2_obj['role'])

                                if 'arguments' in arg_nest2_obj:
                                    arg_nest3 = arg_nest2_obj['arguments']
                                    for arg_nest3_id, arg_nest3_obj in arg_nest3.items():
                                        ent_nest3 = entities[arg_nest3_id]
                                        graph.add_node(ent_nest3.id, type=ent_nest3.type, name=ent_nest3.name)
                                        graph.add_edge(ent_nest2.id, ent_nest3.id, key=arg_nest3_obj['role'])

                graph.add_node('master_node')
                for node in graph.nodes:
                    graph.add_edge('master_node', node, key='Other')
                all_graphs.append(graph)

        graphs_doc[datafile.base_name] = all_graphs

    print(f'Saving {len(graphs_doc)} graphs...')
    with open(f'{args.files_path}/graphs.pickle', 'wb') as ff:
        pickle.dump(graphs_doc, ff)


if __name__ == '__main__':
    main()

