#!/usr/bin/env python
import argparse
import sys
import matplotlib.pyplot as plt


def wss(coflows, bandwidth, background_flow):
    # coflows are sorted
    coflow_in_progress = []
    id_2_coflow = dict([(coflow.coflow_id, coflow) for coflow in coflows])
    reducer_2_flow = {}
    for incoming_coflow in coflows:
        # current_time
        current_time = incoming_coflow.arrival_time
        # check if any flow completes
        global_min_flow, global_min_time = None, sys.maxint
        for _, flows_this_reducer in reducer_2_flow.items():
            related_coflows = [id_2_coflow.get(item.coflow_id)
                    for item in flows_this_reducer]
            related_coflow_sizes = [item.get_size() 
                    for item in related_coflows]
            total_size = sum(related_coflow_sizes)
            allocated_bandwidths = [bandwidth * size / total_size
                    for size in related_coflow_sizes]
            test_completion_time = [flow.test(bandwidth) for flow, bandwidth in
                    zip(flows_this_reducer, allocated_bandwidths)]
            min_flow, min_time = min(enumerate(test_completion_time),
                    key=lambda x: x[1])
            if min_time < global_min_time:
                global_min_flow, global_min_time = min_flow, min_time



        # enqueue
        coflow_in_progress.append(incoming_coflow)
        # arrange each flow to designated port
        for reducer, flow in zip(incoming_coflow.reducers,
                incoming_coflow.flows):
            flows_each_reducer = reducer_2_flow.get(reducer, [])
            flows_each_reducer.append(flow)
            reducer_2_flow[reducer] = flows_each_reducer


            

class flow:
    def __init__(self, size, start_time, coflow_id):
        self.status = 'w'
        self.size = size
        self.start_time = start_time
        self.current_time = start_time
        self.complete_time = None
        self.coflow_id = coflow_id

    def test(self, allocated_bandwidth):
        return self.size / allocated_bandwidth + self.current_time

    def run(self, allocated_bandwidth, run_until):
        self.size -= (run_until - self.current_time) * allocated_bandwidth
        self.current_time = run_until


class coflow:
    def __init__(self, line):
        split = line.split()
        self.coflow_id, arrival_time, num_mapper = split[:3]
        # ms
        self.arrival_time = int(arrival_time)
        num_mapper = int(num_mapper)
        self.num_mapper = num_mapper
        self.loc_mappers = split[3:3 + num_mapper]
        # size:MB
        reducer_size_pairs = [item.split(":")
                for item in split[4 + num_mapper:]]

        self.reducers = [item[0] for item in reducer_size_pairs]
        self.flows = [flow(float(item[1], self.arrival_time, self.coflow_id))
                for item in reducer_size_pairs]
        self.status = 'w'
        # w(aiting), r(unning), f(inish)

    def get_size(self):
        total_size = 0
        for flow in self.flows:
            total_size += flow.size
        return total_size
    
    def update_status(self, current_time):
        status = [item.status for item in self.flows]
        if 'w' not in status and 'r' not in status:
            if self.status != 'f':
                self.complete_time = current_time
                self.status = 'f'
        elif 'r' in status:
            if self.status != 'r':
                self.status = 'r'
                self.start_time = current_time




def getargs():
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', required=True, choices=['wss', 'fifo','sebf'])
    args = parser.parse_args()
    return args


def parse_file(fn):
    with open(fn) as f:
        content = f.readlines()
    num_ports, num_coflow = content[0].split()
    coflows = [coflow(line) for line in content[1:]]
    return coflows


def simulate(algorithm, coflows, background_flow=0.0, bandwidth=125.0):
    # default bandwidth 1Gbps
    if algorithm == 'wss':
        wss(coflows, bandwidth, background_flow)
    elif algorithm == 'fifo':
        fifo(coflows, bandwidth, background_flow)
    elif algorithm == 'sebf':
        sebf(coflows, bandwidth, background_flow)

    

def main():
    args = getargs()
    coflows = parse_file('./FB2010-1Hr-150-0.txt')
    sorted_coflows = sorted(coflows, key=lambda x: x.arrival_time)
    simulate(args.algorithm, sorted_coflows)


if __name__ == '__main__':
    main()
