#!/usr/bin/env python
import argparse
import sys
import matplotlib.pyplot as plt

# global variables
current_time = 0


class Simulator:
    def __init__(self):
        self.current_time = 0
        self.reducer_dict = {}
        self.coflow_sizes = {} 

    def simulate(self, algorithm, coflows,
            background_flow=0.0, bandwidth=125.0):
        # default bandwidth 1Gbps
        if algorithm == 'wss':
            self.wss(coflows, bandwidth, background_flow)
        elif algorithm == 'fifo':
            self.fifo(coflows, bandwidth, background_flow)
        elif algorithm == 'sebf':
            self.sebf(coflows, bandwidth, background_flow)

    def fifo(self, coflows, bandwidth, background_flow):
        bandwidth *= 1 - background_flow
        id_2_coflow = dict([(coflow.coflow_id, coflow) for coflow in coflows])
        id_2_priority = dict(zip(coflows.keys(), range(len(coflows.keys()))))
        heavy_size_2xavg = sum(map(lambda x: x.total_size, coflows)) / len(coflows) * 2
        proceeding_coflows = []

        time_period = [0]
        time_period.extend([coflow.arrival_time for coflow in coflows])
        last_tp = 0
        while time_period:
            tp = time_period.pop(0)
            next_tp = sys.maxsize
            # see if anyone arrives
            for coflow in coflows:
                if coflow.arrival_time == tp:
                    coflow.status = 'r'
                    for r_id, flow in coflow.reducers, coflow.flows:
                        self.reducer_dict[r_id].flows.append(flow)
                        self.reducer_dict[r_id].flows.sort(key=
                                lambda x: id_2_priority[x.coflow_id])
                        
            # see if any flow will complete
            # see if any flow will become heavy

            last_tp = tp
        '''
        for incoming_coflow in coflows:
            tp = incoming_coflow.arrival_time
            # check every port and allocate bandwidth
            for r_id, reducer in self.reducer_dict.items():
                # assign flows with bandwidth
                reducer.flows.sort(key=lambda x: id_2_priority[x.coflow_id])
                # check how many flows are allowed to transfer
                allowed_flows = []
                for flow in reducer.flows:
                    if id_2_coflow[flow.coflow_id].fifo_is_heavy and flow.status != 'f':
                        allowed_flows.append(flow)
                        continue
                    allowed_flows.append(flow)
                    break
                for flow in allowed_flows:
                    flow.status = 'r'
                    flow.allocated_bandwidth = bandwidth / len(allowed_flows)

            # iterate through coflows to see if anyone completes before the next coflow comes
            estimate_heavy_times = [item.estimate_time_to_heavy(heavy_size_2xavg) for item in coflows]
        '''

    def wss(self, coflows, bandwidth, background_flow):
        # coflows are sorted
        bandwidth *= 1 - background_flow
        id_2_coflow = dict([(coflow.coflow_id, coflow) for coflow in coflows])
        self.reducer_dict = {}
        for incoming_coflow in coflows:
            tp = incoming_coflow.arrival_time
            # see if anything completes before new coflow arrives
            # iterate through reducers and calculate completion time for each one
            for r_id, reducer in self.reducer_dict.items():
                reducer_total_size = sum([item.size for item in reducer.flows])
                if reducer_total_size > bandwidth * (tp - self.current_time):
                    # not complete
                    # * (reducer_total_size - bandwidth * (tp - self.current_time)) / reducer_total_size
                    for item in reducer.flows:
                        item.size *= (reducer_total_size - bandwidth * (tp - self.current_time)) / reducer_total_size 
                else:
                    # complete
                    complete_time = self.current_time + reducer_total_size / bandwidth
                    for item in reducer.flows:
                        item.complete_time = complete_time
                        item.status = 'f'
                        related_coflow = id_2_coflow[item.coflow_id]
                        related_coflow.update_status(complete_time)
            # add incoming flows to related reducers
            for r_id, flow in zip(incoming_coflow.reducers, incoming_coflow.flows):
                reducer = self.reducer_dict.get(r_id, Reducer())
                reducer.add_flow(flow)
                self.reducer_dict[r_id] = reducer

        for r_id, reducer in self.reducer_dict.items():
            reducer_total_size = sum([item.size for item in reducer.flows])
            # complete
            complete_time = self.current_time + reducer_total_size / bandwidth
            for item in reducer.flows:
                item.complete_time = complete_time
                item.status = 'f'
                related_coflow = id_2_coflow[item.coflow_id]
                related_coflow.update_status(complete_time)
        # debug
        # for c_id, coflow in id_2_coflow.items():
        #     print(c_id, coflow.total_size, coflow.complete_time)
        average_cct = sum(map(lambda x: x[1].complete_time, id_2_coflow.items())) / len(id_2_coflow.items())
        print('avg_cct: ', average_cct)



class Reducer:
    def __init__(self):
        self.flows = []

    def add_flow(self, flow):
        self.flows.append(flow)


            

class flow:
    def __init__(self, size, start_time, coflow_id):
        self.status = 'w'
        self.size = size
        self.start_time = start_time
        self.current_time = start_time
        self.complete_time = None
        self.coflow_id = coflow_id
        self.allocated_bandwidth = 0


class coflow:
    def __init__(self, line):
        split = line.split()
        self.coflow_id, arrival_time, num_mapper = split[:3]
        # ms
        self.arrival_time = int(arrival_time)
        num_mapper = int(num_mapper)
        self.num_mapper = num_mapper
        self.loc_mappers = split[3:3 + num_mapper]
        self.fifo_is_heavy = False
        # size:MB
        reducer_size_pairs = [item.split(":")
                for item in split[4 + num_mapper:]]

        self.reducers = [item[0] for item in reducer_size_pairs]
        self.flows = [flow(float(item[1]), self.arrival_time, self.coflow_id)
                for item in reducer_size_pairs]
        self.status = 'w'
        self.total_size = sum(map(lambda x: x.size, self.flows))
        # w(aiting), r(unning), f(inish)

    def get_min_flow_completion_time(self):
        min_time = sys.maxsize
        for flow in self.flows:
            if flow.status == 'r':
                time = flow.size / flow.allocated_bandwidth
                min_time = if min_time > time then time else min_time
        return min_time


    def get_total_bandwidth(self):
        return sum(map(lambda x: x.allocated_bandwidth, self.flows))

    def estimate_time_to_heavy(self, heavy_size):
        return (heavy_size - self.total_size + self.get_size()) / self.get_total_bandwidth()

    def get_size(self):
        total_size = 0
        for flow in self.flows:
            total_size += flow.size
        return total_size

    def change_status(self, status):
        self.status = status
    
    def update_status(self, current_time):
        if current_time >= self.arrival_time and self.status == 'w':
            self.status = 'r'
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



    

def main():
    args = getargs()
    coflows = parse_file('./FB2010-1Hr-150-0.txt')
    sorted_coflows = sorted(coflows, key=lambda x: x.arrival_time)
    simulator = Simulator()
    simulator.simulate(args.algorithm, sorted_coflows)


if __name__ == '__main__':
    main()
