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
        id_2_priority = dict(zip(id_2_coflow.keys(), range(len(coflows))))
        heavy_size_2xavg = sum(map(lambda x: x.total_size, coflows)) / len(coflows) * 2
        proceeding_coflows = []
        self.reducer_dict = {}

        time_period = [0]
        time_period.extend([coflow.arrival_time for coflow in coflows])

        coflow_status = [cf.status for cf in coflows]
        time = 0
        time_step = 100
        next_coflow = coflows.pop(0)
        arrived_coflows = 0
        all_coflows = coflows.copy()
        while 'r' in coflow_status or 'w' in coflow_status:
            print("time", time)
            print("arrived", arrived_coflows)
            print('running', len([item for item in coflow_status if item == 'r']))
            print("finished", len([item for item in coflow_status if item == 'f']))
            if next_coflow != None and time >= next_coflow.arrival_time:
                arrived_coflows += 1
                # coflow arrives
                # update reducers
                for r_id, flow in zip(next_coflow.reducers, next_coflow.flows):
                    reducer = self.reducer_dict.get(r_id, Reducer())
                    self.reducer_dict[r_id] = reducer
                    reducer.flows.append(flow)
                    reducer.flows.sort(key=lambda x: id_2_priority[flow.coflow_id])
                # update next coflow
                next_coflow = coflows.pop(0) if coflows else None
            # update flow size and status, for the next 1ms
            for _, reducer in self.reducer_dict.items():
                # allocate bandwidth
                allowed_flows = []
                reducer.flows = [flow for flow in reducer.flows if flow.status != 'f']
                if not reducer.flows:
                    continue
                for flow in reducer.flows:
                    if id_2_coflow[flow.coflow_id].fifo_is_heavy:
                        allowed_flows.append(flow)
                        continue
                    allowed_flows.append(flow)
                    break
                for flow in allowed_flows:
                    flow.status = 'r'
                    flow.allocated_bandwidth = bandwidth / len(allowed_flows)
                    flow.size -= flow.allocated_bandwidth * time_step * 0.001
                    id_2_coflow[flow.coflow_id].status = 'r'
                    if flow.size <= 0:
                        flow.status = 'f'
            # update coflow status
            for coflow in all_coflows:
                if coflow.total_size - heavy_size_2xavg >= coflow.get_size():
                    coflow.fifo_is_heavy = True
                flow_status = [flow.status for flow in coflow.flows]
                if 'w' not in flow_status and 'r' not in flow_status:
                    if coflow.status != 'f':
                        coflow.status = 'f'
                        coflow.complete_time = time
            coflow_status = [coflow.status for coflow in all_coflows]
            # print(coflow_status)
            time += time_step
        average_cct = sum([coflow.get_cct() for
            coflow in all_coflows]) / len(all_coflows)
        print(average_cct)

    def sebf(self, coflows, bandwidth, background_flow):
        bandwidth *= 0.001
        usable_bandwidth = bandwidth * (1 - background_flow)
        id_2_coflow = dict([(coflow.coflow_id, coflow) for coflow in coflows])
        current_time = 0
        reducer_dict = {}
        all_coflows = coflows.copy()
        incoming_coflow = coflows.pop(0)
        T = 1000
        sig = 100
        running_coflow_id = []
        last_sigma_arrived_coflow_list = []
        # 1. rank 1 coflow completes
        # 2. new coflow arrives

        tp = [incoming_coflow.arrival_time]

        def reschedule(coflow_id_list):
            coflow_id_list = [cid for cid in coflow_id_list
                    if id_2_coflow[cid].status != 'f']
            bottlenecks = [id_2_coflow[cid].get_bottleneck()
                    for id in coflow_id_list]
            sorted_id_n_bottleneck = sorted(zip(coflow_id_list, bottlenecks),
                    key=lambda x: x[1])
            id_2_priority = [(i, cid) for i, cid
                    in enumerate(sorted_id_n_bottleneck)]
            id_2_tl = [(cid, bot / bandwidth) for cid, bot in sorted_id_n_bottleneck]
            not_scheduled_cid = set()
            for _, reducer in reducer_dict:
                band_left = usable_bandwidth
                reducer.flows.sort(key=lambda x: id_2_priority[x.coflow_id])
                for flow in reducer.flows:
                    tl = id_2_tl[flow.coflow_id]
                    madd_bandwidth = flow.size / tl
                    if madd_bandwidth < band_left:
                        band_left -= madd_bandwidth
                        flow.allocated_bandwidth = madd_bandwidth
                    else:
                        flow.allocated_bandwidth = band_left
                        if band_left == 0:
                            not_scheduled_cid.add(flow.coflow_id)
                        band_left = 0
            return not_scheduled_cid

        coflow_id_list = []
        while tp:
            # check status
            is_new_coflow_arrival = tp == incoming_coflow.arrival_time
            is_flow_completion = False
            is_in_T = tp % (T + sig) < T
            is_in_s = not is_in_T
            related_coflow_id = None
            for _, reducer in reducer_dict.items():
                for flow in reducer.flows:
                    if flow.status = 'f':
                        is_flow_completion = True
                        related_coflow_id = flow.coflow_id
                        break
            # update all status
            if is_new_coflow_arrival:
                coflow_id_list.append(incoming_coflow.coflow_id)
                incoming_coflow.status = 'w'
                # allocate bandwidth for each flow
                reschedule(coflow_id_list)
                

            # update tp


    def wss(self, coflows, bandwidth, background_flow):
        # coflows are sorted
        bandwidth *= 1 - background_flow
        bandwidth *= 0.001
        id_2_coflow = dict([(coflow.coflow_id, coflow) for coflow in coflows])
        self.reducer_dict = {}
        self.current_time = 0
        for incoming_coflow in coflows:
            tp = incoming_coflow.arrival_time
            print(tp)
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
            self.current_time = tp

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
        for c_id, coflow in id_2_coflow.items():
            print(c_id, coflow.arrival_time, coflow.complete_time)
        average_cct = sum(map(lambda x: x[1].get_cct(), id_2_coflow.items())) / len(id_2_coflow.items())
        print('avg_cct: ', average_cct)




class Reducer:
    def __init__(self):
        self.flows = []

    def add_flow(self, flow):
        self.flows.append(flow)


            

class flow:
    def __init__(self, size, start_time, coflow_id):
        self.status = 'r'
        self.size = size
        self.complete_time = None
        self.coflow_id = coflow_id
        self.allocated_bandwidth = 0

    def run_from_to(self, current_time, end_time):
        if self.status == 'f':
            return
        max_size = self.allocated_bandwidth * (end_time - current_time)
        if max_size > self.size:
            self.complete_time = (current_time +
                    self.size / self.allocated_bandwidth)
            self.status = 'f'
        else:
            self.size -= max_size


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

    def get_cct(self):
        return self.complete_time - self.arrival_time

    def get_bottleneck(self):
        return max([item.size for item in self.coflows])

    def get_min_flow_completion_time(self):
        min_time = sys.maxsize
        for flow in self.flows:
            if flow.status == 'r':
                time = flow.size / flow.allocated_bandwidth
                min_time = time if min_time > time else min_time
        return min_time


    def get_total_bandwidth(self):
        return sum(map(lambda x: x.allocated_bandwidth, self.flows))

    def estimate_time_to_heavy(self, heavy_size):
        if self.get_total_bandwidth() == 0:
            return sys.maxsize
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
