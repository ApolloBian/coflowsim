#!/usr/bin/env python
import argparse
import sys
import matplotlib.pyplot as plt

# global variables
current_time = 0

def get_average(a_list):
    return sum(a_list) / len(a_list)

def get_percentile(a_list, percent=0.8):
    order = sorted(a_list)
    index = round(len(a_list) * percent - 0.5)
    return order[index]

class Simulator:
    def __init__(self):
        self.current_time = 0
        self.reducer_dict = {}
        self.coflow_sizes = {} 

    def simulate(self, algorithm, coflows,
            background_flow=0.0, bandwidth=125.0):
        # default bandwidth 1Gbps
        if algorithm == 'wss':
            return self.wss(coflows, bandwidth, background_flow)
        elif algorithm == 'fifo':
            # self.fifo(coflows, bandwidth, background_flow)
            return self.fifo_new(coflows, bandwidth, background_flow)
        elif algorithm == 'sebf':
            return self.sebf(coflows, bandwidth, background_flow)
        elif algorithm == 'sebf_rr':
            return self.sebf_response_ratio(coflows, bandwidth, background_flow)

    def fifo(self, coflows, bandwidth, background_flow):
        bandwidth *= 0.001
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
            # print("time", time)
            # print("arrived", arrived_coflows)
            # print('running', len([item for item in coflow_status if item == 'r']))
            # print("finished", len([item for item in coflow_status if item == 'f']))
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
        ccts = [coflow.get_cct() for coflow in all_coflows]
        print('avg cct', get_average(ccts))
        print('80 per', get_percentile(ccts, 0.8))

    def fifo_new(self, coflows, bandwidth, background_flow):
        bandwidth *= 0.001
        bandwidth *= 1 - background_flow
        id_2_coflow = dict([(coflow.coflow_id, coflow) for coflow in coflows])
        id_2_priority = dict(zip(id_2_coflow.keys(), range(len(coflows))))
        heavy_size_2xavg = sum(map(lambda x: x.total_size, coflows)) / len(coflows) * 2
        proceeding_coflows = []
        self.reducer_dict = {}
        all_coflows = coflows.copy()

        # loop states
        incoming_coflow = all_coflows.pop(0)
        tp = incoming_coflow.arrival_time
        is_new_coflow_arrival = True
        is_flow_completion = False
        is_coflow_completion = False
        is_coflow_heavy = False
        heavy_coflow_id = None
        state = 'coflow_arrival'
        flow_2_complete = None

        def reschedule():
            coflow_id_list = [cid for cid, cf in id_2_coflow.items()
                    if cf.status != 'f']
            for _, reducer in self.reducer_dict.items():
                # allocate bandwidth
                allowed_flows = []
                reducer.flows = [flow for flow in reducer.flows if flow.status != 'f']
                if not reducer.flows:
                    continue
                reducer.flows.sort(key=lambda x: id_2_priority[x.coflow_id])
                for flow in reducer.flows:
                    if id_2_coflow[flow.coflow_id].fifo_is_heavy:
                        allowed_flows.append(flow)
                        continue
                    allowed_flows.append(flow)
                    break
                for flow in allowed_flows:
                    flow.allocated_bandwidth = bandwidth / len(allowed_flows)
        
        while True:
            # print(tp)
            # print(state)
            # strip finished flows
            for _, reducer in self.reducer_dict.items():
                reducer.flows = [i for i in reducer.flows if i.status != 'f']

            # update all status
            if state == 'coflow_arrival':
                # update_reducers
                for r_id, flow in zip(incoming_coflow.reducers, incoming_coflow.flows):
                    reducer = self.reducer_dict.get(r_id, Reducer())
                    self.reducer_dict[r_id] = reducer
                    reducer.flows.append(flow)
                    reducer.flows.sort(key=lambda x: id_2_priority[flow.coflow_id])
                incoming_coflow = all_coflows.pop(0) if all_coflows else None
            elif state == 'flow_completion':
                # if coflow is complete ,set the status
                for _, cf in id_2_coflow.items():
                    if cf.status != 'f' and cf.get_completion_status() == 'f':
                        cf.status = 'f'
                        cf.complete_time = tp
                running_coflows = [cf for _, cf in id_2_coflow.items()
                        if cf.status != 'f']
                if not running_coflows:
                    break

            elif state == 'coflow_heavy':
                heavy_coflow = id_2_coflow[heavy_coflow_id]
                heavy_coflow.fifo_is_heavy = True


            reschedule()


            # calculate next tp
            next_tp = sys.maxsize
            # 1. flow completion
            for _, reducer in self.reducer_dict.items():
                for flow in reducer.flows:
                    if flow.allocated_bandwidth:
                        time = flow.size / flow.allocated_bandwidth + tp
                        if next_tp > time:
                            flow_2_complete = flow
                            next_tp = time
                            state = 'flow_completion'

            # 2. incoming coflow
            if incoming_coflow:
                arrival = incoming_coflow.arrival_time
                if next_tp > arrival:
                    next_tp = arrival
                    state = 'coflow_arrival'
            # 3. coflow becomes heavy
            heavy_coflow_id = None
            for cid, coflow in id_2_coflow.items():
                if coflow.status != 'f':
                    time = coflow.estimate_time_to_heavy(heavy_size_2xavg) + tp
                    if next_tp > time and time > tp:
                        next_tp = time
                        heavy_coflow_id = cid
                        state = 'coflow_heavy'

            if next_tp == sys.maxsize:
                break

            # update till next tp
            for _, reducer in self.reducer_dict.items():
                for flow in reducer.flows:
                    flow.run_from_to(tp, next_tp)
            tp = next_tp
        ccts = [item.get_cct() for _, item in id_2_coflow.items()]
        return ccts



    def sebf_response_ratio(self, coflows, bandwidth, background_flow):
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
        # 1. coflow completes
        # 2. new coflow arrives
        # 3. when responce_ratio becomes high


        def rearrange_bandwidth(id_2_priority, id_2_tl):
            for _, reducer in reducer_dict.items():
                band_left = usable_bandwidth
                # special condition
                # if p1 bottleneck and p1.5 bottleneck
                # 1/size, size is of bottleneck
                if not reducer.flows:
                    continue
                hpid = reducer.flows[0].coflow_id
                hppriority = id_2_priority.get(hpid, None)
                is_p1 = hppriority == 1
                if is_p1:
                    p1coflow = id_2_coflow[hpid]
                    is_p1_bottleneck = reducer.flows[0].size == p1coflow.get_bottleneck()
                    if is_p1_bottleneck:
                        def get_flow_priority(flow):
                            coflowid = flow.coflow_id
                            return id_2_priority[coflowid]
                        p1fakelist = [flow for flow in reducer.flows
                                if get_flow_priority(flow) == 1.5]
                        if p1fakelist:
                            p1fakebottlenecks = [flow for flow in p1fakelist
                                    if flow.size == id_2_coflow[flow.coflow_id].get_bottleneck()]
                            p1fakebottlenecks.append(reducer.flows[0])
                            total_weights = sum(map(lambda x: 1 / x.size, p1fakebottlenecks))
                            for flow in p1fakebottlenecks:
                                flow.allocated_bandwidth = band_left / (flow.size * total_weights)
                            continue
                # else
                reducer.flows = [item for item in reducer.flows if item.coflow_id in id_2_priority]
                reducer.flows.sort(key=lambda x: id_2_priority[x.coflow_id])
                for flow in reducer.flows:
                    tl = id_2_tl[flow.coflow_id]
                    madd_bandwidth = flow.size / tl
                    if madd_bandwidth < band_left:
                        band_left -= madd_bandwidth
                        flow.allocated_bandwidth = madd_bandwidth
                    else:
                        flow.allocated_bandwidth = band_left
                        band_left = 0

        def reschedule(coflow_id_list, heavy_id_2_responce_ratio):
            coflow_id_list = [cid for cid in coflow_id_list
                    if id_2_coflow[cid].status != 'f']
            bottlenecks = [id_2_coflow[cid].get_bottleneck()
                    for cid in coflow_id_list]
            sorted_id_n_bottleneck = sorted(zip(coflow_id_list, bottlenecks),
                    key=lambda x: x[1])
            id_2_priority = dict([(i, cid) for i, cid
                    in sorted_id_n_bottleneck])
            # assign heavy ids with priority 1.5
            for cid in heavy_id_2_responce_ratio.keys():
                if id_2_priority[cid] != 1:
                    id_2_priority[cid] = 1.5
            id_2_tl = dict([(cid, bot / bandwidth) for cid, bot in sorted_id_n_bottleneck])
            not_scheduled_cid = set()
            rearrange_bandwidth(id_2_priority, id_2_tl)
            return id_2_priority, id_2_tl

        coflow_id_list = []
        flow_2_complete = None
        tp = incoming_coflow.arrival_time
        id_2_priority = dict()
        id_2_tl = dict()
        count = 0
        heavy_id_2_responce_ratio = {}
        heavy_id_proposal = None
        while True:
            # print("tp:", tp)
            # check status
            if incoming_coflow:
                is_new_coflow_arrival = tp == incoming_coflow.arrival_time
            else:
                is_new_coflow_arrival = False
            is_flow_completion = False
            is_in_T = tp % (T + sig) < T
            is_in_s = not is_in_T
            is_coflow_completion = True

            related_coflow_id = None
            for _, reducer in reducer_dict.items():
                for flow in reducer.flows:
                    if flow.status == 'f':
                        is_flow_completion = True
                        related_coflow_id = flow.coflow_id
                        cf = id_2_coflow[related_coflow_id]
                        flow_status = [item.status for item in cf.flows]
                        if 'r' in flow_status:
                            is_coflow_completion = False
                        else:
                            cf.status = 'f'
                            cf.complete_time = tp
                        break
            # print(is_new_coflow_arrival)
            # print(is_flow_completion)
            # update all status
            if is_new_coflow_arrival:
                # print("coflow_arrival")
                coflow_id_list.append(incoming_coflow.coflow_id)
                incoming_coflow.status = 'w'
                # assign flows to reducers
                for r_id, flow in zip(incoming_coflow.reducers, incoming_coflow.flows):
                    reducer = reducer_dict.get(r_id, Reducer())
                    reducer_dict[r_id] = reducer
                    reducer.flows.append(flow)
                # allocate bandwidth for each flow
                id_2_priority, id_2_tl = reschedule(coflow_id_list,
                       heavy_id_2_responce_ratio)
                # print(id_2_priority, id_2_tl)
                # calculate next tp
                # 1. flow completion
                next_tp = sys.maxsize
                for _, reducer in reducer_dict.items():
                    for flow in reducer.flows:
                        if flow.status == 'f':
                            continue
                        elif flow.allocated_bandwidth:
                            time = flow.size / flow.allocated_bandwidth + tp
                            if next_tp > time:
                                flow_2_complete = flow
                                next_tp = time
                # 2. incoming coflow
                incoming_coflow = coflows.pop(0) if coflows else None
                if incoming_coflow:
                    next_tp = min(incoming_coflow.arrival_time, next_tp)
                if next_tp == sys.maxsize:
                    break
                # 3. some flow becomes heavy
                heavy_threshold = len(id_2_priority)
                for cid in id_2_priority.keys():
                    coflow = id_2_coflow[cid]
                    # estimate the time until it gets heavy
                    if coflow.begin_time == None:
                        # only those haven't started could be heavy
                        time = coflow.arrival_time + coflow.total_size * (heavy_threshold - 1) / bandwidth
                        # ( > current_time
                        if time <= tp:
                            continue
                        if next_tp > time:
                            next_tp = time
                            heavy_id_proposal = cid

            elif is_flow_completion:
                if is_coflow_completion:
                    id_2_priority, id_2_tl = reschedule(coflow_id_list,
                            heavy_id_2_responce_ratio)
                else:
                    rearrange_bandwidth(id_2_priority, id_2_tl)
                # calculate next tp
                # 1. flow completion
                next_tp = sys.maxsize
                for _, reducer in reducer_dict.items():
                    for flow in reducer.flows:
                        if flow.status == 'f':
                            continue
                        elif flow.allocated_bandwidth:
                            time = flow.size / flow.allocated_bandwidth + tp
                            if next_tp > time:
                                flow_2_complete = flow
                                next_tp = time
                # 2. incoming coflow
                if incoming_coflow:
                    next_tp = min(incoming_coflow.arrival_time, next_tp)
                if next_tp == sys.maxsize:
                    break
                # 3. some flow becomes heavy
                heavy_threshold = len(id_2_priority)
                for cid in id_2_priority.keys():
                    coflow = id_2_coflow[cid]
                    # estimate the time until it gets heavy
                    if coflow.begin_time == None:
                        # only those haven't started could be heavy
                        time = coflow.arrival_time + coflow.total_size * (heavy_threshold - 1) / bandwidth
                        # ( > current_time
                        if time <= tp:
                            continue
                        if next_tp > time:
                            next_tp = time
                            heavy_id_proposal = cid
            else:
                # is coflow becomes heavy
                # handling logics
                # check if heavy_id_proposal is heavy
                alternative_heavy_coflow = id_2_coflow[heavy_id_proposal]
                rr = alternative_heavy_coflow.get_responce_ratio_at(tp, bandwidth)
                heavy_threshold = len(id_2_priority)
                if rr == heavy_threshold:
                    # it is heavy now
                    # append to heavy dict
                    heavy_id_2_responce_ratio[heavy_id_proposal] = rr
                    id_2_priority, id_2_tl = reschedule(coflow_id_list,
                            heavy_id_2_responce_ratio)
                # end handling logics
                # calculate next tp
                # 1. flow completion
                next_tp = sys.maxsize
                for _, reducer in reducer_dict.items():
                    for flow in reducer.flows:
                        if flow.status == 'f':
                            continue
                        elif flow.allocated_bandwidth:
                            time = flow.size / flow.allocated_bandwidth + tp
                            if next_tp > time:
                                flow_2_complete = flow
                                next_tp = time
                # 2. incoming coflow
                if incoming_coflow:
                    next_tp = min(incoming_coflow.arrival_time, next_tp)
                if next_tp == sys.maxsize:
                    break
                # 3. some flow becomes heavy
                heavy_threshold = len(id_2_priority)
                for cid in id_2_priority.keys():
                    coflow = id_2_coflow[cid]
                    # estimate the time until it gets heavy
                    if coflow.begin_time == None:
                        # only those haven't started could be heavy
                        time = coflow.arrival_time + coflow.total_size * (heavy_threshold - 1) / bandwidth
                        # ( > current_time
                        if time <= tp:
                            continue
                        if next_tp > time:
                            next_tp = time
                            heavy_id_proposal = cid

            # update till next tp
            for _, reducer in reducer_dict.items():
                for flow in reducer.flows:
                    flow.run_from_to(tp, next_tp)
                    # update coflow status
                    cf = id_2_coflow.get(flow.coflow_id)
                    if cf.begin_time == None and flow.allocated_bandwidth > 0:
                        cf.begin_time = tp
                        cf.status = 'r'

            tp = next_tp
            count += 1
            if count == 350:
                continue
        ccts = [coflow.get_cct() for _, coflow in id_2_coflow.items()]
        return ccts



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
        def rearrange_bandwidth(id_2_priority, id_2_tl):
            for rid, reducer in reducer_dict.items():
                band_left = usable_bandwidth
                reducer.flows = [item for item in reducer.flows if item.coflow_id in id_2_priority]
                reducer.flows.sort(key=lambda x: id_2_priority[x.coflow_id])
                for flow in reducer.flows:
                    # tl = id_2_tl[flow.coflow_id]
                    # madd_bandwidth = flow.size / tl
                    cf = id_2_coflow[flow.coflow_id]
                    rid_2_madd_speed = cf.rid_2_madd_speed
                    madd_bandwidth = rid_2_madd_speed[rid]
                    if madd_bandwidth < band_left:
                        band_left -= madd_bandwidth
                        flow.allocated_bandwidth = madd_bandwidth
                    else:
                        flow.allocated_bandwidth = band_left
                        band_left = 0

        def reschedule(coflow_id_list):
            coflow_id_list = [cid for cid in coflow_id_list
                    if id_2_coflow[cid].status != 'f']
            bottlenecks = [id_2_coflow[cid].get_bottleneck()
                    for cid in coflow_id_list]
            sorted_id_n_bottleneck = sorted(zip(coflow_id_list, bottlenecks),
                    key=lambda x: x[1])
            id_2_priority = dict([(i, cid) for i, cid
                    in sorted_id_n_bottleneck])
            id_2_tl = dict([(cid, bot / bandwidth) for cid, bot in sorted_id_n_bottleneck])
            not_scheduled_cid = set()
            for cid in coflow_id_list:
                tl = id_2_tl[cid]
                cf = id_2_coflow[cid]
                cf.madd_speeds = [item.size / tl for item in cf.flows]
                cf.rid_2_madd_speed = dict([(rid, item.size / tl)
                    for rid, item in zip(cf.reducers, cf.flows)])
            rearrange_bandwidth(id_2_priority, id_2_tl)
            return id_2_priority, id_2_tl

        coflow_id_list = []
        flow_2_complete = None
        tp = incoming_coflow.arrival_time
        id_2_priority = dict()
        id_2_tl = dict()
        count = 0
        while True:
            # print("tp:", tp)
            # check status
            if incoming_coflow:
                is_new_coflow_arrival = tp == incoming_coflow.arrival_time
            else:
                is_new_coflow_arrival = False
            is_flow_completion = False
            is_in_T = tp % (T + sig) < T
            is_in_s = not is_in_T
            is_coflow_completion = True

            related_coflow_id = None
            for _, reducer in reducer_dict.items():
                for flow in reducer.flows:
                    if flow.status == 'f':
                        is_flow_completion = True
                        related_coflow_id = flow.coflow_id
                        cf = id_2_coflow[related_coflow_id]
                        flow_status = [item.status for item in cf.flows]
                        if 'r' in flow_status:
                            is_coflow_completion = False
                        else:
                            cf.status = 'f'
                            cf.complete_time = tp

                        break

            # strip completed flows
            for _, reducer in reducer_dict.items():
                reducer.flows = [item for item in reducer.flows if item.status != 'f']
            # update all status
            if is_new_coflow_arrival:
                # print("coflow_arrival")
                coflow_id_list.append(incoming_coflow.coflow_id)
                incoming_coflow.status = 'w'
                # assign flows to reducers
                for r_id, flow in zip(incoming_coflow.reducers, incoming_coflow.flows):
                    reducer = reducer_dict.get(r_id, Reducer())
                    reducer_dict[r_id] = reducer
                    reducer.flows.append(flow)
                # allocate bandwidth for each flow
                id_2_priority, id_2_tl = reschedule(coflow_id_list)
                # print(id_2_priority, id_2_tl)
                # calculate next tp
                # 1. flow completion
                next_tp = sys.maxsize
                for _, reducer in reducer_dict.items():
                    for flow in reducer.flows:
                        if flow.status == 'f':
                            continue
                        elif flow.allocated_bandwidth:
                            time = flow.size / flow.allocated_bandwidth + tp
                            if next_tp > time:
                                flow_2_complete = flow
                                next_tp = time
                # 2. incoming coflow
                incoming_coflow = coflows.pop(0) if coflows else None
                if incoming_coflow:
                    next_tp = min(incoming_coflow.arrival_time, next_tp)
                if next_tp == sys.maxsize:
                    break
            elif is_flow_completion:
                if is_coflow_completion:
                    id_2_priority, id_2_tl = reschedule(coflow_id_list)
                else:
                    # print('flow')
                    # pass
                    rearrange_bandwidth(id_2_priority, id_2_tl)
                # calculate next tp
                # 1. flow completion
                next_tp = sys.maxsize
                for _, reducer in reducer_dict.items():
                    for flow in reducer.flows:
                        if flow.status == 'f':
                            continue
                        elif flow.allocated_bandwidth:
                            time = flow.size / flow.allocated_bandwidth + tp
                            if next_tp > time:
                                flow_2_complete = flow
                                next_tp = time
                # 2. incoming coflow
                if incoming_coflow:
                    next_tp = min(incoming_coflow.arrival_time, next_tp)
                if next_tp == sys.maxsize:
                    break

            # update till next tp
            for _, reducer in reducer_dict.items():
                for flow in reducer.flows:
                    flow.run_from_to(tp, next_tp)
            tp = next_tp
        ccts = [item.get_cct() for _, item in id_2_coflow.items()]
        return ccts


    def wss(self, coflows, bandwidth, background_flow):
        # coflows are sorted
        bandwidth *= 1 - background_flow
        bandwidth *= 0.001
        id_2_coflow = dict([(coflow.coflow_id, coflow) for coflow in coflows])
        self.reducer_dict = {}
        self.current_time = 0
        for incoming_coflow in coflows:
            tp = incoming_coflow.arrival_time
            # print(tp)
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
        # for c_id, coflow in id_2_coflow.items():
        #     print(c_id, coflow.arrival_time, coflow.complete_time)
        ccts = [item.get_cct() for _, item in id_2_coflow.items()]
        # average_cct = sum(map(lambda x: x[1].get_cct(), id_2_coflow.items())) / len(id_2_coflow.items())
        # print('avg_cct: ', average_cct)
        return ccts




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
        if max_size >= self.size - 1e-10:
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
        self.begin_time = None
        self.madd_speeds = None

    def get_responce_ratio_at(self, current_time, global_max_bandwidth):
        if current_time < self.arrival_time:
            return None
        elif self.begin_time == None:
            waiting_time = current_time - self.arrival_time
        else:
            # waiting_time = self.begin_time - self.arrival_time
            return 1

        required_time = self.total_size / global_max_bandwidth
        responce_ratio = round(required_time / required_time - 0.5) + 1
        return responce_ratio

    def get_cct(self):
        # print(self.status)

        # print([item.status for item in self.flows])
        # print(self.coflow_id)
        return self.complete_time - self.arrival_time

    def get_bottleneck(self):
        return max([item.size for item in self.flows])

    def get_min_flow_completion_time(self):
        min_time = sys.maxsize
        for flow in self.flows:
            if flow.status == 'r':
                time = flow.size / flow.allocated_bandwidth
                min_time = time if min_time > time else min_time
        return min_time

    def get_completion_status(self):
        stat = [item.status for item in self.flows]
        if 'r' not in stat and 'w' not in stat:
            return 'f'
        else:
            return 'r'

    def get_total_bandwidth(self):
        return sum(map(lambda x: x.allocated_bandwidth, self.flows))

    def estimate_time_to_heavy(self, heavy_size):
        if self.get_total_bandwidth() == 0 or self.fifo_is_heavy:
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
    parser.add_argument('--algorithm', required=True, choices=['wss', 'fifo','sebf','sebf_rr'])
    parser.add_argument('--bgflow', default=0.0, type=float)
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
    ccts = simulator.simulate(args.algorithm, sorted_coflows, background_flow=args.bgflow)
    # print(get_average(ccts), get_percentile(ccts))
    print("%f\t%f\t%f" % (args.bgflow, get_average(ccts), get_percentile(ccts)))

def all_exps():
    coflows = parse_file('./FB2010-1Hr-150-0.txt')
    sorted_coflows = sorted(coflows, key=lambda x: x.arrival_time)
    algorithms = ['wss', 'fifo', 'sebf', 'sebf_rr']
    # bgflows = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    bgflows = [0.2]
    for algorithm in algorithms:
        print(algorithm)
        for bgflow in bgflows:
            simulator = Simulator()
            ccts = simulator.simulate(algorithm, sorted_coflows.copy(), background_flow=bgflow)
            print(ccts)
            print(bgflows, get_average(ccts), get_percentile(ccts))

if __name__ == '__main__':
    main()
    # all_exps()
