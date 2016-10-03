import os
import numpy as np

from semantirl.environment import *
from semantirl.utils.file_utils import DATA_DIR


def parse_file(fname):
    with open(fname, 'r') as f:
        sentence = None
        actions = []
        states = []
        for i, line in enumerate(f):
            if i == 0:
                sentence = line.strip().split(' ')
            elif i == 1:
                actions = line.strip().split(',')
            else:
                states.append(parse_state(line.strip()))
        #actions = [ACTION_TO_INDEX[a] for a in actions]
    return Trajectory(states, actions, sentence=sentence)

def parse_state(line):
    """
    >>> parse_state('room,5,0,10,6,4 agent,8,6')
    """
    objs = line.split(' ')
    obj_list = []
    for obj in objs:
        params = obj.split(',')
        obj_type = params[0]
        obj_params = [int(i) for i in params[1:]]
        mdp_obj = OBJ_PARSERS[obj_type](obj_params)
        obj_list.append(mdp_obj)
    return OOState(obj_list)

def parse_room(params):
    color_idx, left, top, right, bottom = params
    color = COLOR[color_idx]
    box = Box(left, top, right, bottom)
    return MDPObj('room', {'color':color, 'box':box})

def parse_door(params):
    left, top, right, bottom = params
    box = Box(left, top, right, bottom)
    return MDPObj('door', {'box':box})

def parse_block(params):
    color_idx, shape_idx, x, y = params
    color = COLOR[color_idx]
    shape = SHAPES[shape_idx]
    pos = Position(x, y)
    return MDPObj('block', {'color':color, 'shape':shape, 'pos': pos})

def parse_agent(params):
    x, y = params
    pos = Position(x, y)
    return MDPObj('agent', {'pos': pos})


OBJ_PARSERS = {'room': parse_room, 'door': parse_door,
               'block': parse_block, 'agent': parse_agent}


def load_dataset(name):
    """ Return an iterator through Trajectory objects """
    dirname = os.path.join(DATA_DIR, name)
    for i, fname in enumerate(os.listdir(dirname)):
        traj = parse_file(os.path.join(dirname, fname))
        yield traj


def load_turk_train():
    for traj in load_dataset('allTurkTrain'):
        yield traj


def load_turk_train_limited():
    for traj in load_dataset('allTurkTrainLimitedCommand'):
        yield traj


def main():
    for traj in load_turk_train():
        traj.pretty_print()
        break

if __name__ == "__main__":
    main()

