import os
from collections import namedtuple
import numpy as np

from render import Canvas

#COLOR = ['blue', 'green', 'magenta', 'red', 'yellow']
COLOR = ["black", "blue", "cyan", "darkGray", "gray", "green", "lightGray", "magenta", "orange", "pink", "red", "white", "yellow"]
ACTIONS = ['north', 'south', 'east', 'west', 'pull']
#SHAPES = ['chair', 'bag', 'backpack', 'basket']
SHAPES = ["star", "moon", "circle", "smiley", "square"]
ENV_DIMENSIONS = [11,11] #[24, 24]
Box = namedtuple('Box', ['left', 'top', 'right', 'bottom'])
Position = namedtuple('Pos', ['x', 'y'])

# Use a restricted color set
COLOR_TO_CHANNEL = {col:channel for (channel, col) in enumerate(['red', 'green', 'blue'])}


class Trajectory(object):
    """ Contains a list of states and actions """
    def __init__(self, states, actions, sentence=None):
        self.states = states
        self.actions = actions
        self.sentence = sentence

    def pretty_print(self):
        if self.sentence:
            print 'Command:', ' '.join(self.sentence)
        for (s, a) in zip(self.states, self.actions):
            print s
            print a


class OOState(object):
    def __init__(self, obj_list):
        self.obj_list = obj_list
        self.env_dim = ENV_DIMENSIONS

    def display(self):
        """ Print state to terminal """
        canvas = Canvas(self.env_dim)
        for obj in self.obj_list:
            obj.draw(canvas)
        canvas.display()

    def to_dense(self):
        """Return a dense matrix/vector-based representation of this state"""

        def create_channels(n=1):
            ch = np.array(np.zeros(self.env_dim), dtype=np.float)
            ch = np.expand_dims(ch, axis=0)
            return np.tile(ch, [n, 1, 1])

        # Create channels for rooms, agent, doors, and blocks
        room_channel = create_channels(n=len(COLOR_TO_CHANNEL))
        agent_channel = create_channels()
        block_channel = create_channels()
        door_channel = create_channels()

        # Populate channels
        for obj in self.obj_list:
            if obj.type == MDPObj.AGENT:
                agent_channel[0, obj.pos.x, obj.pos.y] = 1.0
            elif obj.type == MDPObj.BLOCK:
                block_channel[0, obj.pos.x, obj.pos.y] = 1.0
            elif obj.type == MDPObj.ROOM:
                box = obj.box
                channel = COLOR_TO_CHANNEL[obj.color]
                for i in range(box.left, box.right+1):
                    for j in range(box.bottom, box.top+1):
                        room_channel[channel, i, j] = 1.0
            elif obj.type == MDPObj.DOOR:
                box = obj.box
                for i in range(box.left, box.right+1):
                    for j in range(box.bottom, box.top+1):
                        door_channel[0, i, j] = 1.0
        # Concatenate all channels together
        all_channels = np.r_[room_channel, door_channel, block_channel, agent_channel]
        return all_channels

    def __str__(self):
        return 'State(%s)' % str([obj for obj in self.obj_list])


class MDPObj(object):
    ROOM = 'room'
    DOOR = 'door'
    AGENT = 'agent'
    BLOCK = 'block'
    def __init__(self, _type, attrs):
        self._type = _type
        self.attrs = attrs

    @property
    def type(self):
        return self._type

    @property
    def color(self):
        return self.attrs['color']

    @property
    def box(self):
        return self.attrs['box']

    @property
    def pos(self):
        return self.attrs['pos']

    @property
    def shape(self):
        return self.attrs['shape']

    def draw(self, canvas):
        if self.type == MDPObj.ROOM:
            box = self.box
            for i in range(box.left, box.right+1):
                for j in range(box.bottom, box.top+1):
                    canvas.set(i,j,' ',self.color)
        elif self.type == MDPObj.DOOR:
            box = self.box
            for i in range(box.left, box.right+1):
                for j in range(box.bottom, box.top+1):
                    canvas.set(i,j,'D')
        elif self.type == MDPObj.BLOCK:
            pos = self.pos
            canvas.set(pos.x, pos.y, 'B')
        elif self.type == MDPObj.AGENT:
            pos = self.pos
            canvas.set(pos.x, pos.y, 'A')

    def __repr__(self):
        return '%s' % self.type


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

#============
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
HOME = os.environ['HOME']
DATA_DIR = os.path.join(HOME, 'code/semantirl/data/allTurkTrain')

def main():
    for i, fname in enumerate(os.listdir(DATA_DIR)):
        traj = parse_file(os.path.join(DATA_DIR, fname))
        print traj.sentence
        for state, act in zip(traj.states, traj.actions):
            canvas = state.display()
            print state.to_dense()
            print act
            break
        if i==0:
            break

if __name__ == "__main__":
    main()

