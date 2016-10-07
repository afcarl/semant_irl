import numpy as np
from collections import namedtuple

from semantirl.render import Canvas

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

# Map actions to a discrete set of numbers
ACTION_TO_INDEX = {act:idx for (idx, act) in enumerate(ACTIONS)}


class SokobanEnviron(object):
    def __init__(self, init_state):
        self.state = init_state

    def step(self, action):
        agent = self.state.agent
        pos = [agent.pos.x, agent.pos.y]
        directions = {
            'north': [0, 1],
            'south': [0, -1],
            'east': [1, 0],
            'west': [-1, 0],
        }
        if action in directions:
            offset = np.array(directions[action])
            pos = offset+pos
            new_agent = MDPObj(MDPObj.AGENT, {'pos': Position(pos[0], pos[1])})
            self.state.set_agent(new_agent)

    def render(self):
        self.state.display()


class Trajectory(object):
    """ Contains a list of states and actions """
    def __init__(self, states, actions, sentence=None):
        self.states = states
        self.actions = actions
        self.sentence = sentence

    def pretty_print(self):
        if self.sentence:
            print 'Command:', ' '.join(self.sentence)
        for i, (s, a) in enumerate(zip(self.states, self.actions)):
            print '=='*10
            print 'T=',i
            s.display()
            print 'Action:', a

    def __str__(self):
        return 'Trajectory(l=%d)' % len(self.actions)


class OOState(object):
    def __init__(self, obj_list):
        self.obj_list = obj_list
        self.env_dim = ENV_DIMENSIONS

    @property
    def agent(self):
        for obj in self.obj_list:
            if obj.type == MDPObj.AGENT:
                return obj
        return None

    def set_agent(self, new_agent):
        agent_idx = None
        for i, obj in enumerate(self.obj_list):
            if obj.type == MDPObj.AGENT:
                agent_idx = i
        self.obj_list[agent_idx] = new_agent


    def display(self):
        """ Print state to terminal """
        canvas = Canvas(self.env_dim)
        for obj in self.obj_list:
            obj.draw(canvas)
        canvas.display()

    def to_dense(self, include_room=False):
        """
        Return a dense matrix representation of this state

        Args:
            include_room (bool): If true, includes the room & door channels
    
        Returns:
            An (C, N, N) numpy array, where C = num channels, N=size of environment
            The channels are:
                Room: 1 channel for each room color.
                      1's indicating the area of the room, 0 elsewhere.
                Door: 1 where there are doors, 0 elsewhere
                Block: A 1 where the block is, 0 elsewhere
                Agent: A 1 on the tile of the agent, 0 elsewhere
        """
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
        if include_room:
            all_channels = np.r_[room_channel, door_channel, block_channel, agent_channel]
        else:
            all_channels = np.r_[block_channel, agent_channel]
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
                    canvas.set_back(i,j,self.color)
        elif self.type == MDPObj.DOOR:
            box = self.box
            for i in range(box.left, box.right+1):
                for j in range(box.bottom, box.top+1):
                    canvas.set_fore(i,j,'D')
        elif self.type == MDPObj.BLOCK:
            pos = self.pos
            canvas.set_fore(pos.x, pos.y, 'B')
        elif self.type == MDPObj.AGENT:
            pos = self.pos
            canvas.set_fore(pos.x, pos.y, 'A')

    def __repr__(self):
        return '%s' % self.type

