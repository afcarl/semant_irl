"""
Grammar Tree


--- Movement Primitives ---
Go to X, Y




"""
import numpy as np

from semantirl.utils.colors import ColorLogger

LOGGER = ColorLogger(__name__, 'blue')


def angle_dist(a1, a2):
    """
    >>> angle_dist(0,20)
    20
    >>> angle_dist(350,20), angle_dist(20,350)
    (30, 30)
    >>> angle_dist(190,150), angle_dist(150,190)
    (40, 40)
    """
    if a1<a2:
        return angle_dist(a2, a1)
    return min(abs(a1-a2), abs(360-a1+a2))

def angle_dir(ref, tgt):
    """
    >>> angle_dir(0, 10), angle_dir(10,0)
    (1, 0)
    >>> angle_dir(350, 10), angle_dir(10,350)
    (1, 0)
    >>> angle_dir(190, 150), angle_dir(150, 190)
    (0, 1)
    >>> angle_dir(50, 350)
    0
    >>> angle_dir(160,330)
    1
    """
    if ref<tgt:
        return 1-angle_dir(tgt,ref)
    a1, a2 = ref, tgt
    d1 = abs(a1-a2)
    d2 = abs(360-a1 + a2)
    if d1 <= d2:
        return 0
    return 1


class obswrapper(object):
    def __init__(self, obs_vec):
        self.data = obs_vec

    @property
    def XPos(self):
        return self.data[0]

    @property
    def YPos(self):
        return self.data[1]

    @property
    def ZPos(self):
        return self.data[2]

    @property
    def yaw(self):
        return self.data[3]

    @property
    def pitch(self):
        return self.data[4]


class ActionInterface():
    @property
    def nop(self):
        return np.array([0,0,0,0,0])

    def rotate_left(self, amt=1.):
        amt = np.clip(amt, 0, 1)
        return np.array([0, amt,0,0,0])

    def rotate_right(self, amt=1.):
        amt = np.clip(amt, 0, 1)
        return np.array([0, -amt,0,0,0])

    def move_forward(self, amt=1.):
        amt = np.clip(amt, 0, 1)
        return np.array([amt, 0,0,0,0])

    def look_down(self, amt=1.):
        return np.array([0, 0., amt,0,0,0])

    def look_up(self, amt=1.):
        return self.look_down(-amt)

    def jump(self):
        return np.array([0, 0., 0, 0, 1])

    def use(self):
        return np.array([0, 0., 0, 1, 0])


class Subpolicy(object):
    def __init__(self):
        pass

    def act(self, obs, memory=None):
        """ returns action, done """
        pass

    def run(self, s0, env):
        LOGGER.info("Executing primitive %s", type(self).__name__)
        done = False
        memory = None
        obs = s0
        while not done:
            act, done, memory = self.act(obs, memory=memory)
            obs, rew, env_done, _ = env.step(act)

            if done or env_done:
                break
        return obs


class MoveXZ(Subpolicy):
    """ Move to a relative target"""
    def __init__(self, target, max_t=100):
        super(MoveXZ, self).__init__()
        self.action_interface = ActionInterface()
        self.dist_thresh = 0.05
        self.target = target
        self.max_t = max_t


    def act(self, obs, memory=None):
        obs = obswrapper(obs)
        x, z, yaw = obs.XPos, obs.ZPos, obs.yaw

        if memory is None:
            x, z = obs.XPos, obs.ZPos
            tx, tz = x+self.target[0], z+self.target[1]
            memory = {'tgt': np.array([tx, tz]), 't':0}
        memory['t'] += 1
        tgt = memory['tgt']

        diff_un = np.array([tgt[0]-obs.XPos, tgt[1]-obs.ZPos])
        distance = np.linalg.norm(diff_un)
        diff = diff_un/distance
        fwd_dir = rotate(np.array([0,1]), yaw)
        right_dir = rotate(np.array([1,0]), yaw)
        #print yaw, diff, fwd_dir, right_dir
        if memory['t'] % 5 == 0:
            LOGGER.debug('Dist: %f', distance)
        print 'dist:', distance
        if np.linalg.norm(diff_un) < self.dist_thresh:
            return self.action_interface.nop, True, None

        cos = np.dot(diff, fwd_dir)
        #angle = np.arccos(cos)*180/np.pi
        #print '\t\t',cos
        
        if cos > 0.95:
            #print 'F'
            amt = min(1., distance/2)
            return self.action_interface.move_forward(amt=amt), False, memory

        if np.dot(diff, right_dir) > 0:
            #print 'R'
            return self.action_interface.rotate_right(), False, memory
        else:
            #print 'L'
            return self.action_interface.rotate_left(), False, memory


class PlaceBlockJump(Subpolicy):
    """ Move to a relative target"""
    def __init__(self):
        super(PlaceBlockJump, self).__init__()
        self.action_interface = ActionInterface()

    def act(self, obs, memory=None):
        obs = obswrapper(obs)
        pitch = obs.pitch
        if memory is None:
            memory = {'t':-1, 'state': 0, 'y': obs.YPos}
        memory['t'] += 1
        t = memory['t']

        #print 'pitch:', pitch
        # Look down
        if memory['state'] == 0:
            if pitch > 0:
                return self.action_interface.look_down(), False, memory
            else:
                memory['state'] += 1

        if memory['state'] == 1:
            elevation = obs.YPos - memory['y']
            if elevation < 1:
                return self.action_interface.jump(), False, memory
            else:
                memory['state'] += 1
                return self.action_interface.use(), False, memory

        # Look back up
        if memory['state'] == 2:
            diff = pitch-90
            if np.abs(diff) > 1:
                return self.action_interface.look_down(np.tanh(0.05*diff)), False, memory
            else:
                memory['state'] += 1

        return self.action_interface.nop, True, None


def rotate(v, th):
    th = th/180 * np.pi
    mat = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    return mat.dot(v)
