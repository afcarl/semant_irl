import pickle

from semantirl.data import load_turk_train
from semantirl.environment import SokobanEnviron
from semantirl.models.cmd2act import Cmd2Act
from semantirl.utils.vocab import PAD

def rollout(model, init_state, sentence):
    env = SokobanEnviron(init_state)

    act_seq = model.compute_action_sequence(init_state, sentence)
    for act in act_seq:
        print 'act:', act
        if act == PAD:
            continue
        env.step(act)
        env.render()

def main():
    with open('cmd2act.model', 'r') as f:
        cmd2act = pickle.load(f)


    for traj in load_turk_train():
        print ' '.join(traj.sentence)
        init_state = traj.states[0]
        rollout(cmd2act, init_state, traj.sentence)
        break


if __name__ == "__main__":
    main()
