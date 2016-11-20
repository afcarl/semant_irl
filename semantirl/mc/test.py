from semantirl.mc.mc_env import WorldDef
from semantirl.mc.mc_env import Minecraft
from semantirl.mc.primitives import MoveXZ, PlaceBlockJump

if __name__ == '__main__':
    #mission = MazeDef('TMaze').generate_mission()
    #agent = MalmoPython.AgentHost()
    #mission_record_spec = MalmoPython.MissionRecordSpec()
    #agent.startMission(mission, mission_record_spec)
    mc = Minecraft(WorldDef(), video_dim=(1024,1024), discrete_actions=False, reset=False, time_limit=100)
    s0 = mc.reset()

    for _ in range(10):
        s0 = PlaceBlockJump().run(s0, mc)
        s0 = MoveXZ([0,-1]).run(s0, mc)


    """
    for t in range(100):
        act = mc.action_space.sample()
        print act
        obs, rew, done, _ =  mc.step(act)
        if done:
            print 'Mission done!'
            break
        print t, obs, rew
    """
