from semantirl.data import load_turk_train

def main():
    for traj in load_turk_train():
        traj.pretty_print()
        break

if __name__ == "__main__":
    main()
