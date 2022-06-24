from env_file import Crossroad
from TD3 import *

S_dim=150
A_dim=50
A_range=20
Prioritized=True

if __name__ == '__main__':
    if Prioritized:
        replay_buffer = Memory_prioritized(REPLAY_BUFFER_SIZE)
    else:
        replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    counter=0
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    brain=TD3(S_dim, A_dim, A_range, HIDDEN_DIM, replay_buffer, Prioritized, POLICY_TARGET_UPDATE_INTERVAL, Q_LR, POLICY_LR)
    T0 = time.time()
    env = Crossroad()

    if args.train:
        frame_idx = 0
        All_episode_reward = []
        env.initial()

        State=env.state_output().astype(np.float32)

        brain.policy_net([State])
        brain.target_policy_net([State])

        for epi in range(TRAIN_EPISODES):
            env.initial()
            State = env.state_output().astype(np.float32)
            episode_Reward = 0

            while True:
                if frame_idx > EXPLORE_STEPS:
                    Action=brain.policy_net.get_action(State, EXPLORE_NOISE_SCALE)
                else:
                    Action=brain.policy_net.sample_action()
                State_,Done,Reward=env.step(Action)
                State_=State_.astype(np.float32)
                Done = 1 if Done is True else 0
                if Prioritized:
                    replay_buffer.store(np.hstack((State,Action,Reward,State_,Done)))
                else:
                    replay_buffer.push(State,Action,Reward,State_,Done)
                State=State_
                episode_Reward += Reward
                frame_idx += 1
                if counter > BATCH_SIZE:


                    for i in range(UPDATE_ITR):
                        brain.update(BATCH_SIZE,EVAL_NOISE_SCALE,REWARD_SCALE)
                if Done==1:
                    break
                counter+=1
            if epi == 0:
                All_episode_reward.append(episode_Reward)
            else:
                All_episode_reward.append(All_episode_reward[-1]*0.9+episode_Reward)
            print(
                'Training  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    epi + 1, TRAIN_EPISODES, episode_Reward,
                    time.time() - T0
                )
            )
        brain.save()
        plt.plot(All_episode_reward)
        if not os.path.exists('image_new'):
            os.makedirs('image_new')
        plt.savefig(os.path.join('image_new', '_'.join([ALG_NAME, 'Cross_road'])))

    if args.test:
        brain.load()
        env.initial()

        State = env.state_output().astype(np.float32)
        brain.policy_net([State])
        for epi in range(TEST_EPISODES):
            env.initial()
            State = env.state_output().astype(np.float32)
            episode_Reward = 0

            while True:
                Action=brain.policy_net.get_action(State, EXPLORE_NOISE_SCALE,greedy=True)
                State, Done, Reward = env.step(Action)
                State=State.astype(np.float32)
                episode_Reward+=Reward

                if Done:
                    break
            print(
                'Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    epi + 1, TEST_EPISODES, episode_Reward,
                    time.time() - T0
                )
            )







