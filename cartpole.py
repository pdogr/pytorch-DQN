from pkgs import *
from agent import DQNAgent, DDQNAgent
from model import CartPoleDuelDNN
from env import make_env
from logger import Logger

params = {
    'replay_size': 10000,
    'lr': 0.0007,
    'gamma': 0.99,
    'batch_size': 32,
    'network_update_freq': 100,
    'initial_eps': 1,
    'min_eps': 0.01,
    'final_explore_frame': 2000,
    'no_ops': 30,
    'avg_history_length': 4,
    'action_repeat': 4,
    'dims': (1, 84, 84),
    'replay_start': 1000,
    'max_epsisode_len': 2000
}


if __name__ == '__main__':
    env_name = 'CartPole-v0'

    log_dir = 'log/'
    save_dir = 'model/'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    env = gym.make(env_name)

    optimizer_params = {
        'lr': params['lr']
    }
    device = T.device('cuda' if T.cuda.is_available() else 'cpu')

    model = CartPoleDuelDNN
    agent = DQNAgent(initial_eps=params['initial_eps'], min_eps=params['min_eps'],
                     final_explore_frame=params['final_explore_frame'],
                     gamma=params['gamma'], C=params['network_update_freq'],
                     batch_size=params['batch_size'],
                     actions=env.action_space.n,
                     replay_size=params['replay_size'], replay_start=params['replay_start'],
                     model=model, model_params={'input_dims': env.observation_space.shape,
                                                'action_dim': env.action_space.n},
                     optimizer_params=optimizer_params,
                     device=device)

    session_name = '_'.join(
        [env_name, model.__name__, agent.__class__.__name__])

    logger = Logger(session_name, log_dir)
    logger.info('session_name: {}'.format(session_name))
    logger.info('model: {}'.format(model))
    logger.info('agent: {}'.format(agent))

    def add_prefix(x): return save_dir + \
        session_name + '_'+x+'.pkl'

    start = 0
    max_steps = 15000
    n_epsidoes = 500
    best_score = -np.inf
    save_every = 20
    n_steps = 0

    scores, episodes = [], []
    losses = []
    for i in range(n_epsidoes):
        if n_steps >= max_steps:
            break
        done = False
        s = env.reset()
        episode_score = 0
        episode_losses = []
        while not done:
            a = agent.action(s)
            s_, r, done, _ = env.step(a)
            agent.save(s, a, r, s_, done)
            loss = agent.step()
            losses.append(loss)
            episode_score += r
            episode_losses.append(loss)
            s = s_
            n_steps += 1

        scores.append(episode_score)
        episodes.append(i+1)
        avg_score = np.mean(scores[-10:])
        if avg_score >= best_score:
            logger.info('Saving best model...')
            agent.save_model(add_prefix('best'))
            best_score = avg_score
        if i % save_every == 0:
            agent.save_model(add_prefix(str(n_steps)))

        info_dict = {
            'episode': i,
            'steps': n_steps,
            'episode_score': episode_score,
            'mean_episode_loss': np.mean(episode_losses),
            'epsilon': agent.epsilon,
        }
        msg = session_name+'\t'
        logger.set_step(n_steps)
        for k, v in info_dict.items():
            msg = msg + '{}: {}\t'.format(k, v)
            if v is not None:
                logger.add_scalar(k, v)
        logger.info(msg)
