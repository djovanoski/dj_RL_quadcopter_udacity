import tensorflow as tf
import numpy as np
import os
import pandas as pd
import time


from agents.actor_model import ActorNetwork
from agents.critic_model import CriticNetwork
from agents.ornstein_uhlenbeck_action_noise import OrnsteinUhlenbeckActionNoise
from agents.replay_buffer import ReplayBuffer

class Agent:
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task, sess,  agent_params=None, training_params=None, dir_params=None, is_opengym=True):
        self.task = task
        self.sess = sess
        self.is_opengym = is_opengym
        if agent_params:
            self.agent_params = agent_params
        else:
            self.agent_params = {'actor_lr':0.0001,'critic_lr': 0.001, 'gamma': 0.99, 'tau':0.001, 'buffer_size':1000000,'minibatch_size':64}
        if training_params:
            self.training_params = training_params
        else:
            self.training_params = {'random_seed': 1234, 'max_episodes': 30, 'max_episodes_len': 2000}
        if dir_params:
            self.dir_params = dir_params
        else:
            self.dir_params = {'monitor_dir': '/notebooks/graphs/monitor', 'summary_dir': '/notebooks/graphs/ddqn' }
        if self.is_opengym:
            self.state_size = task.observation_space.shape[0]
            self.action_size = task.action_space.shape[0]
            self.action_low = task.action_space.low
            self.action_bound = task.action_space.high
            self.action_range = self.action_bound - self.action_low
            #assert (self.action_high == -self.action_low)
        else:
            self.state_size = task.state_size
            self.action_size = task.action_size
            self.action_low = task.action_low
            self.action_bound = task.action_high
            #self.action_range = self.action_high - self.action_low

        #Define Actor Network
        self.actor = ActorNetwork(self.sess,
                                  self.state_size,
                                  self.action_size,
                                  self.action_bound,
                                  self.agent_params['actor_lr'],
                                  self.agent_params['tau'],
                                  self.agent_params['minibatch_size'])
        #Define Critic Network
        self.critic = CriticNetwork(self.sess,
                                    self.state_size,
                                    self.action_size,
                                    self.agent_params['critic_lr'],
                                    self.agent_params['tau'],
                                    self.agent_params['gamma'],
                                    self.actor.get_num_trainable_vars())
        self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_size))

        self.summary_ops, self.summary_vars = self.build_summaries()
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(self.dir_params['summary_dir'], self.sess.graph)


         # Initialize target network weights
        self.actor.update_target_network()
        self.critic.update_target_network()
        self.replay_buffer = ReplayBuffer(self.agent_params['buffer_size'], self.training_params['random_seed'])
        self.ep_ave_max_q, self.ep_reward  = 0,0
        self.best_q_reward,self.best_ep_reward, self.final_q_reward = 0,0,0
        self.num_step, self.num_episode, self.whichepisode_reward, self.whichepisode_q = 0,0,0,0
        self.stats_columns = ['episode', 'total_reward', 'best_q_reward']
        self.stats_filename = os.path.join(self.dir_params['monitor_dir'],"stats_{}.csv".format(time.strftime("%Y%m%d")))


    def act(self, state):
        return self.actor.predict(np.reshape(state, (1, self.actor.s_dim))) + self.actor_noise()

    def buffer_me(self, state, action, reward, terminal_state, next_state):
        s_ = np.reshape(state, (self.actor.s_dim,))
        a_ = np.reshape(action, (self.actor.a_dim,))
        next_s = np.reshape(next_state, (self.actor.s_dim,))
        return self.replay_buffer.add(s_, a_, reward, terminal_state, next_s)

    def step(self, action, state):
        self.num_step += 1
        if self.is_opengym:
            next_state, reward, terminal_state, info = self.task.step(action[0])
        else:
            next_state, reward, terminal_state = self.task.step(action[0])
        self.buffer_me(state, action, reward, terminal_state, next_state)
        self.get_episode_rewards(reward)
        self.experience_learn()
        if terminal_state:
            self.get_best_score()
            self.get_final_q_rewards()
            self.write_stats([self.num_episode, self.ep_reward, self.best_q_reward])
            self.writer.add_summary(self.write_summaries(), self.num_episode)
        return next_state, reward, terminal_state
    def get_best_score(self):
        if self.ep_reward > self.best_ep_reward:
            self.best_q_reward = self.ep_reward
            self.whichepisode_reward = self.num_episode
        if self.ep_ave_max_q > self.best_q_reward:
            self.whichepisode_q = self.num_episode
            self.best_q_reward = self.ep_ave_max_q

    def get_q_rewards(self,predicted_q_value):
        self.ep_ave_max_q += np.amax(predicted_q_value)

    def get_final_q_rewards(self):
        self.final_q_reward = self.ep_ave_max_q/self.num_step

    def get_episode_rewards(self, reward):
        self.ep_reward += reward

    def reset_episode(self):
        state = self.task.reset()
        self.num_episode += 1
        self.num_step,self.ep_reward, self.ep_ave_max_q = 1, 0, 0
        return state

    def experience_learn(self):
        if self.replay_buffer.size() > self.agent_params['minibatch_size']:
            s_batch, a_batch, r_batch, t_batch, s2_batch = self.replay_buffer.sample_batch(self.agent_params['minibatch_size'])
            target_q = self.target_q(s2_batch)
            update_q = self.update_targets(r_batch, t_batch,target_q)
            predict_q = self.predict_q(s_batch, a_batch, update_q)
            self.get_q_rewards(predict_q)
            self.actor_train(s_batch)
            self.actor.update_target_network()
            self.critic.update_target_network()

    def target_q(self, next_state):
        actor_predict_target = self.actor.predict_target(next_state)
        critic_predict_target = self.critic.predict_target(next_state,actor_predict_target)
        return critic_predict_target

    def update_targets(self,reward,terminal_state, critic_predict_target):
        y_i = []
        for k in range(self.agent_params['minibatch_size']):
            if terminal_state[k]:
                y_i.append(reward[k])
            else:
                y_i.append(reward[k] + self.critic.gamma * critic_predict_target[k])
        return y_i

    def predict_q(self, state, action, y_i):
        predicted_q_value, _ = self.critic.train(state, action, np.reshape(y_i, (self.agent_params['minibatch_size'], 1)))
        return predicted_q_value

    def update_actor_policy(self,state):
        return self.actor.predict(state)

    def update_gradients(self, state):
        return self.critic.action_gradients(state, self.update_actor_policy(state))

    def actor_train(self,state):
        return self.actor.train(state, self.update_gradients(state)[0])

    def build_summaries(self):
        episode_reward = tf.Variable(0.)
        tf.summary.scalar("Reward", episode_reward)
        episode_ave_max_q = tf.Variable(0.)
        tf.summary.scalar("Qmax_Value", episode_ave_max_q)
        summary_vars = [episode_reward, episode_ave_max_q]
        summary_ops = tf.summary.merge_all()
        return summary_ops, summary_vars

    def write_summaries(self):
        return self.sess.run(self.summary_ops, feed_dict={self.summary_vars[0]: self.ep_reward,self.summary_vars[1]: self.final_q_reward})
    #Todo add summaries
    def write_stats(self, stats):
        """Write single episode stats to CSV file."""
        df_stats = pd.DataFrame([stats], columns=self.stats_columns)  # single-row dataframe
        df_stats.to_csv(self.stats_filename, mode='a', index=False, header=not os.path.isfile(self.stats_filename))  # write header first time only
