# OpenGym CartPole-v0 with A3C on GPU
# -----------------------------------
#
# A3C implementation with GPU optimizer threads.
# 
# Made as part of blog series Let's make an A3C, available at
# https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/
#
# author: Jaromir Janisch, 2017

import numpy as np
import tensorflow as tf

import time, random, threading

from keras.models import *
from keras.layers import *
from keras import backend as K

from PixelDrawEnv import PixelDrawEnv

RUN_TIME = 1000
THREADS = 10
OPTIMIZERS = 3
THREAD_DELAY = 0.001

GAMMA = 0.99

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.8
EPS_STOP = 0.2
EPS_STEPS = 75000

MIN_BATCH = 128
LEARNING_RATE = 5e-3
LOSS_V = .5  # v loss coefficient
LOSS_ENTROPY = .03  # entropy coefficient


# ---------
class Brain:
    train_queue = [[], [], [], [], []]  # s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()

    def __init__(self):
        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.model = self._build_model()
        self.model.summary()
        self.graph = self._build_graph(self.model)

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()

        # self.default_graph.finalize()  # avoid modifications

    def _build_model(self):

        l_image_input = Input(batch_shape=(None, 28, 28, 1))
        l_dense = Conv2D(32, padding="same", kernel_size=(3, 3), activation='relu')(l_image_input)
        # l_dense = Conv2D(32, padding="same", kernel_size=(3, 3), activation='relu')(l_dense)
        l_image_flatten = Flatten()(l_dense)
        l_dense = BatchNormalization()(l_image_flatten)
        model_image = Model(inputs=[l_image_input], outputs=[l_dense])

        l_image_input = Input(batch_shape=(None, 28, 28, 1))
        l_dense = Conv2D(32, padding="same", kernel_size=(3, 3), activation='relu')(l_image_input)
        # l_dense = Conv2D(32, padding="same", kernel_size=(3, 3), activation='relu')(l_dense)
        l_image_flatten = Flatten()(l_dense)
        l_dense = BatchNormalization()(l_image_flatten)
        # l_dense = Dropout(0.5)(l_dense)

        model_image_target = Model(inputs=[l_image_input], outputs=[l_dense])

        l_position_input = Input(batch_shape=(None, 1, 2))
        l_dense = Flatten()(l_position_input)
        l_dense = Dense(32, activation='relu')(l_dense)
        l_dense = BatchNormalization()(l_dense)
        position_model = Model(inputs=[l_position_input], outputs=[l_dense])

        combined = concatenate([model_image.output, model_image_target.output, position_model.output])

        l_dense = Dense(32, activation='relu')(combined)
        # l_dense = Dense(32, activation='relu')(l_dense)
        l_dense = Dropout(0.5)(l_dense)
        out_actions = Dense(NUM_ACTIONS, activation='softmax')(l_dense)
        out_value = Dense(1, activation='linear')(l_dense)

        model = Model(inputs=[model_image.input, model_image_target.input, position_model.input],
                      outputs=[out_actions, out_value])
        model._make_predict_function()  # have to initialize before threading

        return model

    def _build_graph(self, model):
        s_t_i = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
        s_t_t = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
        s_t_p = tf.placeholder(tf.float32, shape=(None, 1, 2))
        a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        r_t = tf.placeholder(tf.float32, shape=(None, 1))  # not immediate, but discounted n step reward

        p, v = model([s_t_i, s_t_t, s_t_p])

        log_prob = tf.log(tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
        advantage = r_t - v

        loss_policy = - log_prob * tf.stop_gradient(advantage)  # maximize policy
        loss_value = LOSS_V * tf.square(advantage)  # minimize value error
        entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1,
                                               keep_dims=True)  # maximize entropy (regularization)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
        minimize = optimizer.minimize(loss_total)

        return s_t_i, s_t_t, s_t_p, a_t, r_t, minimize

    def optimize(self):
        if len(self.train_queue[0]) < MIN_BATCH:
            time.sleep(0)  # yield
            return

        with self.lock_queue:
            if len(self.train_queue[0]) < MIN_BATCH:  # more thread could have passed without lock
                return  # we can't yield inside lock

            s, a, r, s_, s_mask = self.train_queue
            self.train_queue = [[], [], [], [], []]

        s = np.array(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.array(s_)
        s_mask = np.vstack(s_mask)
        if len(s) > 5 * MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))
        s_batch, p_batch, t_batch = [], [], []
        for i in s_:
            s_batch.append(i[0])
            t_batch.append(i[1])
            p_batch.append(i[2])
        s_batch = np.array(s_batch)
        p_batch = np.array(p_batch)
        t_batch = np.array(t_batch)
        v = self.predict_v([s_batch, t_batch, p_batch])
        r = r + GAMMA_N * v * s_mask  # set v to 0 where s_ is terminal state

        s_t_i, s_p_t, s_t_p, a_t, r_t, minimize = self.graph
        self.session.run(minimize, feed_dict={s_t_i: s_batch, s_p_t: t_batch, s_t_p: p_batch, a_t: a, r_t: r})

    def train_push(self, s, a, r, s_):
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if s_ is None:
                self.train_queue[3].append(NONE_STATE)
                self.train_queue[4].append(0.)
            else:
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)

    def predict(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p, v

    def predict_p(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)  # [s,np.array([[1,2]])]
            return p

    def predict_v(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return v


# ---------
frames = 0


class Agent:
    def __init__(self, eps_start, eps_end, eps_steps):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps

        self.memory = []  # used for n_step return
        self.R = 0.

    def getEpsilon(self):
        if (frames >= self.eps_steps):
            return self.eps_end
        else:
            return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps  # linearly interpolate

    def act(self, s):
        eps = self.getEpsilon()
        global frames;
        frames = frames + 1

        if random.random() < eps:
            return random.randint(0, NUM_ACTIONS - 1)

        else:
            # s = np.array([s])
            ss = [np.expand_dims(s[0], axis=0), np.expand_dims(s[1], axis=0), np.expand_dims(s[2], axis=0)]
            p = brain.predict_p(ss)[0]

            # a = np.argmax(p)
            a = np.random.choice(NUM_ACTIONS, p=p)

            return a

    def train(self, s, a, r, s_):
        def get_sample(memory, n):
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n - 1]

            return s, a, self.R, s_

        a_cats = np.zeros(NUM_ACTIONS)  # turn action into one-hot representation
        a_cats[a] = 1

        self.memory.append((s, a_cats, r, s_))

        self.R = (self.R + r * GAMMA_N) / GAMMA

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                brain.train_push(s, a, r, s_)

                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)

            self.R = 0

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            brain.train_push(s, a, r, s_)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)


# possible edge case - if an episode ends in <N steps, the computation is incorrect


# ---------
class Environment(threading.Thread):
    stop_signal = False

    def __init__(self, render=False, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS):
        threading.Thread.__init__(self)
        self.is_test = False
        self.count = 5
        self.render = render
        self.env = PixelDrawEnv()
        self.agent = Agent(eps_start, eps_end, eps_steps)

    def runEpisode(self):
        s = self.env.reset()

        R = 0
        while True:
            time.sleep(THREAD_DELAY)  # yield

            if self.render: self.env.render()

            a = self.agent.act(s)
            s_, r, done, info = self.env.step(a)

            if done:  # terminal state
                s_ = None

            self.agent.train(s, a, r, s_)

            s = s_
            R += r

            if done or self.stop_signal:
                with open('rewards.txt', 'a') as a_writer:
                    a_writer.write(str(R) + '\n')
                if R > 100:
                    self.env.render()
                break

        print("Total R:", R)

    def run(self):
        while not self.stop_signal:
            self.runEpisode()
            if self.is_test:
                self.env.new_episode()
                self.count -= 1
                if self.count == 0:
                    break

    def stop(self):
        self.stop_signal = True


# ---------
class Optimizer(threading.Thread):
    stop_signal = False

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while not self.stop_signal:
            brain.optimize()

    def stop(self):
        self.stop_signal = True


env_test = Environment(render=True)
NUM_STATE = env_test.env.num_state
NUM_ACTIONS = env_test.env.num_action
NONE_STATE = np.zeros(NUM_STATE)

brain = Brain()  # brain is global in A3C

envs = [Environment() for i in range(THREADS)]
opts = [Optimizer() for i in range(OPTIMIZERS)]

for o in opts:
    o.start()

for e in envs:
    e.start()

time.sleep(RUN_TIME)

for e in envs:
    e.stop()
for e in envs:
    e.join()

for o in opts:
    o.stop()
for o in opts:
    o.join()

print("Training finished --------------------------------------------------------------")
env_test.is_test = True
env_test.run()
