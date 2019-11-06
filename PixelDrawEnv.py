import numpy as np
from PIL import Image
from keras.datasets import mnist
from scipy.spatial.distance import cdist


class PixelDrawEnv(object):
    i = 1
    prefix = 1

    def __init__(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.target_original = x_train[0]
        self.target = np.array(list(map(lambda x: 255 if x > 0 else 0, x_train[0].flatten())))
        self.test = [[i, j] for i in range(28) for j in range(28) if self.target.reshape((28, 28))[i][j] > 0]
        self.canvas_shape = 28
        # self.num_state = self.canvas_shape **2
        self.num_state = (28, 28, 1)
        self.num_action = 16
        self.color = 255
        self.canvas = np.zeros((self.canvas_shape, self.canvas_shape), dtype=np.uint8)
        self.pen_position = (7, 7)
        self.no_drawn_pixels = 0
        self.max_drawn_pixels_per_episode = (30 / 100) * self.num_state[0] ** 2
        self.history = []
        self.last_distance = 0
        self.draw_action = [0,3,6,9,12,15,18,21]
        self.do_nothing_action = [1,4,7,10,13,16,19,22]
        self.erase_action = [2,5,8,11,14,17,20,23]
        """ 
        012
        3*4
        567
        """
        self.mask_value = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
        self.action_mask = {
            0:self.mask_value[0], 1: self.mask_value[0],2:self.mask_value[0],
            3:self.mask_value[1], 4: self.mask_value[1],5:self.mask_value[1],
            6:self.mask_value[2], 7: self.mask_value[2],8:self.mask_value[2],
            9:self.mask_value[3], 10: self.mask_value[3],11:self.mask_value[3],
            12:self.mask_value[4], 13: self.mask_value[4],14:self.mask_value[4],
            15:self.mask_value[5], 16: self.mask_value[5],17:self.mask_value[5],
            18:self.mask_value[6], 19: self.mask_value[6],20:self.mask_value[6],
            21:self.mask_value[7], 22: self.mask_value[7],23:self.mask_value[7],
        }

    def render(self):
        image = Image.fromarray(self.canvas)
        image.save('test/' + str(PixelDrawEnv.prefix) + "_" + str(PixelDrawEnv.i) + '.png')
        PixelDrawEnv.i += 1

    def reset(self):
        self.canvas = np.zeros((self.canvas_shape, self.canvas_shape), dtype=np.uint8)
        self.history = []
        self.pen_position = (7, 7)
        self.no_drawn_pixels = 0
        return self.state()

    def step(self, a): # ------------------- in functie de cat de departe este pozitia actuala fata de cel mai apropiat punct nedesenat
        """ s_, r, done """
        temp_position = tuple([sum(x) for x in zip(self.pen_position, self.action_mask[a])])
        if len([i for i in temp_position if i < 0 or i >= 28]) > 0:
            return self.state(), -100, True, ''

        self.pen_position = temp_position
        previos_value = self.canvas[self.pen_position]
        if a in self.draw_action:
            self.canvas[self.pen_position] = self.color
            self.no_drawn_pixels += 1
            self.history.append(self.pen_position)
        if a in self.erase_action:
            self.no_drawn_pixels -= 1 if self.canvas[self.pen_position] > 0 else 0
            self.canvas[self.pen_position] = 0
            if self.pen_position in self.history:
                self.history.remove(self.pen_position)
        distance = self.get_distance()
        reward = -1
        test = cdist([self.pen_position],self.history)
        if a in self.draw_action:
            if distance < self.last_distance:
                reward = 10
        if a in self.do_nothing_action:
            reward = 0
        if a in self.erase_action:
            if previos_value > 0 and self.target_original[self.pen_position] == 0:
                reward = 1
            if previos_value > 0 and self.target_original[self.pen_position] > 0:
                reward = -10
        done = False
        if self.no_drawn_pixels >= self.max_drawn_pixels_per_episode:
            done = True
            # reward = -100
        elif distance < 10:
            done = True
            reward = 100
        self.last_distance = distance
        return self.state(), reward, done, ''

    # def get_distance(self):
    #     return (((self.canvas.flatten() - self.target)/255) ** 2).sum()
    def get_distance(self):
        return np.array(list(filter(lambda x: x > 0, (self.target - self.canvas.flatten())))).size

    def state(self):
        img = np.expand_dims(self.canvas, axis=3) / 255
        target = np.expand_dims(self.target_original, axis=3) / 255
        metadata = [self.pen_position]
        return np.array([img, target, metadata])
        # return self.canvas.e
        # return np.concatenate((self.canvas.flatten(),np.array(self.pen_position),np.array([self.no_drawn_pixels])))

    def new_episode(self):
        PixelDrawEnv.prefix += 1

    def get_reward_decision_tree(self, previous, a):
        return {
            'start':self.get_type,
            'draw': lambda x: 'd_empty' if previous==0 else -1,
            'd_empty': lambda x: 10 if self.target_original[self.pen_position] > 0 else -1 * cdist([self.pen_position],self.history[:-1]).min(),
            'do_nothing':0,
            'erase':-1 if previous ==0 else 'e_full',
            'e_full': lambda x: -10 if self.target_original[self.pen_position] > 0 else 10,
        }
    def get_type(self,a):
        if a in self.draw_action: return 'draw'
        if a in self.erase_action: return 'erase'
        return 'do_nothing' # adauga distane

    def get_reward(self, previous,a,key='start'):
        if isinstance(key,float):
            return key
        new_node = self.get_reward_decision_tree(previous, a)[key]()

        return self.get_reward(previous,a,new_node)
