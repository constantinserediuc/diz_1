import numpy as np
from PIL import Image
from keras.datasets import mnist


class PixelDrawEnv(object):
    i=1
    prefix = 1
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.target = np.array(list(map(lambda x:255 if x > 0 else 0,x_train[0].flatten())))
        self.target_full = np.array(list(filter(lambda x:x>0,self.target))).size
        self.canvas_shape = 28
        # self.num_state = self.canvas_shape **2
        self.num_state = (28,28,1)
        self.num_action = 8
        self.color = 255
        self.canvas = np.zeros((self.canvas_shape, self.canvas_shape), dtype=np.uint8)
        self.pen_position = (7, 7)
        self.no_drawn_pixels = 0
        self.max_drawn_pixels_per_episode = (30 / 100)*self.num_state[0]**2
        self.history = []
        self.wrong_attemps = 50
        """ 
        012
        3*4
        567
        """
        self.action_mask = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

    def render(self):
        image = Image.fromarray(self.canvas)
        image.save('test/' + str(PixelDrawEnv.prefix)+"_"+str(PixelDrawEnv.i) + '.png')
        PixelDrawEnv.i+=1

    def reset(self):
        self.canvas = np.zeros((self.canvas_shape, self.canvas_shape), dtype=np.uint8)
        self.history = []
        self.wrong_attemps = 50
        self.pen_position = (7,7)
        self.no_drawn_pixels = 0
        return self.state()

    def step(self, a):
        """ s_, r, done """
        temp_position = tuple([sum(x) for x in zip(self.pen_position, self.action_mask[a])])
        if len([i for i in temp_position if i < 0 or i >= 28]) > 0:
            return self.state(), self.get_distance(), True, ''

            self.wrong_attemps -= 1
            if self.wrong_attemps == 0:
                return self.state(), -1000, True, ''
            return self.state(), -100, False, ''
        # if temp_position in self.history:
        #     return self.state(), self.get_distance(), False, ''
        self.pen_position = temp_position
        self.canvas[self.pen_position] = self.color
        self.no_drawn_pixels += 1
        distance = self.get_distance()
        done = self.no_drawn_pixels >= self.max_drawn_pixels_per_episode or distance < 10
        # done = self.no_drawn_pixels >= self.max_drawn_pixels_per_episode or distance < 10
        if distance >= self.target_full-2:
            reward = -1
        else:
            reward = 10
        # reward = (28**2) - distance
        return self.state(), reward, done, ''

    # def get_distance(self):
    #     return (((self.canvas.flatten() - self.target)/255) ** 2).sum()
    def get_distance(self):
        return np.array(list(filter(lambda x:x > 0,(self.target - self.canvas.flatten())))).size


    def state(self):
        return np.expand_dims(self.canvas,axis=3)/255

        # return self.canvas.e
        # return np.concatenate((self.canvas.flatten(),np.array(self.pen_position),np.array([self.no_drawn_pixels])))

    def new_episode(self):
        PixelDrawEnv.prefix +=1
# env = PixelDrawEnv()
# import random
# R = 0
# for i in range(4):
#     print(i,"-------------------------")
#     env.reset()
#     while True:
#         print(env.no_drawn_pixels)
#
#         # if self.render: self.env.render()
#
#         a = random.choice(range(8))
#         s_, r, done, info = env.step(a)
#
#         if done:  # terminal state
#             s_ = None
#
#         s = s_
#         R += r
#
#         if done:
#             break
#     env.render()
# print("Total R:", R)
