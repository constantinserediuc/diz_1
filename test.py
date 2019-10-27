import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# plt.style.use('fivethirtyeight')

def print_r(i):
    with open('rewards.txt', 'r') as reader:
        r = reader.readlines()
        r = list(map(lambda x:float(x.strip()),r))
        plt.cla()
        plt.scatter(range(len(r)),r)
        plt.tight_layout()

ani = FuncAnimation(plt.gcf(), print_r, interval=1000)

plt.tight_layout()
plt.show()
