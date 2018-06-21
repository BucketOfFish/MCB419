import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

reward = [ 18.48608, 21.15056, 24.56328, 32.06112, 37.72344, 44.5864,  50.0136
, 49.19776, 46.2612,  46.30632, 46.87752, 48.98184, 42.81528, 47.07744
, 47.26024, 50.73008, 46.0892,  54.41888, 49.30392, 49.01256]
reward_std = [ 9.70413705, 12.07145823, 16.37769156, 22.40730404, 26.24439133
, 27.74863787, 32.85142819, 32.56028917, 31.53450029, 31.6892668
, 32.31323225, 32.72769764, 28.62972025, 29.06575872, 33.06970615
, 33.54501102, 31.65074852, 37.37234588, 33.59690862, 33.76900634]
reward_std = [i/np.sqrt(250) for i in reward_std]

reward_salience = [ 18.89192, 23.50224, 27.17112, 33.45408, 43.23672, 46.15216, 47.94472
, 43.30432, 57.09632, 57.41416, 56.32544, 62.15512, 53.886,  54.93552
, 56.66624, 55.60904, 53.32792, 60.48248, 64.13328, 53.93904]
reward_salience_std = [ 10.05223112, 13.96085898, 18.00789721, 20.36728521, 28.22627862
, 29.37832401, 31.4633918,  31.03954958, 34.6680267,  35.02643442
, 36.58852213, 37.51870698, 35.68953606, 35.4567942,  36.0316864
, 35.07178713, 33.65270796, 38.28011699, 38.95149205, 32.33762914]
reward_salience_std = [i/np.sqrt(250) for i in reward_salience_std]

x = np.arange(1, 21, 1)*5
sns.set_style("whitegrid")
plt.errorbar(x, reward, reward_std, fmt='o-', markersize=8, label='RL')
plt.errorbar(x, reward_salience, reward_salience_std, fmt='o-', markersize=8, label='RL w/ salience')
plt.xlabel("Training Games")
plt.ylabel("Reward (Evaluated 500 Times)")
plt.title("Average Reward vs. Training Time (Evaluated by Training 25 Times)")
xmax = max(x) + 5
ymax = max([i+j for (i, j) in zip(reward, reward_std)] + [i+j for (i, j) in zip(reward_salience, reward_salience_std)]) + 5
plt.xlim(0, xmax)
plt.ylim(0, ymax)
plt.legend(loc=4)
plt.show()
