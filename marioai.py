!pip install gym_super_mario_bros==7.3.0 nes_py
!pip install gym
!apt-get install -y python-opengl
!apt-get install python-opengl -y
!pip install pyglet
import gym
import numpy as np


#import the game
import gym_super_mario_bros
#iport the joypad wrapper
from nes_py.wrappers import JoypadSpace
#import the SIMPLIFIED controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

SIMPLE_MOVEMENT

#setting up game
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env.action_space
env.observation_space.shape


#creating a flag
done = True
#loop through each frame in the game
for step in range(100000):
  if done:
    #start the game to begin with
    env.reset()
  #do random action
  state, reward, done, info = env.step(env.action_space.sample())  #action_space.sample() provides random action
  #show the game on the screen
  env.render()
#close the game
env.close()


state = env.reset()
env.step(1) #state
env.step(1)[1] #reward
env.step(1)[3]


# installing pytorch
!pip3 install torch
!pip3 install torch torchvision torchaudio
!pip install stable-baselines3
#import frame stacker and greyscale
from gym.wrappers import GrayScaleObservation
#import vectorization wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
#import matplotlib
from matplotlib import pyplot as plt


# 1. Create the base environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
# 2. Simplify the controls 
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# 3. Grayscale
env = GrayScaleObservation(env, keep_dim=True)
# 4. Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])
# 5. Stack the frames
env = VecFrameStack(env, 4, channels_order='last')


state = env.reset()


state, reward, done, info = env.step([5])


plt.figure(figsize=(20,16))
for idx in range(state.shape[3]):
    plt.subplot(1,4,idx+1)
    plt.imshow(state[0][:,:,idx])
plt.show()


# Import os for file path management
import os 
# Import PPO for algos
from stable_baselines3 import PPO
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback


class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True


CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'


# Setup model saving callback
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)


# This is the AI model started
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, 
            n_steps=512)


# Train the AI model, this is where the AI model starts to learn
model.learn(total_timesteps=1000000, callback=callback)


# Load model
model = PPO.load('./train/best_model_1000000')


state = env.reset()


# Start the game 
state = env.reset()
# Loop through the game
while True: 
    
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()


