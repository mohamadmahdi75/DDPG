import tensorflow as tf
import numpy as np

import keras
from keras.layers import Dense,Input,LeakyReLU

from keras.losses import MeanSquaredError
from keras.optimizers import Adam

import gym

env = gym.make('Pendulum-v0')



class Memory():

    def __init__(self,size=1000000):

        self.size = size

        self.mem_cntr = 0

        self.state = np.zeros((size,3))
        self.action = np.zeros((size,1))
        self.reward = np.zeros((size,1))
        self.new_state = np.zeros((size,3))

        self.done = np.zeros((size,1))
        print('Object ceated!!')

    def store (self,state,action,reward,new_state,done):

        index = self.mem_cntr % self.size

        self.state[index] = state
        self.action[index] = action
        self.reward[index] = reward
        self.new_state[index] = new_state
        self.done[index] = 1-done

        self.mem_cntr += 1
        

    def __call__(self,batch_size=500):

        index = min(self.mem_cntr,self.size)
        load_range = np.random.choice(index , batch_size)

        state=self.state[load_range]
        action=self.action[load_range]
        reward=self.reward[load_range]
        new_state=self.new_state[load_range]
        done=self.done[load_range]

        state = tf.convert_to_tensor(state)
        action = tf.convert_to_tensor(action,dtype=tf.float32)
        reward = tf.convert_to_tensor(reward,dtype=tf.float32)

        new_state = tf.convert_to_tensor(new_state)
        done = tf.convert_to_tensor(done,dtype=tf.float32)

        return (state,action,reward,new_state,done)



memory = Memory()



def create_actor():

    state = Input(shape=(3,))
    l1 = Dense(250)(state)
    l1_ = LeakyReLU()(l1)

    l2 = Dense(250)(l1_)
    l2_=LeakyReLU()(l2)
    


    l3 = Dense(50)(l2_)
    l3_ = LeakyReLU()(l3)

    out = 2 * Dense(1,activation='tanh')(l3_)

    model = keras.models.Model([state],out)

    model.summary()

    return model


actor = create_actor()
target_actor = create_actor()

target_actor.set_weights(actor.get_weights())

loss_func = MeanSquaredError()


def craete_critic():
    state= Input(shape=(3,))

    action = Input(shape=(1,))

    state_action = tf.concat([state,action],axis = 1)


    l1 = Dense(250)(state_action)
    l1_ = LeakyReLU()(l1)

    l2 = Dense(250)(l1_)
    l2_=LeakyReLU()(l2)
    
    l3 = Dense(50)(l2_)
    l3_ = LeakyReLU()(l3)

    out = Dense(1)(l3_)

    model = keras.models.Model([state,action],out)

    model.summary()

    return model

critic = craete_critic()
target_critic = craete_critic()

target_critic.set_weights(critic.get_weights())


actor_optim = Adam(learning_rate=0.00001)
critic_optim = Adam(learning_rate=0.0001)




def train_actor_critic(state,action,reward,new_state,done):

    # critic
    with tf.GradientTape() as tape:
        future_actions = target_actor(new_state,training=True)
        Y = reward + 0.999 * target_critic([new_state,future_actions],training=True) 

        critic_val =critic([state,action],training=True)

        critic_loss = tf.reduce_mean(tf.math.square(Y - critic_val))

    critic_grad = tape.gradient(critic_loss,critic.trainable_variables)
    critic_optim.apply_gradients(zip(critic_grad , critic.trainable_variables))

    # actor
    with tf.GradientTape() as tape:

        actions__= actor([state],training=True)
        actor_rewards =critic([state,actions__],training=True) 
        loss__ = - tf.reduce_mean(actor_rewards)
    actor_grad = tape.gradient(loss__,actor.trainable_variables)
    actor_optim.apply_gradients(zip(actor_grad,actor.trainable_variables))



def learn(batch_size):
    (state,action,reward,new_state,done)=memory(batch_size)

    train_actor_critic(state,action,reward,new_state,done)

@tf.function
def update_target(T,W,tau=0.01):

    for (t,w) in zip(T,W):
        t.assign(t*(1-tau) + w*tau)





if __name__=='__main__':

    n_games= 120
    ALL_rewards=[]
    for episode in range(n_games):
        state= env.reset().reshape((1,-1))
        RR=0
        step_=0
        episode_reward=[]
        while True:

            action = np.clip(actor(state)+np.random.normal(0.0,0.001),-2.0,2.0)

            new_state,reward,done,_= env.step(action)


            new_state =new_state.reshape((1,-1))
            
            memory.store(state,action,reward,new_state,done)

            learn(batch_size=500)

            update_target(target_critic.weights,critic.weights)
            update_target(target_actor.weights,actor.weights)



            state = new_state
            RR+=reward
            step_+=1
            episode_reward.append(reward)

            if done:
                break
        ALL_rewards.append(np.mean(episode_reward))

        print(f'**episode{episode+1} ; mean_reward={ALL_rewards[episode]} ; steps={step_}**')












