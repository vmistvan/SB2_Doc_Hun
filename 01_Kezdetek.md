# Kezdetek

## Megjegyzés:
__A Stable-Baselines3 (SB3) belsőleg vektorizált környezeteket (VecEnv) használ. Kérjük, olvassa el a kapcsolódó részt, ha többet szeretne megtudni a jellemzőiről és különbségeiről egyetlen Gym környezet használatához képest.__

A legtöbb könyvtár megpróbálja követni a sklearn-szerű szintaxist a megerősítési tanulási algoritmusokhoz.

Íme egy gyors példa az A2C edzésére és futtatására CartPole környezetben:

<pre>import gymnasium as gym

from stable_baselines3 import A2C

env = gym.make("CartPole-v1", render_mode="rgb_array")

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()
</pre>

## Megjegyzés:
__A logger kimenetére és a nevekre vonatkozó magyarázatokat a Logger részben találja.__

Vagy csak tanítson egy modellt egy sorral, ha a környezet regisztrálva van a Gymnasiumban, és ha a szabályzat regisztrálva van:

<pre>from stable_baselines3 import A2C

model = A2C("MlpPolicy", "CartPole-v1").learn(10000)</pre>


