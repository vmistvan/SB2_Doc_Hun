# PPO - Proximal Policy Optimization

A Proximal Policy Optimization algoritmus egyesíti az A2C (több dolgozóval) és a TRPO (bizalmi régiót használ a aktor fejlesztésére) ötleteit.

A fő gondolat az, hogy a frissítés után az új irányelv ne legyen túl távol a régi irányelvtől. Ehhez a ppo kivágást (clipping) használ, hogy elkerülje a túl nagy frissítést.

### Jegyzet
__A PPO számos, az OpenAI által nem dokumentált módosítást tartalmaz az eredeti algoritmushoz képest: az előnyök normalizálva vannak, és az értékfüggvény is levágható.__

## Jegyzetek
- Eredeti dokumentum: https://arxiv.org/abs/1707.06347
- A PPO egyértelmű magyarázata az Arxiv Insights csatornán: https://www.youtube.com/watch?v=5P7I-xPq8u8
- OpenAI blogbejegyzés: https://openai.com/research/openai-baselines-ppo
- Spinning Up útmutató: https://spinningup.openai.com/en/latest/algorithms/ppo.html
- 37 implementáció részleteit bemutató blog: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

## Használható számomra?

### Jegyzet
__A PPO visszatérő verziója elérhető a hozzájárulási tárunkban: https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html
Azt tanácsoljuk azonban a felhasználóknak, hogy kezdjék az egyszerű képkocka-halmozással (frame-stacking), mint egyszerűbb, gyorsabb és általában versenyképes alternatívával. További információ a jelentésünkben: https://wandb.ai/sb3/no-vel-envs/reports/PPO-vs-RecurrentPPO -aka-PPO-LSTM-on-environments-with-masked-velocity-VmlldzoxOTI4NjE4 Lásd még: Procgen papírmelléklet 11. ábra. A gyakorlatban több megfigyelést is egymásra halmozhat a VecFrameStack segítségével.__

| Space | Action | Observation |
| --- | --- | --- |
| Discrete | ✔️ | ✔️ |
| Box | ✔️ | ✔️ |
| MultiDiscrete | ✔️ |  ✔️ |
| MultiBinary | ✔️ | ✔️ |
| Dict | ❌ | ✔️ |

## Példa
Ez a példa csak a könyvtár és funkciói használatának bemutatására szolgál, és előfordulhat, hogy a képzett ügynökök nem oldják meg a környezeteket. Az optimalizált hiperparaméterek az RL Zoo repository-ban találhatók.

Tanítson meg egy PPO-ügynököt a CartPole-v1-en 4 környezet használatával.

<pre>import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("CartPole-v1", n_envs=4)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_cartpole")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
</pre>

## Eredmények
### Atari Games
A teljes tanulási görbék elérhetők a kapcsolódó PR #110-ben.

### PyBullet környezetek
Eredmények a PyBullet benchmarkon (2 millió lépés) 6 mag használatával. A teljes tanulási görbék a kapcsolódó 48-as számban érhetők el.
