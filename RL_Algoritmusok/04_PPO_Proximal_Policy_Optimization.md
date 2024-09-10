# PPO - Proximal Policy Optimization

A Proximal Policy Optimization algoritmus egyesíti az A2C (több dolgozóval) és a TRPO (bizalmi régiót használ az aktor fejlesztésére) ötleteit.

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
Azt tanácsoljuk azonban a felhasználóknak, hogy kezdjék az egyszerű képkocka-halmozással (frame-stacking), mint egyszerűbb, gyorsabb és általában versenyképes alternatívával. További információ a jelentésünkben:
https://wandb.ai/sb3/no-vel-envs/reports/PPO-vs-RecurrentPPO -aka-PPO-LSTM-on-environments-with-masked-velocity-VmlldzoxOTI4NjE4 Lásd még: Procgen papírmelléklet 11. ábra. A gyakorlatban több megfigyelést is egymásra halmozhat a VecFrameStack segítségével.__

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
A teljes tanulási görbék elérhetők a [kapcsolódó PR #110-ben](https://github.com/DLR-RM/stable-baselines3/pull/110).

### PyBullet környezetek
Eredmények a PyBullet benchmarkon (2 millió lépés) 6 mag használatával. A teljes tanulási görbék a kapcsolódó 48-as számban érhetők el.

### Jegyzet
__A gSDE papír hiperparamétereit használtuk (a PyBullet envs-hez hangolva).__
A Gaussian azt jelenti, hogy a strukturálatlan Gauss-zajt használják a feltáráshoz, a gSDE-t (generalized State-Dependent Exploration) pedig egyébként.

| Környezetek | A2C | A2C | PPO | PPO |
| --- | --- | --- | --- | --- |
| | Gaussian | gSDE | Gaussian | gSDE |
| HalfCheetah | 2003 +/- 54 | 2032 +/- 122 | 1976 +/- 479 | 2826 +/- 45 |
| Ant | 2286 +/- 72 | 2443 +/- 89 | 2364 +/- 120 | 2782 +/- 76- |
| Hopper | 1627 +/- 158 | 1561 +/- 220 | 1567 +/- 339 | 2512 +/- 21 |
| Walker2D | 577 +/- 65 | 839 +/- 56 | 1230 +/- 147 | 2019 +/- 64 |

Hogyan lehet megismételni az eredményeket?
Az rl-zoo repo klónozása:

<pre>git clone https://github.com/DLR-RM/rl-baselines3-zoo
cd rl-baselines3-zoo/</pre>

Futtassa a benchmarkot (cserélje ki a $ENV_ID kódot a fent említett envs-re):
<pre>python train.py --algo ppo --env $ENV_ID --eval-episodes 10 --eval-freq 10000</pre>

Ábrázolja az eredményeket (itt csak a PyBullet envs esetén):
<pre>python scripts/all_plots.py -a ppo -e HalfCheetah Ant Hopper Walker2D -f logs/ -o logs/ppo_results
python scripts/plot_from_file.py -i logs/ppo_results.pkl -latex -l PPO</pre>

## Paraméterek
**classstable_baselines3.ppo.PPO(policy, env, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, clip_range_vf=None, normalize_advantage=True, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, use_sde=False, sde_sample_freq=-1, rollout_buffer_class=None, rollout_buffer_kwargs=None, target_kl=None, stats_window_size=100, tensorboard_log=None, policy_kwargs=None, verbose=0, seed=None, device='auto', _init_setup_model=True)[source]
Proximal Policy Optimization algorithm (PPO) (clip version)**

Papír: https://arxiv.org/abs/1707.06347 Kód: Ez a megvalósítás az OpenAI Spinning Up kódját kölcsönzi (https://github.com/openai/spinningup/) https://github.com/ikostrikov/pytorch-a2c -ppo-acktr-gail és Stable Baselines (PPO2 a https://github.com/hill-a/stable-baselines webhelyről)

A PPO bemutatása: https://spinningup.openai.com/en/latest/algorithms/ppo.html

### Paraméterek:
- policy (ActorCriticPolicy) – A használandó irányelvmodell (MlpPolicy, CnnPolicy,…)

- env (Env | VecEnv | str) – A környezet, amelyből tanulni kell (ha regisztrálva van az edzőteremben, str lehet)

- learning_rate (float | Callable[[float], float]) – A tanulási sebesség, ez lehet az aktuális hátralévő haladás függvénye (1-től 0-ig)

- n_steps (int) – Az egyes környezetekben futtatandó lépések száma frissítésenként (azaz a közzétételi puffer mérete n_steps * n_envs ahol n_envs a párhuzamosan futó környezetpéldányok száma) MEGJEGYZÉS: Az n_steps * n_envs értéknek nagyobbnak kell lennie 1-nél (a előny normalizálása) Lásd: https://github.com/pytorch/pytorch/issues/29372

- batch_size (int) – Minibatch méret

- n_epochs (int) – A korszak száma a helyettesítő veszteség optimalizálásakor

- gamma (float) – Kedvezménytényező

- gae_lambda (float) – A torzítás és a variancia kompromisszumos tényezője az Általános Előny Becslőnél

- clip_range (float | Callable[[float], float]) – Vágási paraméter, ez lehet az aktuális hátralévő folyamat függvénye (1-től 0-ig).

- clip_range_vf (None | float | Callable[[float], float]) – Az értékfüggvény vágóparamétere, ez lehet az aktuális hátralévő folyamat függvénye (1-től 0-ig). Ez az OpenAI megvalósításra jellemző paraméter. Ha a None (Nincs) értéket adja meg (alapértelmezett), akkor az értékfüggvényen nem történik vágás. FONTOS: ez a kivágás a jutalom skálázásától függ.

- normalize_advantage (bool) – Normalizálja-e az előnyt vagy sem

- ent_coef (float) – Entrópia együttható a veszteségszámításhoz

- vf_coef (float) – Értékfüggvény-együttható a veszteségszámításhoz

- max_grad_norm (float) – A gradiens kivágásának maximális értéke

- use_sde (bool) – Az általános állapotfüggő feltárást (gSDE) kell-e használni az akciózaj-felderítés helyett (alapértelmezett: False)

- sde_sample_freq (int) – Új zajmátrix mintavétele n lépésenként gSDE használata esetén. Alapértelmezés: -1 (csak minta a közzététel elején)

- rollout_buffer_class (Type[RolloutBuffer] | None) – Használandó közzétételi pufferosztály. Ha nincs, akkor a rendszer automatikusan kiválasztja.

- rollout_buffer_kwargs (Dict[str, Any] | None) – A létrehozáskor a közzétételi pufferbe átadandó kulcsszóargumentumok

- target_kl (float | None) – Korlátozza a KL eltérést a frissítések között, mert a kivágás nem elegendő a nagy frissítések megakadályozásához, lásd a 213. számú problémát (vö. https://github.com/hill-a/stable-baselines/issues/213) A kl div alapértelmezés szerint nincs korlátozva.

- stats_window_size (int) – A közzétételi naplózás ablakmérete, amely megadja az epizódok számát a jelentett sikerességi arány átlagához, az epizód átlagos hosszát és az átlagos jutalmat.

- tensorboard_log (str | Nincs) – a tensorboard naplózási helye (ha nincs, nincs naplózás)

- policy_kwargs (Dict[str, Any] | None) – további argumentumok, amelyeket át kell adni a létrehozási szabályzatnak

- verbose (int) – Bőbeszédűségi szint: 0, ha nincs kimenet, 1 az információs üzenetekhez (mint például az eszköz vagy a használt burkoló), 2 a hibakereső üzenetekhez

- seed (int | None) – Seed a pszeudo véletlen generátorokhoz

- device (device | str) – Eszköz (cpu, cuda, …), amelyen a kódot le kell futtatni. Automatikusra állítva a kód a GPU-n fut le, ha lehetséges.

- _init_setup_model (bool) – Függetlenül attól, hogy ki kell-e építeni a hálózatot a példány létrehozásakor vagy sem.

