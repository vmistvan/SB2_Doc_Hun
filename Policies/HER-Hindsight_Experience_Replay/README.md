
Visszatekintés Élmények visszajátszása - hát ez hülyeség.
Szerintem a
Korábbi Tapasztalatok Visszajátszása talán jobb, de meg kell kérdezni

A HER egy olyan algoritmus, amely a szabályzaton kívüli (off-policy) módszerekkel működik (például DQN, SAC, TD3 és DDPG). A HER azt a tényt használja fel, hogy még ha egy kívánt célt nem is sikerült elérni, más célt is elérhettek a bevezetés során. „Virtuális” átmeneteket hoz létre azáltal, hogy átcímkézi a múltbeli epizódok átmeneteit (megváltoztatja a kívánt célt).
Más megvalósításokhoz képest a jövőbeni célmintavételi stratégia inkluzív: a jelenlegi átmenet használható újramintavételezéskor. 

##Példa
Ez a példa csak a könyvtár és funkciói használatának bemutatására szolgál, és előfordulhat, hogy a trénelt ágensek nem oldják meg a környezeteket. Az optimalizált hiperparaméterek az RL Zoo repository-ban találhatók.

<pre>
from stable_baselines3 import HerReplayBuffer,DDPG,DQN,SAC,TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
From stable_baselines3.common.envs import BitFlippingEnv

model_class=DQN # works also with SAC, DDPG and TD3
N_BITS=15

env=BitFlippingEnv(n_bits=N_BITS,continuous=model_classin[DDPG,SAC,TD3],max_steps=N_BITS)

# Available strategies (cf paper): future, final, episode
goal_selection_strategy="future"# equivalent to GoalSelectionStrategy.FUTURE

# Initialize the model
model=model_class(
  "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy,
    ),
    verbose=1,
)
# Train the model
model.learn(1000)

model.save("./her_bit_env")
# Because it needs access to `env.compute_reward()`
# HER must be loaded with the env
model=model_class.load("./her_bit_env",env=env)

obs,info=env.reset()
for_inrange(100):
    action, _ = model.predict(obs,deterministic=True)
    obs,reward,terminated,truncated,_=env.step(action)
    if terminated or truncated:
        obs,info=env.reset()
</pre>

Reprodukálás:
beszerzés
git clone https://github.com/DLR-RM/rl-baselines3-zoo
cd rl-baselines3-zoo/

Tréning:
python train.py --algo tqc --env parking-v0 --eval-episodes 10 --eval-freq 10000

Eredmények kinyomtatása:
python scripts/all_plots.py -a tqc -e parking-v0 -f logs/ --no-million


##Paraméterek:
<b>buffer_size (int)</b> – Az elemek maximális száma a pufferben
observation_space (Dict) – Megfigyelési tér
action_space (Space) – Akciótér
env (VecEnv | Nincs) – A képzési környezet
device (eszköz | str) – PyTorch eszköz
n_envs (int) – Párhuzamos környezetek száma
optimize_memory_usage (bool) – Memóriatakarékos változat engedélyezése Jelenleg letiltva (lásd: https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702)
handle_timeout_termination (bool) – Az időtúllépés megszüntetését (időkorlát miatt) külön kezelje, és a feladatot végtelen horizontú feladatként kezelje. https://github.com/DLR-RM/stable-baselines3/issues/284
n_sampled_goal (int) – A valódi átmenetenként létrehozandó virtuális átmenetek száma új célok mintavételével.
goal_selection_strategy (GoalSelectionStrategy | str) – Stratégia a célok mintavételéhez az újrajátszáshoz. Az [„episode”, „final”, „future”] közül az egyik.
copy_info_dict (bool) – Az információs szótár másolása és átadása a compute_reward() metódusnak. Kérjük, vegye figyelembe, hogy a másolás lassulást okozhat. Alapértelmezés szerint hamis.


add(obs, next_obs, action, reward, done, infos) 
 Elemeket ad hozzá a pufferhez.
Paraméterek:
obs (Dict[str, ndarray]) –
next_obs (Dict[str, ndarray]) –
action (ndarray) –
reward (ndarray) –
done (ndarray) –
infos (Lista[Dict[str, Any]]) –
Visszatérés típusa: None

extend(*args, **kwargs)
Transition-oknak az új batch-ét (kötegét) adja a pufferhez
Visszatérés típusa: None

reset()
Visszaállítja a puffert.
Visszatérés típusa: None

sample(batch_size, env=None)
Mintaelemek a visszajátszási pufferből.
Paraméterek:
batch_size (int) – A mintaelemek száma
env (VecNormalize | None) – társított VecEnv a megfigyelések/jutalmak (observations/rewards) normalizálására mintavételkor
Visszatér: Samples
Visszatérés típusa: DictReplayBufferSamples

set_env(env)
Beállítja a környezetet.
Paraméterek:
env (VecEnv) –
Visszatérés típusa: None
size()
Visszatér: A puffer jelenlegi mérete
Visszatérés típusa: int

static swap_and_flatten(arr)
Felcseréli majd, majd kisimítja a 0 (puffer_size) és 1 (n_envs) tengelyt az alakzat [n_steps, n_envs, …]-ból (amikor … a jellemzők alakja) [n_steps * n_envs, …]-re (amely fenntartja a sorrendet)
Paraméterek:
arr (ndarray) –
Visszatér:
Visszatérés típusa: ndarray

to_torch(array, copy=True)
Konvertálja a numpy tömböt PyTorch tenzorrá. Megjegyzés: alapértelmezés szerint másolja az adatokat
Paraméterek:
array (ndarray) –
copy (bool) – Másolja-e az adatokat vagy sem (hasznos lehet, hogy elkerülje a dolgok hivatkozással történő megváltoztatását). Ez az argumentum nem működik, ha az eszköz nem a CPU.
Visszatér:
Visszatérés típusa: Tensor

truncate_last_trajectory()
Ha meghívjuk, feltételezzük, hogy a visszajátszási puffer utolsó pályája befejeződött (és csonkoljuk). Ha nem hívjuk, feltételezzük, hogy ugyanazt a pályát folytatjuk (ugyanaz az epizód).
Visszatérés típusa: None

Gól választási stratégiák
Class stable_baselines3.her.GoalSelectionStrategy(value)
Az új célok kiválasztásának stratégiái mesterséges átmenetek létrehozásakor.

