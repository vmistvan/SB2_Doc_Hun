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

- Recurrent policies: ❌
- Multi processing: ✔️
- Gym spaces:

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
- __policy (ActorCriticPolicy)__ – A használandó irányelvmodell (MlpPolicy, CnnPolicy,…)

- __env (Env | VecEnv | str)__ – A környezet, amelyből tanulni kell (ha regisztrálva van az edzőteremben, str lehet)

- __learning_rate (float | Callable[[float], float])__ – A tanulási sebesség, ez lehet az aktuális hátralévő haladás függvénye (1-től 0-ig)

- __n_steps (int)__ – Az egyes környezetekben futtatandó lépések száma frissítésenként (azaz a közzétételi puffer mérete n_steps * n_envs ahol n_envs a párhuzamosan futó környezetpéldányok száma) MEGJEGYZÉS: Az n_steps * n_envs értéknek nagyobbnak kell lennie 1-nél (a előny normalizálása) Lásd: https://github.com/pytorch/pytorch/issues/29372

- __batch_size (int)__ – Minibatch méret

- __n_epochs (int)__ – A korszak száma a helyettesítő veszteség optimalizálásakor

- __gamma (float)__ – Kedvezménytényező

- __gae_lambda (float)__ – A torzítás és a variancia kompromisszumos tényezője az Általános Előny Becslőnél

- __clip_range (float | Callable[[float], float])__ – Vágási paraméter, ez lehet az aktuális hátralévő folyamat függvénye (1-től 0-ig).

- __clip_range_vf (None | float | Callable[[float], float])__ – Az értékfüggvény vágóparamétere, ez lehet az aktuális hátralévő folyamat függvénye (1-től 0-ig). Ez az OpenAI megvalósításra jellemző paraméter. Ha a None (Nincs) értéket adja meg (alapértelmezett), akkor az értékfüggvényen nem történik vágás. FONTOS: ez a kivágás a jutalom skálázásától függ.

- __normalize_advantage (bool)__ – Normalizálja-e az előnyt vagy sem

- __ent_coef (float)__ – Entrópia együttható a veszteségszámításhoz

- __vf_coef (float)__ – Értékfüggvény-együttható a veszteségszámításhoz

- __max_grad_norm (float)__ – A gradiens kivágásának maximális értéke

- __use_sde (bool)__ – Az általános állapotfüggő feltárást (gSDE) kell-e használni az akciózaj-felderítés helyett (alapértelmezett: False)

- __sde_sample_freq (int)__ – Új zajmátrix mintavétele n lépésenként gSDE használata esetén. Alapértelmezés: -1 (csak minta a közzététel elején)

- __rollout_buffer_class (Type[RolloutBuffer] | None)__ – Használandó közzétételi pufferosztály. Ha nincs megadva, akkor a rendszer automatikusan kiválasztja.

- __rollout_buffer_kwargs (Dict[str, Any] | None)__ – A létrehozáskor a közzétételi pufferbe átadandó kulcsszóargumentumok

- __target_kl (float | None)__ – Korlátozza a KL eltérést a frissítések között, mert a kivágás nem elegendő a nagy frissítések megakadályozásához, lásd a 213. számú problémát (vö. https://github.com/hill-a/stable-baselines/issues/213) A kl div alapértelmezés szerint nincs korlátozva.

- __stats_window_size (int)__ – A közzétételi naplózás ablakmérete, amely megadja az epizódok számát a jelentett sikerességi arány átlagához, az epizód átlagos hosszát és az átlagos jutalmat.

- __tensorboard_log (str | Nincs)__ – a tensorboard naplózási helye (ha nincs, nincs naplózás)

- __policy_kwargs (Dict[str, Any] | None)__ – további argumentumok, amelyeket át kell adni a létrehozási szabályzatnak

- __verbose (int)__ – Kifejtési szint: 0, ha nincs kimenet, 1 az információs üzenetekhez (mint például az eszköz vagy a használt burkoló), 2 a hibakereső üzenetekhez

- __seed (int | None)__ – Seed a pszeudo véletlen generátorokhoz

- __device (device | str)__ – Eszköz (cpu, cuda, …), amelyen a kódot le kell futtatni. Automatikusra állítva a kód a GPU-n fut le, ha lehetséges.

- ___init_setup_model (bool)__ – Ki kell-e építeni a hálózatot a példány létrehozásakor vagy sem?

## collection_rollouts(env, callback, rollout_buffer, n_rollout_steps)
Tapasztalatokat gyűjt a jelenlegi házirend használatával, és kitölti a RolloutBuffert. A "rollout" kifejezés itt a modell-free fogalomra vonatkozik, és nem használható a modellalapú RL-ben vagy tervezésben használt bevezetés fogalmával.

### Paraméterek:
- __env (VecEnv)__ – A képzési környezet
- __callback (BaseCallback)__ – Visszahívás, amely minden lépésnél (és a közzététel elején és végén) meg lesz hívva.
- __rollout_buffer (RolloutBuffer)__ – Puffer, amelyet közzé kell tenni
- __n_rollout_steps (int)__ – A gyűjtendő tapasztalatok száma környezetenként

__Visszatér:__ Igaz, ha a függvény legalább n_rollout_steps összegyűjtésével tért vissza, hamis, ha a visszahívás idő előtt leállt.

__Visszatérés típusa:__ bool

## get_env()
Az aktuális környezetet adja vissza (Nincs lehet, ha nincs megadva).

__Visszatér:__ A jelenlegi környezet

__Visszatérés típusa:__ VecEnv | None

## get_parameters()
Visszaadja az ügynök paramétereit. Ide tartoznak a különböző hálózatok paraméterei, pl. kritikusok (értékfüggvények) és irányelvek (pi függvények).

__Visszatér:__ Leképezés az objektumok nevéből PyTorch __state-dict__-ekre.

__Visszatérés típusa:__ Dict[str, Dict]

## get_vec_normalize_env()
Ha létezik, a train env (edzési környezet) VecNormalize wrapperét küldi vissza.

__Visszatér:__ A VecNormalize env.

__Visszatérés típusa:__ VecNormalize | Egyik sem

## learn(total_timesteps, callback=None, log_interval=1, tb_log_name='PPO', reset_num_timesteps=True, progress_bar=False)
Visszaad egy betanított modellt.

### Paraméterek:
- __total_timesteps (int)__ – A képzéshez szükséges minták (env lépések) teljes száma

- __callback (None | Callable | Lista[BaseCallback] | BaseCallback)__ – a minden lépésben meghívott visszahívás(ok) az algoritmus állapotával.

- __log_interval (int)__ – on-policy algoritmusoknál (pl. PPO, A2C, …) ez a betanítási iterációk száma (azaz log_interval * n_steps * n_envs timesteps) a naplózás előtt; a szabályzaton kívüli algok esetében (pl. TD3, SAC, …) ez a naplózás előtti epizódok száma.

- __tb_log_name (str)__ – a TensorBoard naplózási futtatásának neve

- __reset_num_timesteps (bool)__ – vissza kell-e állítani az aktuális időlépési számot (naplózásban használatos)

- __progress_bar (bool)__ – folyamatjelző sáv megjelenítése tqdm és rich használatával.

- __self (SelfPPO)__ – (sajátmagára hivatkozás - fordító)

__Visszatér:__ a képzett modell

__Visszatérés típusa:__ SelfPPO

## classmethod load(path, env=None, device='auto', custom_objects=None, print_system_info=False, force_reset=True, **kwargs)
Betölti a modellt egy zip-fájlból.
__Figyelmeztetés: a terhelés a semmiből újra létrehozza a modellt, nem frissíti a helyén! Helyi terhelés (in-place load) esetén használja helyette a set_parameters paramétert.__

### Paraméterek:
- __path (str | Path | BufferedIOBase)__ – a fájl (vagy fájlszerű) elérési útja, ahonnan az ügynök betölthető

- __env (Env | VecEnv | None)__ – a betöltött modell futtatására szolgáló új környezet (None értékű lehet, ha csak egy betanított modellt használunk előrejelzésre) elsőbbséget élvez bármely mentett környezettel szemben.

- __device (device | str)__ – Eszköz, amelyen a kódnak futnia kell.

- __custom_objects (Dict[str, Any] | None)__ – A betöltéskor cserélendő objektumok szótára. Ha egy változó kulcsként szerepel ebben a szótárban, akkor az nem lesz deszerializálva, hanem a megfelelő elem kerül felhasználásra. Hasonló a keras.models.load_model custom_objects-hez. Hasznos, ha olyan objektum van a fájlban, amelyet nem lehet deszerializálni.

- __print_system_info (bool)__ – Kinyomtatja-e a rendszerinformációkat a mentett modellből és az aktuális rendszerinformációkat (hasznos a betöltési problémák elhárításához)

- __force_reset (bool)__ – A reset() hívás kényszerítése edzés előtt, hogy elkerülje a váratlan viselkedést.
        Lásd: https://github.com/DLR-RM/stable-baselines3/issues/597

- __kwargs__ – extra argumentumok a modell megváltoztatásához betöltéskor

__Visszatér:__ új modellpéldány betöltött paraméterekkel

__Visszatérés típusa:__ SelfBaseAlgoritm

## property logger: Logger
Getter a logger objektumhoz.

## predict(observation, state=None, episode_start=None, deterministic=False)
Beszerzi a házirend-műveletet egy megfigyelésből (és opcionálisan rejtett állapotból). Tartalmazza a cukorbevonatot a különböző megfigyelések kezelésére (például a képek normalizálására).

### Paraméterek:
- __observation (ndarray | Dict[str, ndarray])__ – a bemeneti megfigyelés

- __state (Tuple[ndarray, ...] | None)__ – Az utolsó maszkok (None értékű lehet, ha az ismétlődő házirendekben/recurrent policies használatos)

- __episode_start (ndarray | None)__ – Az utolsó maszkok (Nincs lehet, az ismétlődő házirendekben használatos) ez az epizódok kezdetének felel meg, ahol az RNN rejtett állapotait vissza kell állítani.

- __deterministic (bool)__ – Hogy visszaadja-e a determinisztikus műveleteket vagy sem.

__Visszatér:__ a modell akciója és a következő rejtett állapot (ismétlődő szabályzatokban használatos)

__Visszatérés típusa:__ Tuple[ndarray, Tuple[ndarray, …] | None]

## save(path, exclude=None, include=None)
Elmenti az objektum összes attribútumát és a modell paramétereit egy zip-fájlba.

### Paraméterek:
- __path (str | Path | BufferedIOBase)__ – annak a fájlnak az elérési útja, ahová az rl ügynököt menteni kell

- __exclude (Iterable[str] | None)__ – azon paraméterek neve, amelyeket az alapértelmezetteken kívül ki kell zárni

- __include (Iterable[str] | None)__ – azon paraméterek neve, amelyek kizárhatók, de mindenképpen szerepelniük kell

__Visszatérés típusa:__ None

## set_env(env, force_reset=True)
Ellenőrzi a környezet érvényességét, és ha koherens, akkor beállítja aktuális környezetnek. Továbbá minden nem vektorizált env-t vektorizált ellenőrzött paraméterekbe csomagol: - megfigyelési_terület - cselekvési_terület / observation_space - action_space

### Paraméterek:
- __env (Env | VecEnv)__ – A politika tanulásának környezete

- __force_reset (bool)__ – A reset() hívás kényszerítése edzés előtt, hogy elkerülje a váratlan viselkedést. Lásd a problémát: https://github.com/DLR-RM/stable-baselines3/issues/597

__Visszatérés típusa:__ None

## set_logger(logger)
Setter a logger objektumhoz.

### Figyelmeztetés
__Egyéni naplózó objektum átadásakor ez felülírja a tensorboard_log és a konstruktornak átadott részletes beállításokat.__

### Paraméterek:
logger (Logger) –

__Visszatérés típusa:__ None

## set_parameters(load_path_or_dict, exact_match=True, device='auto')
Paraméterek betöltése egy adott zip-fájlból vagy egy beágyazott szótárból, amely különböző modulok paramétereit tartalmazza (lásd a get_parameters).

### Paraméterek:
- __load_path_or_iter__ – A mentett adatok helye (útvonal vagy fájlszerű, lásd a mentést), vagy a házirend által használt nn.Module paramétereket tartalmazó beágyazott szótár. A szótár az objektumneveket a torch.nn.Module.state_dict() által visszaadott állapotszótárba képezi le.

- __exact_match (bool)__ – Ha igaz, akkor a megadott paramétereknek tartalmazniuk kell az egyes modulokhoz és azok paramétereihez tartozó paramétereket, ellenkező esetben kivételt vet fel. Ha False értékre van állítva, ez csak bizonyos paraméterek frissítésére használható.

- __device (device | str)__ – Eszköz, amelyen a kódnak futnia kell.

- __load_path_or_dict (str | Dict[str, Tenzor])__ –

__Visszatérés típusa:,__ None

## set_random_seed(seed=None)
Beállítja a pszeudo-véletlen generátorok magját (python, numpy, pytorch, gym, action_space)

### Paraméterek:
seed (int | None) –

__Visszatérés típusa:__ None

## train()
Frissíti a házirendet a jelenleg összegyűjtött közzétételi pufferrel.

__Visszatérés típusa:__ None


# PPO Policies / házirendek

## stable_baselines3.ppo.MlpPolicy
alias a ActorCriticPolicy-hoz

## class stable_baselines3.common.policies.ActorCriticPolicy(observation_space, action_space, lr_schedule, net_arch=None, activation_fn=<class 'torch.nn.modules.activation.Tanh'>, ortho_init=True, use_sde=False, log_std_init=0.0, full_std=True, use_expln=False, squash_output=False, features_extractor_class=<class 'stable_baselines3.common.torch_layers.FlattenExtractor'>, features_extractor_kwargs=None, share_features_extractor=True, normalize_images=True, optimizer_class=<class 'torch.optim.adam.Adam'>, optimizer_kwargs=None)

Irányelvosztály a szereplőkritikus algoritmusokhoz (rendelkezésre áll irányelv és érték előrejelzése is). Az A2C, PPO és hasonlók használják.

### Paraméterek:
- __observation_space (Space)__ – Megfigyelési tér

- __action_space (Space)__ – Akciótér

- __lr_schedule (Callable[[float], float])__ – Tanulási ütem ütemezése (lehet állandó)

- __net_arch (List[int] | Dict[str, List[int]] | None)__ – Az irányelvek és az értékhálózatok specifikációja.

- __activation_fn (Type[Module]__ – Aktiválási funkció

- __ortho_init (bool)__ – Használja-e az ortogonális inicializálást vagy sem

- __use_sde (bool)__ – Használja-e az állapotfüggő feltárást vagy sem

- __log_std_init (float)__ – A log szórásának kezdeti értéke

- __full_std (bool)__ – GSDE használatakor használjunk-e (n_features x n_actions) paramétereket az std-hez a csak (n_features,) helyett

- __use_expln (bool)__ – Használja az expln() függvényt az exp() helyett a pozitív szórás biztosításához (vö. PPO_Paper). Lehetővé teszi a szórás nulla felett tartását és megakadályozza, hogy túl gyorsan növekedjen. A gyakorlatban általában elég az exp() is.

- __squash_output (bool)__ – Függetlenül attól, hogy a kimenetet tanh függvénnyel kell-e tömöríteni, ez lehetővé teszi a határok biztosítását gSDE használatakor.

- __features_extractor_class (Type[BaseFeaturesExtractor])__ – Használandó szolgáltatások kivonó.

- __features_extractor_kwargs (Dict[str, Any] | None)__ – A szolgáltatáskivonónak átadandó kulcsszóargumentumok.

- __share_features_extractor (bool)__ – Ha igaz (True), a szolgáltatáskivonó meg van osztva a házirend és az értékhálózat között.

- __normalize_images (bool)__ – Normalizálja-e a képeket vagy sem, osztva 255,0-val (alapértelmezés szerint igaz)

- __optimizer_class (Type[Optimizer])__ – A használandó optimalizáló, alapértelmezés szerint th.optim.Adam

- __optimizer_kwargs (Dict[str, Any] | None)__ – További kulcsszó-argumentumok, a tanulási sebesség kivételével, amelyeket át kell adni az optimalizálónak

## evaluate_actions(obs, actions)
Értékeli a cselekvéseket a jelenlegi politika szerint, figyelembe véve a megfigyeléseket.

### Paraméterek:
- __obs (Tensor | Dict[str, Tensor])__ – Megfigyelés

- __actions (Tensor)__ – Akciók

__Visszatér:__ a becsült érték, a műveletek megtételének logikus valószínűsége és a műveleteloszlás entrópiája.

__Visszatérés típusa:__ Tuple[Tensor, Tensor, Tensor | None]

## extract_features(obs, features_extractor=None)
Ha szükséges, feldolgozza a megfigyelést, és kivonja a jellemzőket.

### Paraméterek:
- __obs (Tensor | Dict[str, Tensor])__ – Megfigyelés

- __features_extractor (BaseFeaturesExtractor | None)__ – A használható szolgáltatások kivonó. Ha nincs, akkor a self.features_extractor használatos.

__Visszatér:__ A kivont jellemzők. Ha a jellemzők kivonója nincs megosztva, akkor egy sort ad vissza a színész jellemzőivel és a kritikus jellemzőivel.

__Visszatérés típusa:__ Tensor | Tuple[tenzor, tenzor]

