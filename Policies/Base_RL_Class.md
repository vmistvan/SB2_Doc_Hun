# Base RL Class - Abstract alap osztály RL algoritmusoknak

<cit>class stable_baselines3.common.base_class.BaseAlgorithm(policy, env, learning_rate, policy_kwargs=None, stats_window_size=100, tensorboard_log=None, verbose=0, device='auto', support_multi_env=False, monitor_wrapper=True, seed=None, use_sde=False, sde_sample_freq=-1, supported_action_spaces=None)</cit>

## Paraméterek:
- __policy__ (BasePolicy) – A használandó házirend-modell (MlpPolicy, CnnPolicy,…)
- __env__ (Env | VecEnv | str | None) – A környezet, amelyből tanulni kell (ha regisztrálva van az edzőteremben, akkor str. Lehet, hogy nincs a betanított modellek betöltéséhez)
- __learning_rate__ (float | Callable[[float], float]) – tanulási sebesség az optimalizáló számára, ez lehet az aktuális hátralévő haladás függvénye (1-től 0-ig)
- __policy_kwargs__ (Dict[str, Any] | None) – További argumentumok, amelyeket át kell adni a létrehozási szabályzatnak
- __stats_window_size__ (int) – A közzétételi naplózás ablakmérete, amely megadja az epizódok számát a jelentett sikerességi arány átlagához, az epizód átlagos hosszát és az átlagos jutalmat.
- __tensorboard_log__ (str | Nincs) – a tensorboard naplózási helye (ha nincs, nincs naplózás)
- __verbose__ (int) – Bőbeszédűségi szint: 0, ha nincs kimenet, 1 az információs üzenetekhez (mint például az eszköz vagy a használt burkoló), 2 a hibakereső üzenetekhez
- __device__ (device | str) – Eszköz, amelyen a kódnak futnia kell. Alapértelmezés szerint megpróbál egy Cuda-kompatibilis eszközt használni, és visszatér a CPU-hoz, ha ez nem lehetséges.
- __support_multi_env__ (bool) – Az algoritmus támogatja-e a képzést több környezetben (mint az A2C-ben)
- __monitor_wrapper__ (bool) – Környezet létrehozásakor, hogy becsomagolja-e a monitor burkolójába vagy sem.
- __seed__ (int | Nincs) – Seed a pszeudo véletlen generátorokhoz
- __use_sde__ (bool) – Az általános állapotfüggő feltárást (gSDE) kell-e használni az akciózaj-felderítés helyett (alapértelmezett: hamis)
- __sde_sample_freq__ (int) – Új zajmátrix mintavétele n lépésenként gSDE használata esetén. Alapértelmezés: -1 (csak minta a közzététel elején)
- __supported_action_spaces__ (Tuple[Típus[szóköz], ...] | Nincs) – Az algoritmus által támogatott műveleti terek.

## get_env()
Az aktuális környezetet adja vissza (Nincs lehet, ha nincs megadva).
Visszatér: A jelenlegi környezet
Visszatérés típusa: VecEnv | None

## get_parameters()
Visszaadja az ügynök paramétereit. Ide tartoznak a különböző hálózatok paraméterei, pl. kritikusok (értékfüggvények) és irányelvek (pi függvények).
Visszatér: A mappelt objektumok nevéből PyTorch state-dicts -ekre.
Visszatérés típusa: Dict[str, Dict]

## get_vec_normalize_env()
Ha létezik, küldje vissza a képzési env VecNormalize burkolóját.
Visszatér: A VecNormalize env.
Visszatérés típusa: VecNormalize | None

## abstract learn(total_timesteps, callback=Nincs, log_interval=100, tb_log_name='run', reset_num_timesteps=Igaz, progress_bar=False)__
Visszaad egy betanított modellt.

### Paraméterek:
- __total_timesteps__ (int) – A képzéshez szükséges minták (env lépések) teljes száma
- __callback__ (None | Callable | List[BaseCallback] | BaseCallback) – minden lépésben visszahívás(ok) az algoritmus állapotával.
- __log_interval__ (int) – on-policy algoritmusoknál (pl. PPO, A2C, …) ez a betanítási iterációk száma (azaz log_interval * n_steps * n_envs timesteps) a naplózás előtt; a szabályzaton kívüli algok esetében (pl. TD3, SAC, …) ez a naplózás előtti epizódok száma.
- __tb_log_name__ (str) – a TensorBoard naplózási futtatásának neve
- __reset_num_timesteps__ (bool) – vissza kell-e állítani az aktuális időlépési számot (naplózásban használatos)
- __progress_bar__ (bool) – Folyamatjelző sáv megjelenítése tqdm és rich használatával.
- __self__ (SelfBaseAlgoritm) –

__Visszatér:__ a képzett modell.

__Visszatérés típusa:__ SelfBaseAlgoritm


## classmethod __load__(path, env=None, device='auto', custom_objects=None, print_system_info=False, force_reset=True, **kwargs)

Betölti a modellt egy zip-fájlból. Figyelmeztetés: a __load__ a semmiből újra létrehozza a modellt, nem frissíti a helyén! Helyi __load__ esetén használja helyette a __set_parameters__t.

### Paraméterek:
- __path__ (str | path | BufferedIOBase) – a fájl (vagy fájlszerű) elérési útja, ahonnan az ügynök betölthető
- __env__ (Env | VecEnv | None) – a betöltött modell futtatására szolgáló új környezet (Nincs lehet, ha csak egy betanított modelltől van szüksége előrejelzésre) elsőbbséget élvez bármely mentett környezettel szemben.
- __device__ (device | str) – Eszköz, amelyen a kódnak futnia kell.
- __custom_objects__ (Dict[str, Any] | None) – A betöltéskor cserélendő objektumok szótára. Ha egy változó kulcsként szerepel ebben a szótárban, akkor az nem lesz deszerializálva, hanem a megfelelő elem kerül felhasználásra. Hasonló a custom_objects-hez a keras.models.load_model fájlban. Hasznos, ha olyan objektum van a fájlban, amelyet nem lehet deszerializálni.
- __print_system_info__ (bool) – Kinyomtatja-e a rendszerinformációkat a mentett modellből és az aktuális rendszerinformációkat (hasznos a betöltési problémák elhárításához)
- __force_reset__ (bool) – A __reset()__ hívás kényszerítése edzés előtt, hogy elkerülje a váratlan viselkedést. Lásd: https://github.com/DLR-RM/stable-baselines3/issues/597
- __kwargs__ – extra argumentumok a modell megváltoztatásához betöltéskor

Visszatér: új modellpéldány betöltött paraméterekkel

Visszatérés típusa: SelfBaseAlgoritm

## propertylogger: Logger
Getter a logger objektumhoz

## predict(observation, state=None, episode_start=None, deterministic=False)
Beszerzi a házirend-műveletet egy megfigyelésből (és opcionálisan rejtett állapotból). Tartalmazza a cukorbevonatot a különböző megfigyelések kezelésére (például a képek normalizálására).

### Paraméterek:
- __observation__ (ndarray | Dict[str, ndarray]) – a bemeneti megfigyelés
- __state__ (Tuple[ndarray, ...] | Nincs) – Az utolsó rejtett állapotok (Nincs lehet, az ismétlődő házirendekben használatos)
- __episode_start__ (ndarray | Nincs) – Az utolsó maszkok (Nincs lehet, az ismétlődő házirendekben használatos) ez az epizódok kezdetének felel meg, ahol az RNN rejtett állapotait vissza kell állítani.
- __deterministic__ (bool) – Változó arra, hogy visszaadja-e a determinisztikus műveleteket vagy sem.

__Visszatér:__ a modell művelete és a következő rejtett állapot (ismétlődő szabályzatokban használatos)

__Visszatérés típusa:__ Tuple[ndarray, Tuple[ndarray, …] | None]

## save(path, exclude=None, include=None)
