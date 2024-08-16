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

__get_env()__
Az aktuális környezetet adja vissza (Nincs lehet, ha nincs megadva).
Visszatér: A jelenlegi környezet
Visszatérés típusa: VecEnv | None

__get_parameters()__
Visszaadja az ügynök paramétereit. Ide tartoznak a különböző hálózatok paraméterei, pl. kritikusok (értékfüggvények) és irányelvek (pi függvények).
Visszatér: A mappelt objektumok nevéből PyTorch state-dicts -ekre.
Visszatérés típusa: Dict[str, Dict]

__get_vec_normalize_env()__
Ha létezik, küldje vissza a képzési env VecNormalize burkolóját.
Visszatér: A VecNormalize env.
Visszatérés típusa: VecNormalize | None

__abstract learn(total_timesteps, callback=Nincs, log_interval=100, tb_log_name='run', reset_num_timesteps=Igaz, progress_bar=False)__
Visszaad egy betanított modellt.

### Paraméterek:
- __total_timesteps__ (int) – A képzéshez szükséges minták (env lépések) teljes száma
- __callback__ (None | Callable | List[BaseCallback] | BaseCallback) – minden lépésben visszahívás(ok) az algoritmus állapotával.
- __log_interval__ (int) – on-policy algoritmusoknál (pl. PPO, A2C, …) ez a betanítási iterációk száma (azaz log_interval * n_steps * n_envs timesteps) a naplózás előtt; a szabályzaton kívüli algok esetében (pl. TD3, SAC, …) ez a naplózás előtti epizódok száma.
- __tb_log_name__ (str) – a TensorBoard naplózási futtatásának neve
- __reset_num_timesteps__ (bool) – vissza kell-e állítani az aktuális időlépési számot (naplózásban használatos)
- __progress_bar__ (bool) – Folyamatjelző sáv megjelenítése tqdm és rich használatával.
- __self__ (SelfBaseAlgoritm) –

__Visszatér:__ a képzett modell
__Visszatérés típusa:__ SelfBaseAlgoritm
