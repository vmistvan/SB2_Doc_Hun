# Reinforcement Learning - Megerősítő Tanulás

# Tippek és trükkök

Ennek a szakasznak az a célja, hogy segítsen megerősítő tanulási kísérletek futtatásában. Tartalmazza az RL-re vonatkozó általános tanácsokat (hol kezdje, melyik algoritmust válassza, hogyan értékelje ki az algoritmust stb.), valamint tippeket és trükköket tartalmaz egyéni környezet használata vagy RL-algoritmus megvalósítása során.

## Jegyzet
__Van egy [videó a YouTube](https://www.youtube.com/watch?v=Ikngt0_DXJg)-on, amely részletesebben lefedi ezt a részt. A diákat [itt is](https://araffin.github.io/slides/rlvs-tips-tricks/) megtalálod.__

## Jegyzet
__Van egy [videónk a Valós világméretű RL-kísérletek tervezéséről és futtatásáról](https://www.youtube.com/watch?v=eZ6ZEpCi6D8), a diák [megtalálhatók az interneten](https://araffin.github.io/slides/design-real-rl-experiments/).__

## Általános tanácsok az Inforcement Learning használatához
TL;DR

1. Olvasson az RL-ről és a Stable Baselines-ről!
2. Ha szükséges, végezzen kvantitatív kísérleteket és hiperparaméter hangolást!
3. Értékelje a teljesítményt egy külön tesztkörnyezet segítségével (ne felejtse el ellenőrizni a burkolatokat (wrappers)!)
4. A jobb teljesítmény érdekében növelje a képzési ráfordítást!

Mint minden más témával, ha az RL-vel szeretne dolgozni, először olvassa el ezt (van egy külön forrásoldalunk a kezdéshez), hogy megértse, mit használ. Azt is javasoljuk, hogy olvassa el a Stable Baselines3 (SB3) dokumentációját, és végezze el az oktatóanyagot. Lefedi az alapvető használatot, és elvezeti Önt a könyvtár fejlettebb koncepcióihoz (pl. visszahívások (callbacks) és wrapperek).

A megerősítéses tanulás több szempontból is különbözik a többi gépi tanulási módszertől. Az ügynök betanításához használt adatokat maga az ügynök gyűjti össze a környezettel való interakciók során (összehasonlítva a felügyelt tanulással, ahol például rögzített adatkészlettel rendelkezik). Ez a függőség ördögi körhöz vezethet: ha az ügynök rossz minőségű adatokat gyűjt (pl. jutalom nélküli pályákat), akkor nem fog javulni, és továbbra is rossz pályákat halmoz fel.

Ez a tényező többek között megmagyarázza, hogy az RL-ben elért eredmények futásonként változhatnak (vagyis amikor csak a pszeudo-véletlen generátor magja változik). Emiatt mindig többször kell futtatnia, hogy kvantitatív eredményeket kapjon.

Az RL-ben elért jó eredmények általában a megfelelő hiperparaméterek megtalálásától függenek. A legújabb algoritmusok (PPO, SAC, TD3, DroQ) általában kevés hiperparaméter-hangolást igényelnek, de ne számítsunk arra, hogy az alapértelmezettek bármilyen környezetben működjenek.

Ezért nagyon ajánljuk, hogy vessen egy pillantást az [RL zoo-ra](https://github.com/DLR-RM/rl-baselines3-zoo) (vagy az eredeti papírokra) a hangolt hiperparaméterekért. Ha egy új problémára alkalmazza az RL-t, a legjobb gyakorlat az automatikus hiperparaméter-optimalizálás. Ez megint benne van az [RL zoo-ban](https://github.com/DLR-RM/rl-baselines3-zoo).

Amikor az RL-t egyéni problémára alkalmazza, mindig normalizálja az ügynök bemenetét (például a VecNormalize használatával PPO/A2C-hez), és nézze meg a más környezetekben végzett általános előfeldolgozást (például Atari, frame-stack stb.). Az egyéni környezetekkel kapcsolatos további tanácsokért olvassa el a __Tippek és trükkök - egyedi környezet létrehozása__ című részt az egyéni környezet létrehozásakor.

## Az RL jelenlegi korlátai
Tisztában kell lennie a megerősítő tanulás jelenlegi [korlátaival](https://www.alexirpan.com/2018/02/14/rl-hard.html).

A modell nélküli RL algoritmusok (azaz az SB-ben megvalósított összes algoritmus) általában nem hatékonyak a mintavételezésben. Sok mintára van szükségük (néha milliónyi interakcióra), hogy megtanuljanak valami hasznosat. Ezért a legtöbb sikert az RL-ben játékokon vagy csak szimulációban érte el. Például az ETH Zurich [ebben a munkájában](https://www.youtube.com/watch?v=aTDkYFZFWug) az ANYmal robotot csak szimulációra képezték ki, majd a való világban tesztelték.

Általános tanácsként a jobb teljesítmény elérése érdekében növelni kell az ügynök költségvetését (a képzési időlépések számát).

A kívánt viselkedés eléréséhez gyakran szakértői tudásra van szükség a megfelelő jutalmazási funkció kialakításához. Ez a jutalomfejlesztés (vagy a [Freek Stulp](http://www.freekstulp.net/) által megalkotott RewArt) több iterációt tesz szükségessé. A jutalom alakításának jó példájaként megtekintheti a [Deep Mimic papírt](https://xbpeng.github.io/projects/DeepMimic/index.html), amely az imitációs tanulást és a megerősítő tanulást ötvözi az akrobatikus mozdulatok elvégzéséhez.

Az RL utolsó korlátja az edzés instabilitása. Vagyis edzés közben hatalmas teljesítménycsökkenést figyelhetünk meg. Ez a viselkedés különösen jelen van a DDPG-ben, ezért a TD3 kiterjesztése megpróbálja megoldani ezt a problémát. Más módszerek, például a TRPO vagy a PPO, megbízható régiót használnak a probléma minimalizálására a túl nagy frissítés elkerülésével.


## Hogyan kell kiértékelni egy RL algoritmust?

## Jegyzet
__Ügynöke kiértékelésekor és az eredmények összehasonlításakor mások eredményeivel fordítson figyelmet a enviroment wrapper-ekre. Az epizódjutalmak vagy az epizódok hosszának módosítása szintén befolyásolhatja az értékelési eredményeket, ami nem feltétlenül kívánatos. Tekintse meg az assessment_policy helper funkciót az Evaluation Helper részben.__

Mivel a legtöbb algoritmus felderítési zajt használ a képzés során, külön tesztkörnyezetre van szükség az ügynök teljesítményének adott időpontban történő értékeléséhez. Javasoljuk, hogy rendszeres időközönként értékelje ügynökét n tesztepizódra vonatkozóan (n általában 5 és 20 között van), és az epizódonkénti jutalmat átlagolja a jó becslés érdekében.
