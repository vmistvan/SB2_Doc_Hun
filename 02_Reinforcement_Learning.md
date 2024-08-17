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
2. Ha szükséges, végezzen kvantitatív kísérleteket és hiperparaméter hangolást
3. Értékelje a teljesítményt egy külön tesztkörnyezet segítségével (ne felejtse el ellenőrizni a burkolatokat!)
4. A jobb teljesítmény érdekében növelje a képzési költségvetést

Mint minden más témával, ha az RL-vel szeretne dolgozni, először olvassa el ezt (van egy külön forrásoldalunk a kezdéshez), hogy megértse, mit használ. Azt is javasoljuk, hogy olvassa el a Stable Baselines3 (SB3) dokumentációját, és végezze el az oktatóanyagot. Lefedi az alapvető használatot, és elvezeti Önt a könyvtár fejlettebb koncepcióihoz (pl. visszahívások és wrapperek).

A megerősítéses tanulás több szempontból is különbözik a többi gépi tanulási módszertől. Az ügynök betanításához használt adatokat maga az ügynök gyűjti össze a környezettel való interakciók során (összehasonlítva a felügyelt tanulással, ahol például rögzített adatkészlettel rendelkezik). Ez a függőség ördögi körhöz vezethet: ha az ügynök rossz minőségű adatokat gyűjt (pl. jutalom nélküli pályákat), akkor nem fog javulni, és továbbra is rossz pályákat halmoz fel.

Ez a tényező többek között megmagyarázza, hogy az RL-ben elért eredmények futásonként változhatnak (vagyis amikor csak a pszeudo-véletlen generátor magja változik). Emiatt mindig többször kell futtatnia, hogy kvantitatív eredményeket kapjon.

Az RL-ben elért jó eredmények általában a megfelelő hiperparaméterek megtalálásától függenek. A legújabb algoritmusok (PPO, SAC, TD3, DroQ) általában kevés hiperparaméter-hangolást igényelnek, de ne számítsunk arra, hogy az alapértelmezettek bármilyen környezetben működjenek.

Ezért nagyon ajánljuk, hogy vessen egy pillantást az RL állatkertbe (vagy az eredeti papírokra) a hangolt hiperparaméterekért. Ha egy új problémára alkalmazza az RL-t, az az automatikus hiperparaméter-optimalizálás. Ez megint benne van az RL állatkertben.

Amikor az RL-t egyéni problémára alkalmazza, mindig normalizálja az ügynök bemenetét (például a VecNormalize használatával PPO/A2C-hez), és nézze meg a más környezetekben végzett általános előfeldolgozást (például Atari, frame-stack stb.). Az egyéni környezetekkel kapcsolatos további tanácsokért olvassa el a Tippek és trükkök című részt az egyéni környezet létrehozásakor.
