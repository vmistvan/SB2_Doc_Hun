# Proximal Policy Optimization Algorithms

# MUNKA ALATT, IDEIGLENES BECSEKKOLÁS!!


### John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov
### OpenAI
### {joschu, filip, prafulla, alec, oleg}@openai.com

## Abstract
A megerősítő tanuláshoz a politika gradiens módszereinek új családját javasoljuk, amelyek váltakoznak az adatok mintavétele között a környezettel való interakción keresztül, és a „helyettesítő” célfüggvény optimalizálása között sztochasztikus gradiens emelkedés segítségével. Míg a szabványos házirend gradiens módszerek adatmintánként egy gradiens frissítést hajtanak végre, egy új célfüggvényt javasolunk, amely lehetővé teszi a minibatch frissítések több korszakát. Az új módszerek, amelyeket proximális politikaoptimalizálásnak (PPO) nevezünk, rendelkeznek a bizalmi régió politika optimalizálásának előnyeivel.
tion (TRPO), de sokkal egyszerűbb a megvalósításuk, általánosabbak és jobb a minta összetettsége (empirikusan). Kísérleteink a PPO-t benchmark feladatok gyűjteményén tesztelik, beleértve a szimulált robotmozgást és az Atari-játékokat, és megmutatjuk, hogy a PPO felülmúlja a többi online irányelv gradiens módszerét, és összességében kedvező egyensúlyt talál a minta összetettsége, egyszerűsége és a fali idő között.

## 1 Bevezetés
Az elmúlt években számos különböző megközelítést javasoltak a neurális hálózati függvény közelítőkkel történő megerősítő tanulásra. A vezető versenyzők a mély Q-learning [Mni+15], a „vanília” politikai gradiens módszerek [Mni+16] és a bizalmi régió/természetpolitikai gradiens módszerek [Sch+15b]. Van azonban még mit javítani egy olyan módszer kifejlesztésében, amely méretezhető (nagy modellekre és párhuzamos implementációkra), adathatékony és robusztus (vagyis hiperparaméterek hangolása nélkül is sikeres számos probléma esetén). A Q-learning (függvényközelítéssel) sok egyszerű problémában kudarcot vall1, és kevéssé érthető, a vanília politikai gradiens módszerek gyenge adathatékonysággal és robusztussággal rendelkeznek; és a bizalmi régió házirend-optimalizálása (TRPO) viszonylag bonyolult, és nem kompatibilis azokkal az architektúrákkal, amelyek zajt (például lemorzsolódást) vagy paramétermegosztást (a házirend és az érték függvény között, vagy segédfeladatokat) tartalmaznak.
Ez a cikk a jelenlegi állapot javítására törekszik egy olyan algoritmus bevezetésével, amely eléri a TRPO adathatékonyságát és megbízható teljesítményét, miközben csak elsőrendű optimalizálást alkalmaz.
Új célt javasolunk levágott valószínűségi mutatókkal, amely pesszimista becslést (azaz alsó korlátot) képez a politika teljesítményére vonatkozóan. A házirendek optimalizálása érdekében felváltva mintavételezzük a házirendből származó adatokat, és több optimalizálási időszakot is végrehajtunk a mintavételezett adatokon.
Kísérleteink összehasonlítják a helyettesítő cél különböző változatainak teljesítményét, és azt találták, hogy a vágott valószínűségi arányokkal rendelkező verzió teljesít a legjobban. Összehasonlítjuk a PPO-t számos korábbi irodalmi algoritmussal is. A folyamatos vezérlési feladatoknál jobban teljesít, mint az általunk összehasonlítható algoritmusok. Atari-n lényegesen jobban teljesít (a minta összetettségét tekintve), mint az A2C, és hasonlóan az ACER-hez, bár sokkal egyszerűbb.

## 2 Háttér: A policy optimalizálása
### 2.1. Szabályzati gradiens módszerek
A policy gradiens módszerek úgy működnek, hogy kiszámítják a policy gradiens becslését, és egy sztochasztikus gradiens emelkedési algoritmushoz csatlakoztatják. A leggyakrabban használt gradiensbecslő a következővel rendelkezik:

(1)

ahol πθ egy sztochasztikus politika és ˆAt az előnyfüggvény becslése a t időpontban.
Itt az elvárás 


az empirikus átlagot jelzi egy véges mintakötegre egy olyan algoritmusban, amely a mintavétel és az optimalizálás között váltakozik. Automatikus alkalmazást használó megvalósítások
a differenciáló szoftver olyan célfüggvény felépítésével működik, amelynek gradiense a politikai gradiens becslése; a ˆg becslőt a cél differenciálásával kapjuk
(2)
Bár vonzó több optimalizálási lépést végrehajtani ezen a veszteséges LP G-n ugyanazon a pályán, ez nem kellően indokolt, és empirikusan gyakran pusztítóan nagy házirend-frissítésekhez vezet (lásd a 6.1. szakaszt; az eredmények nem jelennek meg, de hasonlóak voltak vagy rosszabb, mint a „nincs kivágás vagy büntetés” beállítás).

### 2.2 Bizalmi régió metódusok
A TRPO-ban [Sch+15b] egy célfüggvény (a „helyettesítő” cél) maximalizálásra kerül, a politikafrissítés méretére vonatkozó korlátozás függvényében. Pontosabban maximalizálni

(3)
(4)
Itt a θold a házirend-paraméterek vektora a frissítés előtt. Ez a probléma hatékonyan közelítőleg megoldható a konjugált gradiens algoritmussal, miután lineáris közelítést végzünk a célhoz és másodfokú közelítést a kényszerhez.
A TRPO-t igazoló elmélet valójában egy büntetés alkalmazását javasolja megszorítás helyett, azaz a korlátlan optimalizálási probléma megoldását


valamilyen β együtthatóra. Ez abból a tényből következik, hogy egy bizonyos helyettesítő cél (amely kiszámítja
a max. KL over states az átlag helyett) alsó korlátot (azaz pesszimista korlátot) képez a
a politika teljesítménye π. A TRPO kemény kényszert alkalmaz, nem pedig büntetést, mert kemény
hogy egyetlen β-értéket válasszunk, amely jól teljesít a különböző problémák között – vagy akár egyetlenegyen belül is
probléma, ahol a jellemzők a tanulás során változnak. Ezért a célunk elérése érdekében
egy elsőrendű algoritmus, amely a TRPO monoton javítását emulálja, a kísérletek azt mutatják
hogy nem elegendő egyszerűen egy rögzített β büntetési együtthatót választani és optimalizálni a büntetett
objektív egyenlet (5) SGD-vel; további módosítások szükségesek.



valamilyen β együtthatóra. Ez abból a tényből következik, hogy egy bizonyos helyettesítő cél (amely az átlag helyett a max. KL-t számolja ki az állapotok felett) alsó korlátot (azaz pesszimista korlátot) képez a π politika teljesítményére. A TRPO kemény megszorítást alkalmaz, nem pedig büntetést, mert nehéz egyetlen β-értéket kiválasztani, amely jól teljesít a különböző problémák között – vagy akár egyetlen problémán belül is, ahol a jellemzők a tanulás során változnak. Ezért a TRPO monoton javítását emuláló elsőrendű algoritmus céljának eléréséhez a kísérletek azt mutatják, hogy nem elegendő egyszerűen egy rögzített β büntetési együtthatót választani és optimalizálni a büntetett értéket.
objektív egyenlet (5) SGD-vel; további módosítások szükségesek.


## 3 Kivágott Surrogate Objective
Jelölje rt(θ) a valószínűségi arányt rt(θ) = πθ (at | st)
πθold (at | st) , tehát r(θold) = 1. A TRPO maximalizálja a „helyettesítő” célt

(6)
A CP I felső index a konzervatív politikai iterációra utal [KL02], ahol ezt a célt javasolták. Korlátozás nélkül az LCP I maximalizálása túlzottan nagy mértékű szabályzatfrissítéshez vezetne; ezért most megfontoljuk, hogyan módosítsuk a célt, hogy szankcionáljuk a szabályzat azon változtatásait, amelyek az rt(θ)-t 1-ről eltávolítják.

Az általunk javasolt fő cél a következő:

ahol az epsilon egy hiperparaméter, mondjuk = 0,2. E célkitűzés motivációja a következő. A min belül az első tag az LCP I . A második tag, a clip(rt(θ), 1 − , 1 + ) ˆAt, módosítja a helyettesítő célt a valószínűségi arány levágásával, ami eltávolítja az ösztönzést arra, hogy rt az [1 − , 1 + ] intervallumon kívülre helyezze. Végül vesszük a vágott és a le nem vágott objektív minimumát, így a végső cél a levágatlan objektív alsó korlátja (azaz egy pesszimista korlát). Ezzel a sémával csak akkor hagyjuk figyelmen kívül a valószínűségi arány változását, ha az a célt javítaná, és akkor vesszük figyelembe, ha rontja a célt. Figyeljük meg, hogy az LCLIP (θ) = LCP I (θ) a θold körüli első sorrendben (azaz ahol r = 1), azonban eltérőekké válnak, ahogy θ eltávolodik a θoldtól.

1. ábra
egyetlen tagot (azaz egyetlen t-t) ábrázol az LCLIP-ben; vegye figyelembe, hogy az r valószínűségi arányt 1 − vagy 1 + értékre vágjuk, attól függően, hogy az előny pozitív vagy negatív.r
LCLIP
0 1 1+
A > 0r
LCLIP
0 11 −
A < 0

1. ábra: Az LCLIP helyettesítő függvény egy tagját (azaz egyetlen időlépést) ábrázoló diagramok az r valószínűségi arány függvényében, pozitív előnyök (balra) és negatív előnyök (jobb oldala) esetén. Az egyes diagramokon lévő piros kör az optimalizálás kiindulópontját mutatja, azaz r = 1. Vegye figyelembe, hogy az LCLIP ezek közül a kifejezések közül sokat összegez.

A 2. ábra egy másik intuíciós forrást ad az LCLIP helyettesítő objektívvel kapcsolatban. Megmutatja, hogy számos célkitűzés hogyan változik, ahogyan a házirend-frissítési irány mentén interpolálunk, amelyet a proximális házirend-optimalizálással (az algoritmus, amelyet hamarosan bemutatunk) kapunk egy folyamatos ellenőrzési problémára. Láthatjuk, hogy az LCLIP az LCP I alsó korlátja, és büntetés jár a túl nagy házirend-frissítésért.
