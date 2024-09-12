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

2. ábra: Helyettesítő célok, amikor interpolálunk a kezdeti θold házirend-paraméter és a frissített házirend-paraméter között, amelyet a PPO egy iterációja után számítunk ki. A frissített házirend KL eltérése körülbelül 0,02 a kezdeti házirendhez képest, és ez az a pont, ahol az LCLIP maximális. Ez a diagram megfelel a Hopper-v1 probléma első házirend-frissítésének, a 6.1. szakaszban megadott hiperparaméterek használatával.

## 4 Adaptív KL büntetési együttható
Egy másik megközelítés, amely a kivágott helyettesítő célkitűzés alternatívájaként, vagy mellette használható, az, hogy a KL divergenciára büntetést alkalmazunk, és a büntetési együtthatót úgy adaptáljuk, hogy elérjük a KL divergencia dtarg valamilyen célértékét. politika frissítése. Kísérleteink során azt találtuk, hogy a KL-büntetés rosszabbul teljesített, mint a kivágott helyettesítő objektív, azonban ide soroltuk, mert ez egy fontos alapérték.
Ennek az algoritmusnak a legegyszerűbb példányában a következő lépéseket hajtjuk végre minden házirend-frissítésnél:
• Több korszaknyi minibatch SGD használatával optimalizálja a KL-büntetett objektívet

(8)
• Számítsa ki d = ˆEt[KL[πθold (· | st), πθ(· | st)]]
– Ha d < dtarg/1,5, β ← β/2
– Ha d > dtarg × 1,5, β ← β × 2
A frissített β-t a rendszer a következő házirend-frissítéshez használja. Ezzel a sémával időnként láthatunk olyan házirend-frissítéseket, ahol a KL eltérés jelentősen eltér a dtarg-tól, azonban ezek ritkák, és a β gyorsan alkalmazkodik. A fenti 1.5 és 2 paramétereket heurisztikusan választottuk, de az algoritmus nem túl érzékeny rájuk. A β kezdeti értéke egy másik hiperparaméter, de a gyakorlatban nem fontos, mert az algoritmus gyorsan módosítja.

## 5 Algoritmus
Az előző szakaszok helyettesítő veszteségei kiszámíthatók és megkülönböztethetők egy tipikus politikai gradiens implementáció kisebb változtatásával. Az automatikus differenciálást használó megvalósításoknál egyszerűen meg kell alkotni az LCLIP vagy LKLP EN veszteséget az LP G helyett, és több lépésben sztochasztikus gradiens emelkedést kell végrehajtani ezen a célon.

A legtöbb varianciacsökkentett előny-függvény becslő számítási technikája V (s) tanult állapot-érték függvényt használ; például az általánosított előnybecslés [Sch+15a], vagy a véges horizontú becslések [Mni+16]-ban. Ha olyan neurális hálózati architektúrát használunk, amely megosztja a paramétereket a házirend- és az értékfüggvény között, akkor olyan veszteségfüggvényt kell használnunk, amely egyesíti a házirend helyettesítőjét és az értékfüggvény hibatagját. Ezt a célt tovább lehet növelni egy entrópia bónusz hozzáadásával az elegendő feltárás biztosítására, amint azt a korábbi munkákban javasolták [Wil92; Mni+16].
Ezeket a kifejezéseket kombinálva a következő célt kapjuk, amely (hozzávetőlegesen) maximalizált
minden iteráció:

 (9)
ahol c1, c2 együtthatók, S pedig entrópiabónuszt, L VFt pedig négyzetes hibaveszteséget (Vθ(st) − Vtargt)2.

Az [Mni+16]-ban népszerűsített és ismétlődő neurális hálózatokhoz jól használható irányelv-gradiens megvalósítási stílus T időlépésre futtatja a házirendet (ahol T sokkal kisebb, mint az epizód hossza), és az összegyűjtött mintákat egy frissítés. Ez a stílus olyan előnybecslőt igényel, amely nem néz túl a T időlépésen. Az [Mni+16] által használt becslés
 (10)
ahol t adja meg az időindexet [0, T]-ben, egy adott T hosszúságú pályaszakaszon belül. Ezt a választást általánosítva használhatjuk az általánosított előnybecslés csonka változatát, amely a (10) egyenletre redukálódik, ha λ = 1:


Az alábbiakban látható egy proximális házirend-optimalizálási (PPO) algoritmus, amely rögzített hosszúságú pályaszegmenseket használ. Minden iteráció, N (párhuzamos) szereplő mindegyike T időlépésnyi adatot gyűjt. Ezután megszerkesztjük a helyettesítő veszteséget ezeken az NT időlépéseken, és optimalizáljuk a minibatch SGD-vel (vagy általában a jobb teljesítmény érdekében Adam [KB14]) K epochákra.

## 6 Kísérlet
### 6.1 A helyettesítő célok összehasonlítása
Először is összehasonlítunk több különböző helyettesítő célt különböző hiperparaméterek alatt. Itt összehasonlítjuk az LCLIP helyettesítő objektívet számos természetes variációval és ablált változattal.


A KL-büntetéshez használhatunk rögzített β büntetési együtthatót vagy adaptív együtthatót a 4. szakaszban leírtak szerint a dtarg KL célérték használatával. Ne feledje, hogy a naplózónában is próbáltunk vágni, de a teljesítmény nem volt jobb.
Mivel az egyes algoritmusváltozatokhoz hiperparamétereket keresünk, egy számítási szempontból olcsó benchmarkot választottunk az algoritmusok teszteléséhez. Nevezetesen 7 db szimulált robotikai feladatot2 alkalmaztunk az OpenAI Gymben [Bro+16], melyek a MuJoCo [TET12] fizikai motort használják. Mindegyiken egymillió lépésnyi edzést végzünk. A kivágáshoz használt hiperparaméterek ( ) és a KL büntetés (β, dtarg) mellett, amelyekre keresünk, a többi hiperparamétert a 3. táblázat tartalmazza.
A házirend ábrázolásához egy teljesen összekapcsolt MLP-t használtunk két, 64 egységből álló rejtett réteggel és tanh nemlinearitásokkal, a Gauss-eloszlás átlagát adva ki változó szórással, követve [Sch+15b; Dua+16]. Nem osztjuk meg a paramétereket a házirend és az értékfüggvény között (tehát a c1 együttható irreleváns), és nem használunk entrópia bónuszt.
Mindegyik algoritmus mind a 7 környezetben futott, mindegyiken 3 véletlenszerű maggal. Az algoritmus minden egyes futtatását úgy értékeltük, hogy kiszámítottuk az utolsó 100 epizód átlagos teljes jutalmát. Az egyes környezetekhez tartozó pontszámokat eltoltuk és skáláztuk úgy, hogy a véletlenszerű házirend 0-t adjon, a legjobb eredményt pedig 1-re állítottuk, és 21 futásból átlagoltuk, hogy minden algoritmusbeállításhoz egyetlen skalárt kapjunk.
Az eredményeket az 1. táblázat mutatja. Megjegyzendő, hogy a pontszám negatív a vágás vagy büntetés nélküli beállításnál, mivel egy környezet (fél gepárd) esetén nagyon negatív pontszámot ad, ami rosszabb, mint a kezdeti véletlenszerű házirend.

| algorithm avg. | normalized score |
| --- | --- |
| No clipping or penalty | -0.39 |
| Clipping,  = 0.1 | 0.76 |
| Clipping,  = 0.2 | 0.82 |
| Clipping,  = 0.3 | 0.70 |
| Adaptive KL dtarg = 0.003 | 0.68 |
| Adaptive KL dtarg = 0.01 | 0.74 |
| Adaptive KL dtarg = 0.03 | 0.71 |
| Fixed KL, β = 0.3 | 0.62 |
| Fixed KL, β = 1. | 0.71 |
| Fixed KL, β = 3. | 0.72 |
| Fixed KL, β = 10. | 0.69 |

1. táblázat: A folyamatos ellenőrzés benchmark eredményei. Átlagos normalizált pontszámok (az algoritmus több mint 21 futtatása, 7 környezetben) minden algoritmus/hiperparaméter-beállításhoz. β-t 1-re inicializáltuk.

## 6.2 Összehasonlítás a folyamatos tartomány más algoritmusaival
Ezután összehasonlítjuk a PPO-t (a 3. szakasz „kivágott” helyettesítő célkitűzésével) számos más szakirodalmi módszerrel, amelyeket folyamatos problémák esetén hatékonynak tartanak. Összehasonlítottuk a következő algoritmusok hangolt implementációival: bizalmi régió politika optimalizálása [Sch+15b], keresztentrópia módszer (CEM) [SL06], vanília politika gradiens adaptív lépésmérettel3, A2C [Mni+16], A2C bizalmi régióval [ Wan+16]. Az A2C az előny aktor kritikus rövidítése, és az A3C szinkron változata, amelyről azt találtuk, hogy ugyanolyan vagy jobb teljesítményt nyújt, mint az aszinkron verzió. A PPO-hoz az előző szakasz hiperparamétereit használtuk, ahol = 0,2. Azt látjuk, hogy a PPO szinte minden folyamatos vezérlési környezetben felülmúlja az előző módszereket.

3. ábra: Több algoritmus összehasonlítása több MuJoCo környezetben, egymillió időlépésre való betanítás.

## 6.3 Bemutató a folyamatos tartományban: Humanoid futás és kormányzás
Annak érdekében, hogy bemutassuk a PPO teljesítményét a nagy dimenziós folyamatos vezérlési problémákkal kapcsolatban, egy 3D-s humanoidot magában foglaló feladatsoron edzünk, ahol a robotnak futnia, kormányoznia kell, és fel kell kelnie a földről, esetleg miközben kockák dobálják. Az általunk tesztelt három feladat a következő: (1) RoboschoolHumanoid: csak előrefelé történő mozgás, (2) RoboschoolHumanoidFlagrun: a célpont helyzete véletlenszerűen változik 200 lépésenként, vagy amikor elérjük a célt, (3) RoboschoolHumanoidFlagrunHarder, ahol a robotot kockákkal dobálják és fel kell kelnie a földről. Lásd az 5. ábrát egy tanult irányelv állóképeinek megjelenítéséhez, és a 4. ábrán a három feladat tanulási görbéiért. A hiperparamétereket a 4. táblázat tartalmazza. Egyidejű munkában Heess et al. [Hee+17] a PPO adaptív KL-változatát (4. szakasz) használta a 3D-s robotok helyváltoztatási irányelveinek megismerésére.

5. ábra: A RoboschoolHumanoidFlagrun-tól tanult házirend állókép-keretei. Az első hat képkockában a robot egy cél felé fut. Ezután a pozíció véletlenszerűen megváltozik, és a robot megfordul, és az új cél felé fut.

### 6.4 Összehasonlítás más algoritmusokkal az Atari tartományban
A PPO-t az Arcade Learning Environment [Bel+15] benchmarkon is futtattuk, és összehasonlítottuk az A2C [Mni+16] és az ACER [Wan+16] jól hangolt implementációival. Mindhárom algoritmus esetében ugyanazt a házirend-hálózati architektúrát használtuk, mint az [Mni+16]-ban. A PPO hiperparamétereit az 5. táblázat tartalmazza. A másik két algoritmushoz olyan hiperparamétereket használtunk, amelyeket úgy hangoltunk, hogy maximalizáljuk a teljesítményt ezen a referenciaértéken.
Az eredmények és a tanulási görbék táblázata mind a 49 játékra vonatkozóan a B. függelékben található. A következő két pontozási mutatót vesszük figyelembe: (1) epizódonkénti átlagos jutalom a teljes edzési időszak alatt (ami a gyors tanulást segíti elő), és (2) átlagos jutalom per epizód. epizód az edzés utolsó 100 epizódjából (ami kedvez a végső teljesítménynek). A 2. táblázat az egyes algoritmusok által „nyert” játékok számát mutatja, ahol a győztest úgy számítjuk ki, hogy a három próba pontozási mutatóját átlagoljuk.

| | A2C | ACER | PPO | Tie |
| --- | --- | --- | --- | --- |
| (1) avg. episode reward over all of training | 1 | 18 | 30 | 0 |
| (2) avg. episode reward over last 100 episodes | 1 | 28 | 19 | 1 |
2. táblázat: Az egyes algoritmusok által „nyert” játékok száma, ahol a pontozási mutatót három próba átlaga alapján számítják ki.

## 7 Következtetés
Bevezettük a proximális házirend-optimalizálást, a házirend-optimalizálási módszerek egy olyan családját, amely több sztochasztikus gradiens emelkedést használ az egyes szabályzatfrissítések végrehajtásához. Ezek a módszerek a bizalom-régió módszerek stabilitásával és megbízhatóságával rendelkeznek, de sokkal egyszerűbb a megvalósításuk, mindössze néhány sornyi kódváltást igényelnek a vanília házirend gradiens megvalósításához, amely általánosabb beállításokban alkalmazható (például közös architektúra használatakor a házirendhez és értékfüggvény), és jobb általános teljesítményt nyújtanak.

## 8 Köszönetnyilvánítás
Köszönet Rocky Duannak, Peter Chennek és másoknak az OpenAI-nál az éleslátó megjegyzésekért.

## References
<pre>[Bel+15] M. Bellemare, Y. Naddaf, J. Veness, and M. Bowling. “The arcade learning environment: An evaluation platform for general agents”. In: Twenty-Fourth International Joint Conference on Artificial Intelligence. 2015.
[Bro+16] G. Brockman, V. Cheung, L. Pettersson, J. Schneider, J. Schulman, J. Tang, and W. Zaremba. “OpenAI Gym”. In: arXiv preprint arXiv:1606.01540 (2016).
[Dua+16] Y. Duan, X. Chen, R. Houthooft, J. Schulman, and P. Abbeel. “Benchmarking Deep Reinforcement Learning for Continuous Control”. In: arXiv preprint arXiv:1604.06778 (2016).
[Hee+17] N. Heess, S. Sriram, J. Lemmon, J. Merel, G. Wayne, Y. Tassa, T. Erez, Z. Wang, A. Eslami, M. Riedmiller, et al. “Emergence of Locomotion Behaviours in Rich Environments”. In: arXiv preprint arXiv:1707.02286 (2017).
[KL02] S. Kakade and J. Langford. “Approximately optimal approximate reinforcement learning”. In: ICML. Vol. 2. 2002, pp. 267–274.
[KB14] D. Kingma and J. Ba. “Adam: A method for stochastic optimization”. In: arXiv preprint arXiv:1412.6980 (2014).
[Mni+15] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, et al. “Human-level control through deep reinforcement learning”. In: Nature 518.7540 (2015), pp. 529–533.
[Mni+16] V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. P. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu. “Asynchronous methods for deep reinforcement learning”. In: arXiv preprint arXiv:1602.01783 (2016).
[Sch+15a] J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel. “High-dimensional continuous control using generalized advantage estimation”. In: arXiv preprint arXiv:1506.02438 (2015).
[Sch+15b] J. Schulman, S. Levine, P. Moritz, M. I. Jordan, and P. Abbeel. “Trust region policy optimization”. In: CoRR, abs/1502.05477 (2015).
[SL06] I. Szita and A. L¨orincz. “Learning Tetris using the noisy cross-entropy method”. In: Neural computation 18.12 (2006), pp. 2936–2941.
[TET12] E. Todorov, T. Erez, and Y. Tassa. “MuJoCo: A physics engine for model-based control”. In: Intelligent Robots and Systems (IROS), 2012 IEEE/RSJ International Conference on. IEEE. 2012, pp. 5026–5033.
[Wan+16] Z. Wang, V. Bapst, N. Heess, V. Mnih, R. Munos, K. Kavukcuoglu, and N. de Freitas. “Sample Efficient Actor-Critic with Experience Replay”. In: arXiv preprint arXiv:1611.01224 (2016).
[Wil92] R. J. Williams. “Simple statistical gradient-following algorithms for connectionist reinforcement learning”. In: Machine learning 8.3-4 (1992), pp. 229–256.</pre>

## A Hyperparameters


Hyperparameter | Value
--- | ---
Horizon (T) | 2048
Adam stepsize | 3 × 10−4
Num. epochs | 10
Minibatch size | 64
Discount (γ) | 0.99
GAE parameter (λ) | 0.95

Table 3: PPO hyperparameters used for the Mujoco 1 million timestep benchmark.


Hyperparameter | Value
--- | ---
Horizon (T) | 512
Adam stepsize | ∗
Num. epochs | 15
Minibatch size | 4096
Discount (γ) | 0.99
GAE parameter (λ) | 0.95
Number of actors | 32 (locomotion), 128 (flagrun)
Log stdev. of action distribution | LinearAnneal(−0.7, −1.6)

Table 4: PPO hyperparameters used for the Roboschool experiments. Adam stepsize was adjusted based on the target value of the KL divergence.


Hyperparameter | Value
--- | ---
Horizon (T) | 128
Adam stepsize | 2.5 × 10−4 × α
Num. epochs | 3
Minibatch size | 32 × 8
Discount (γ) | 0.99
GAE parameter (λ) | 0.95
Number of actors | 8
Clipping parameter  | 0.1 × α
VF coeff. c1 (9) | 1
Entropy coeff. c2 (9) | 0.01

Table 5: PPO hyperparameters used in Atari experiments. α is linearly annealed from 1 to 0 over the course of learning.


## B Teljesítmény további Atari játékokon

Itt bemutatjuk a PPO és az A2C összehasonlítását egy nagyobb, 49 Atari-játékból álló gyűjteményben. A 6. ábra a három véletlenszerű mag tanulási görbéit mutatja, míg a 6. táblázat az átlagos teljesítményt mutatja.


6. ábra: A PPO és az A2C összehasonlítása mind a 49 ATARI játékon, amelyek az OpenAI Gymben szerepeltek a megjelenés időpontjában.

| | A2C | ACER | PPO |
| --- | --- | --- | --- |
| Alien | 1141.7 | 1655.4 | 1850.3 |
| Amidar | 380.8 | 827.6 | 674.6 |
| Assault | 1562.9 | 4653.8 | 4971.9 |
| Asterix | 3176.3 | 6801.2 | 4532.5 |
| Asteroids | 1653.3 | 2389.3 | 2097.5 |
| Atlantis | 729265.3 | 1841376.0 | 2311815.0 |
| BankHeist | 1095.3 | 1177.5 | 1280.6 |
| BattleZone | 3080.0 | 8983.3 | 17366.7 |
| BeamRider | 3031.7 | 3863.3 | 1590.0 |
| Bowling | 30.1 | 33.3 | 40.1 |
| Boxing | 17.7 | 98.9 | 94.6 |
| Breakout | 303.0 | 456.4 | 274.8 |
| Centipede | 3496.5 | 8904.8 | 4386.4 |
| ChopperCommand | 1171.7 | 5287.7 | 3516.3 |
| CrazyClimber | 107770.0 | 132461.0 | 110202.0 |
| DemonAttack | 6639.1 | 38808.3 | 11378.4 |
| DoubleDunk | -16.2 | -13.2 | -14.9 |
| Enduro | 0.0 | 0.0 | 758.3 |
| FishingDerby | 20.6 | 34.7 | 17.8 |
| Freeway | 0.0 | 0.0 | 32.5 |
| Frostbite | 261.8 | 285.6 | 314.2 |
| Gopher | 1500.9 | 37802.3 | 2932.9 |
| Gravitar | 194.0 | 225.3 | 737.2 |
| IceHockey | -6.4 | -5.9 | -4.2 |
| Jamesbond | 52.3 | 261.8 | 560.7 |
| Kangaroo | 45.3 | 50.0 | 9928.7 |
| Krull | 8367.4 | 7268.4 | 7942.3 |
| KungFuMaster | 24900.3 | 27599.3 | 23310.3 |
| MontezumaRevenge | 0.0 | 0.3 | 42.0 |
| MsPacman | 1626.9 | 2718.5 | 2096.5 |
| NameThisGame | 5961.2 | 8488.0 | 6254.9 |
| Pitfall | -55.0 | -16.9 | -32.9 |
| Pong | 19.7 | 20.7 | 20.7 |
| PrivateEye | 91.3 | 182.0 | 69.5 |
| Qbert | 10065.7 | 15316.6 | 14293.3 |
| Riverraid | 7653.5 | 9125.1 | 8393.6 |
| RoadRunner | 32810.0 | 35466.0 | 25076.0 |
| Robotank | 2.2 | 2.5 | 5.5 |
| Seaquest | 1714.3 | 1739.5 | 1204.5 |
| SpaceInvaders | 744.5 | 1213.9 | 942.5 |
| StarGunner | 26204.0 | 49817.7 | 32689.0 |
| Tennis | -22.2 | -17.6 | -14.8 |
| TimePilot | 2898.0 | 4175.7 | 4342.0 |
| Tutankham | 206.8 | 280.8 | 254.4 |
| UpNDown | 17369.8 | 145051.4 | 95445.0 |
| Venture | 0.0 | 0.0 | 0.0 |
| VideoPinball | 19735.9 | 156225.6 | 37389.0 |
| WizardOfWor | 859.0 | 2308.3 | 4185.3 |
| Zaxxon | 16.3 | 29.0 | 5008.7 |

6. táblázat: A PPO és A2C átlagos végeredményei (legutóbbi 100 epizód) Atari játékokon 40 millió játékkocka (10 millió) után
időlépések).
