# Saisie sur Jetson AGX Orin — rapport de tests

*19 octobre 2026 — conteneur `grasp_bench`, données dans `/data/grasp_bench`.*

**Question posée :** la génération de prises peut-elle tourner sur l'Orin en même temps que le reste de la stack (cuRobo, réseau de RM, commande de la base), ou faut-il la déporter ?

**Réponse à ce stade :** oui, elle peut tourner sur l'Orin, à condition d'accepter environ 1 s par objet et un ralentissement net de la RM.
- **Latence de GraspGenX :** ≈450–550 ms au repos, ≈0,8–1,0 s quand la stack tourne en parallèle.
- **Mémoire :** 0,6 à 1 Go de mémoire GPU, pour environ 46 Go de RAM libres.
- **Effet sur les boucles existantes :**
  - Aucune ne rate son cycle.
  - L'inférence RM passe de 330 à 455 ms en moyenne (≈640 ms au pire), dans un cycle de 1 s.
  - cuRobo MPC passe de 44 à 58 ms (≈106 ms au pire), dans un cycle de 240 ms.
  - La boucle 10 Hz de la base n'est pas affectée (au pire 1,1 ms de retard).
- **Point faible :** le filtrage des prises. Le filtre de collision de GraspGenX, appliqué à une vue unique, laisse passer des prises qui traversent l'étagère.

---

## 1. Conditions de mesure

| Élément | Valeur |
|---|---|
| Matériel | Jetson AGX Orin Developer Kit, GPU SM_87, 62,9 Go de RAM partagée |
| Logiciel | JetPack 7 / L4T R39.2.0, CUDA 13.2, TensorRT 10.16.1.11, Python 3.12 |
| Alimentation | `nvpmodel` MAXN, `jetson_clocks` actif : CPU 12 cœurs à 2,2 GHz, GPU 1,3 GHz, mémoire 3,2 GHz |
| PyTorch | 2.12.1a0+git7269437, compilé pour SM_87 (vérifié intact après toutes les installations) |
| Au repos | RAM 10,5 Go utilisés, GPU 0 % |

**Règle de mesure :** au moins 100 inférences par configuration, précédées de 5 inférences de chauffe. Chaque résultat donne la moyenne et le 95e centile (p95).

## 2. Conteneur `grasp_bench`

Le conteneur a été créé depuis l'image `leeloo_docker:aarch64-jazzy` :

```bash
docker run -d --name grasp_bench --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all --privileged --network host --ipc host \
  -v /data/grasp_bench:/bench -v /dev/bus/usb:/dev/bus/usb \
  --entrypoint sleep leeloo_docker:aarch64-jazzy infinity
```

Ce qui a été ajouté, dans l'ordre :
1. **Les dépendances de GraspGenX.** Le paquet `graspgenx` est installé avec `--no-deps` ; le reste est installé avec un fichier de contraintes qui fige `torch` et `numpy`, pour ne pas écraser le torch SM_87.
2. **torchvision 0.27.0, compilé depuis les sources** contre ce torch (≈25 min). Le wheel est dans `/bench/wheels`.
3. **TensorRT 10.16.1.11 et `trtexec`** (`python3-libnvinfer`, `libnvinfer-bin`), la même version que pour CapaciNet. S'y ajoutent `onnx` et `onnxscript`.
4. **SAM2**, installé sans son extension CUDA. Checkpoint `sam2.1_hiera_tiny.pt`.
5. **Pour HGGD :** open3d 0.20, numba, git-lfs.

**Pièges rencontrés :**
- **Fichiers LFS absents.** Les fichiers volumineux des dépôts (meshes de la pince, profondeurs d'exemple) sont stockés via Git LFS et n'étaient pas téléchargés : il faut lancer `git lfs pull` dans `GraspGenX/` et dans `ext/gripper_descriptions/`. Les benchmarks de latence ne sont pas touchés, car le conditionnement de la pince vient de fichiers JSON qui ne sont pas en LFS.
- **`torch_tensorrt` n'existe pas pour torch 2.12 sur aarch64.** Le chemin TensorRT prévu par GraspGenX est donc inutilisable ; un remplacement ONNX → TensorRT natif a été écrit (`bench/trt_onnx_accel.py`).
- **Configuration DDS invalide.** La variable `CYCLONEDDS_URI` de l'image pointe vers un `profile.xml` absent, réglé pour l'interface réseau d'une autre machine. Pour la Kinect, il faut utiliser `source /bench/rosenv.sh` : domaine ROS 42, Cyclone sur le loopback (`/bench/cyclone_local.xml`), affichage virtuel Xvfb `:98`.

## 3. GraspGenX (candidat principal)

### 3.1 Installation et encodeur

- **Dépôt :** NVlabs/GraspGenX, checkpoint « release » (générateur `epoch_736`, discriminateur `epoch_1056`).
- **Encodeur : PTv3, sans repli PointNet++.** Le checkpoint utilise `ptv3vanilla` pour le générateur comme pour le discriminateur. C'est une implémentation PTv3 en PyTorch pur, qui n'a besoin ni de spconv ni de flash-attn. Elle tourne directement sur SM_87, alors que l'extension PointNet++ ne compile pas ici (sans conséquence).
- **Diffusion :** 20 étapes. Pince utilisée : `robotiq_2f_85`.
- **Test d'installation officiel :** `tests/test_inference_installation.py`, 2 tests passés.

### 3.2 Latence et mémoire au repos

`GraspGenXSampler.sample()` comprend le générateur, le discriminateur et le retrait des points aberrants. L'entrée est un nuage d'objet sous-échantillonné. Valeurs en moyenne / p95 :

| Points | Prises | Total (ms) | Modèle seul (ms) | Mémoire GPU de pointe |
|---|---|---|---|---|
| 1000 | 50 | 445 / 449 | 437 / 439 | 590 Mo |
| 1000 | 100 | 455 / 460 | 447 / 449 | 590 Mo |
| 1000 | 200 | 559 / 567 | 554 / 563 | 590 Mo |
| 2000 | 50 | 504 / 527 | 441 / 444 | 590 Mo |
| 2000 | 100 | 508 / 516 | 447 / 450 | 590 Mo |
| 2000 | 200 | 608 / 609 | 549 / 551 | 591 Mo |
| 3500 | 50 | 615 / 624 | 430 / 433 | 619 Mo |
| 3500 | 100 | 629 / 648 | 443 / 447 | 619 Mo |
| 3500 | 200 | 727 / 733 | 544 / 546 | 619 Mo |
| 7000 | 50 | 1166 / 1169 | 436 / 438 | 799 Mo |
| 7000 | 100 | 1177 / 1180 | 446 / 449 | 799 Mo |
| 7000 | 200 | 1276 / 1280 | 546 / 549 | 799 Mo |
| 10000 | 50 | 1941 / 1947 | 438 / 444 | 1049 Mo |
| 10000 | 100 | 1951 / 1956 | 449 / 454 | 1049 Mo |
| 10000 | 200 | 2058 / 2075 | 556 / 573 | 1049 Mo |

- **Le modèle ne dépend que du nombre de prises demandées.** Il prend ≈440 ms pour 50 ou 100 prises et ≈550 ms pour 200, quel que soit le nombre de points (PTv3 voxelise à 1 cm).
- **Le coût qui grossit avec les points est le prétraitement.** Le retrait des points aberrants calcule toutes les distances entre paires de points (`torch.cdist`), donc son coût augmente avec le carré du nombre de points : +8 ms à 1000 points, +185 ms à 3500, +1,5 s à 10 000. C'est aussi lui qui fait monter la mémoire de pointe.
- **Recommandation : sous-échantillonner le nuage à environ 2000 points** avant l'appel, ou remplacer ce k-NN par une version qui ne compare pas toutes les paires.
- **Données invalidées puis remplacées.** Le premier balayage pour 3500 points et plus a tourné en partie pendant le driver Kinect, qui occupe environ 9 cœurs. Ces mesures ont été jetées et refaites Kinect arrêtée ; le tableau ci-dessus contient les mesures refaites.

Fichiers : `results/sweep_idle.json` (1000 et 2000 points), `results/sweep_idle_rerun.json` (3500 à 10 000 points).

### 3.3 GraspMoE contre diffusion seule

Comparaison faite sur la scène Kinect réelle (section 6), avec un seuil de score de 0,7 et 200 prises demandées.
- **GraspMoE garde 2 à 3 fois plus de prises après le filtre de collision** sur le canard et la manette : 37 contre 12, et 28 contre 11.
- **Ses meilleures prises viennent du dessus**, à moins de 36° de la verticale. Sa branche « boîte englobante » (OBB) fournit 11 à 21 prises par objet.
- **La diffusion seule est 50 à 100 ms plus rapide.** Elle produit davantage de prises latérales, dont certaines traversent l'étagère.

Le benchmark de latence dédié aux deux planificateurs (`bench_graspgenx.py planner`) est prêt mais n'a pas été lancé.

### 3.4 Export ONNX et TensorRT FP16

**Préparé, mais pas encore exécuté.**
- **Ce qui est prêt :** `bench/trt_onnx_accel.py` remplace `torch_tensorrt`. Il exporte la tête de diffusion, appelée 20 fois par inférence, ainsi que les têtes du discriminateur, en ONNX avec un lot dynamique. Il construit ensuite un engine TensorRT FP16 et le branche dans les classes d'enveloppe de GraspGenX. La boucle de diffusion reste en Python et appelle l'engine à chaque étape.
- **Ce qui bloque :** l'encodeur PTv3 n'est pas exportable. Ses opérations dépendent des données (sérialisation des voxels, sélections par masque), et NVIDIA le note aussi dans le dépôt. Le gain attendu se limite donc à la tête de diffusion et au discriminateur.
- **Pour lancer :** `bench_graspgenx.py sweep --tensorrt --tensorrt_precision fp16`.

## 4. Charge « stack » et marge disponible

### 4.1 Charge reproduite (`bench/stack_load.py`)

| Boucle | Reproduction |
|---|---|
| RM (CapaciNet `reachability_pipeline`) | Engine réel `unet3d.trt` (FP16), lot de 9 (grille 3×3), un cycle par seconde |
| cuRobo MPC | `optimize_action_sequence` (robot Franka de l'exemple cuRobo, scène `collision_table`), cycle de 240 ms comme le node m1013, but alterné de ±20 cm |
| Commande de la base | Boucle 10 Hz sur CPU, comme `base_commander.publish_command` |

Cette charge ne reproduit pas tout. **nvblox ne fait plus partie de la stack** (remplacé par cuRobo), donc il n'est pas inclus. Le driver Kinect en OpenGL logiciel, la D405 et le DDS ne sont pas inclus non plus.

### 4.2 Résultats

| Mesure | Stack seule | Stack et GraspGenX en boucle |
|---|---|---|
| RM, inférence moy / p95 / max | 329 / 361 / 388 ms | 455 / 518 / 642 ms |
| cuRobo MPC, moy / p95 / max | 44 / 69 / 71 ms | 58 / 84 / 106 ms |
| Base 10 Hz, période max | 100,2 ms | 101,1 ms |
| GPU occupé (tegrastats, moyenne) | 37 % | 71 % (souvent 99 %) |
| RAM | 16,1 Go / 62,9 Go | 19,5 Go / 62,9 Go |
| CPU | 6 % | 8 % |
| Puissance GPU+SOC | 16,7 W | 25,3 W |

GraspGenX mesuré sous cette charge :

| Points | Prises | Au repos, total / modèle (ms) | Sous charge, total / modèle (ms), p95 total |
|---|---|---|---|
| 2000 | 50 | 504 / 441 | 759 / 658 (p95 849) |
| 2000 | 100 | 508 / 447 | 802 / 699 (p95 899) |
| 2000 | 200 | 608 / 549 | 1000 / 895 (p95 1026) |
| 3500 | 50 | 615 / 430 | 1001 / 666 (p95 1075) |

Les autres configs à 3500 points n'ont pas été mesurées : les tests de charge ont été arrêtés à ta demande.

**Ce qu'il faut en retenir :**
- La saisie est 1,5 à 1,6 fois plus lente quand la stack tourne.
- Aucune boucle ne rate son cycle. En revanche, la RM perd environ 130 ms de marge en moyenne et 250 ms au pire.
- Le GPU est le seul goulot. Il reste de la RAM en abondance, et le CPU est presque libre tant que le driver Kinect logiciel ne tourne pas.

Fichiers : `results/stack_only_*`, `results/stack_ggx_*` (CSV par boucle, logs tegrastats), analyse avec `bench/analyze.py <phase>`.

## 5. Segmentation

**SAM2.1 hiera tiny** a été mesuré une seule fois, dans le test d'utilisation : en bf16, au repos, image 1280×720.
- **Encodeur :** 125–130 ms.
- **Décodeur :** 18–21 ms par objet, avec une boîte comme invite.
- **Masques :** très bons sur le canard, la manette et la figurine Kirby (scores 0,93 à 0,98).

**Pas encore mesuré :**
- **SAM2 sous charge, et sa comparaison fp32 contre bf16.** Le script `bench/bench_seg.py sam2` est prêt.
- **La segmentation géométrique** (plan RANSAC puis clustering DBSCAN). Le script `bench/bench_seg.py geom` est prêt. L'estimation du plan seule prend ≈100 ms sur le nuage Kinect complet.

## 6. Tests d'utilisation sur une scène réelle (sans mouvement du robot)

### 6.1 Capture Kinect

Dossier : `captures/kinect_20261019_064915/`. La scène montre une table et une étagère, avec un canard jaune et une manette bleue sur l'étagère du haut, et une figurine Kirby sur celle du milieu.

**Réglages :** driver `azure_kinect_ros_driver`, profondeur NFOV_UNBINNED, couleur 720P, 5 images par seconde.
- **Pourquoi 5 fps :** à 15 fps avec le nuage calculé par le driver, l'image couleur et le nuage n'arrivaient pas.
- **Nuage de points :** reconstruit à partir de la profondeur alignée sur l'image couleur et des paramètres intrinsèques.

| Fichier | Contenu |
|---|---|
| `rgb_0..4.png` | images couleur 1280×720 |
| `depth_mm_0..4.png` | profondeur alignée, en mm (uint16) |
| `depth_0..4.npy` | même profondeur, en m (float32) |
| `points_0..4.ply` | nuage coloré, environ 510 000 points |
| `intrinsics.json` | paramètres intrinsèques de la caméra |
| `preview.jpg` | aperçu image couleur et profondeur |

Script de capture : `bench/capture_kinect.py`.

### 6.2 Chaîne testée (`bench/use_test.py`)

1. **Repère « table ».** Le plan dominant de la scène, estimé par RANSAC, donne z vers le haut. GraspMoE en a besoin, car il suppose la gravité selon −z. Ici, ce plan est le sol, qui est parallèle à la table et aux étagères.
2. **Segmentation.** Une boîte est donnée par objet, et SAM2 tiny produit le masque.
3. **Nuage de l'objet.** Masque érodé, points trop loin en profondeur écartés, au plus 2000 points.
4. **Prises.** GraspGenX avec la pince 2F-85 et un seuil de score de 0,7.
5. **Filtre de collision.** Comparaison avec le reste de la scène dans un rayon de 0,4 m, seuil de 1 cm.

### 6.3 Résultats

| Objet | Points | GraspMoE : prises au-dessus du seuil → sans collision (score max) | Diffusion seule : idem | Temps GraspGenX (MoE / diffusion) |
|---|---|---|---|---|
| Canard | 1166 | 72 → 37 (0,84) | 50 → 12 (0,79) | 674 / 587 ms |
| Manette | 1910 | 57 → 28 (0,90) | 45 → 11 (0,84) | 689 / 618 ms |
| Kirby | 460 | 46 → 4 (0,72) | 48 → 11 (0,83) | 2772 / 1089 ms |

Autres temps de la chaîne : plan RANSAC ≈100 ms, filtre de collision 100–200 ms par objet. Un objet complet prend donc environ 0,9 à 1 s au repos.

**Analyse qualitative**, faite sur les rendus des 3 meilleures prises :
- **Canard et manette, avec GraspMoE :** les prises viennent du dessus, entre 0 et 36° de la verticale, pince ouverte de part et d'autre de l'objet. Elles sont plausibles.
- **Le filtre de collision de GraspGenX ne suffit pas.** En diffusion seule, des prises du canard arrivent à l'horizontale, au niveau de l'étagère ou en dessous. La pince traverse la planche sans être rejetée, parce que la planche sous l'objet est cachée par l'objet lui-même dans une vue unique.
- **Kirby, environ 3 cm, est à la limite de la Kinect à cette distance.** On n'a que 460 points, les scores sont faibles et les prises dispersées. De plus, GraspGenX relance l'inférence quand il obtient moins de 40 prises au-dessus du seuil, ce qui porte le temps de GraspMoE à 2,8 s.

Dossiers de résultats : `use_tests/kinect_064915_graspmoe/` et `use_tests/kinect_064915_diffusion/`. Ils contiennent :
- `masks.png` : les masques SAM2 sur l'image ;
- `grasps.png` : les prises projetées sur l'image ;
- `<objet>_top.png` : rendu 3D des 3 meilleures prises, vue de côté et vue de dessus (généré par `bench/render_grasps.py`) ;
- `scene.glb` : la scène et les pinces, à ouvrir dans un visualiseur 3D ;
- `<objet>_grasps.npz` : les poses 4×4 dans le repère caméra et le repère table, avec les scores ;
- `summary.json` : les comptes et les temps.

## 7. Repli sans segmentation : E3GNet et HGGD

- **E3GNet : pas de code publié.** Il n'existe que le papier (arXiv 2410.22980) et une page projet. Les 127 ms annoncées sur Xavier NX ne peuvent donc pas être reproduites.
- **HGGD (THU-VCLab, même équipe) : préparé, pas mesuré.**
  - Prêt : dépôt cloné dans `HGGD/`, checkpoints Kinect et RealSense téléchargés dans `ckpt/`, open3d et numba installés.
  - Manque : compiler pytorch3d depuis les sources pour CUDA 13 et torch 2.12.
  - Référence de comparaison : le papier E3GNet mesure HGGD à 418 ms sur Xavier NX.

## 8. Ce qui reste à faire

| Point du plan | État |
|---|---|
| 0. Conditions et marge disponible | Fait |
| 1. Installation et encodeur de GraspGenX | Fait (PTv3 en PyTorch pur) |
| 1. Latence et mémoire selon les prises et les points | Fait (au repos et sous charge) |
| 1. GraspMoE contre diffusion seule | Comparé sur la scène réelle (qualité, temps) ; benchmark de latence dédié prêt, non lancé |
| 1. Export ONNX et TensorRT FP16 | Script prêt, non exécuté ; PTv3 non exportable |
| 1. Qualité vue unique contre nuage fusionné, filtre de collision et IK cuRobo | Vue unique faite (3 objets) ; reste le nuage fusionné cuRobo et l'IK en goalset |
| 2. SAM2 tiny | Au repos fait (130 ms + 20 ms par objet) ; sous charge à faire |
| 2. Segmentation géométrique | Script prêt, non mesuré |
| 3. E3GNet / HGGD | E3GNet sans code ; HGGD bloqué sur pytorch3d |
| 4. Serveur ZMQ sur le poste RTX 4070 | Non fait (accès au poste nécessaire) |
| 5. Chaîne complète sur 20 essais | Non fait (robot et objets nécessaires) |
| 5. Gigue de la RM et de la base pendant la saisie | Fait avec la charge reproduite |

**Recommandations :**
1. Utiliser GraspMoE, avec un nuage d'objet sous-échantillonné à environ 2000 points et 50 à 100 prises demandées : ≈0,7–0,8 s sous charge.
2. Ajouter un filtre géométrique avant cuRobo : rejeter toute prise dont la pince passe sous le plan support, puis valider la collision avec la scène voxelisée de cuRobo plutôt qu'avec la vue unique.
3. Pour les petits objets, utiliser une vue rapprochée (D405 au poignet) ou le nuage fusionné.
4. Avant les essais, fixer la règle de décision : le budget de latence acceptable pour la saisie, et la marge à conserver pour la RM. Avec la saisie en parallèle, l'inférence RM peut atteindre 640 ms dans un cycle de 1 s.
5. Si ce budget est inférieur à 0,5 s, essayer l'export TensorRT FP16 de la tête de diffusion. Sinon, mesurer la déportation sur le poste RTX 4070 (point 4 du plan).

## 9. Arborescence

```
/data/grasp_bench/
├── RAPPORT_grasp_bench.md      ce rapport
├── GraspGenX/  sam2/  HGGD/    dépôts (+ checkpoints GraspGenX dans GraspGenX/ext)
├── ckpt/                       SAM2 tiny, HGGD
├── rm/unet3d.trt               engine RM (copie de CapaciNet)
├── bench/
│   ├── bench_graspgenx.py      balayages latence/mémoire, comparaison des planificateurs, --tensorrt
│   ├── trt_onnx_accel.py       export ONNX → TensorRT natif (remplace torch_tensorrt)
│   ├── stack_load.py           charge RM / cuRobo MPC / base 10 Hz + journaux de cycle
│   ├── run_phase.sh            orchestration charge + tegrastats + benchmark
│   ├── analyze.py              gigue et tegrastats par phase
│   ├── bench_seg.py            SAM2 / segmentation géométrique
│   ├── capture_kinect.py       capture Kinect → png/npy/ply
│   ├── use_test.py             chaîne capture → SAM2 → GraspGenX → collision
│   └── render_grasps.py        rendus 3D des meilleures prises
├── results/                    JSON de latence, CSV de charge, logs tegrastats
├── captures/                   captures Kinect
└── use_tests/                  résultats des tests d'utilisation
```
