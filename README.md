# leeloo_grasping

Évaluation de la génération de prises (GraspGenX, SAM2, HGGD) sur Jetson AGX Orin
(JetPack 7, CUDA 13.2, TensorRT 10.16) en parallèle de la stack leeloo
(cuRobo MPC, réseau de reachability CapaciNet).

**Rapport complet : [RAPPORT_grasp_bench.md](RAPPORT_grasp_bench.md).**

## Contenu

| Dossier | Contenu |
|---|---|
| `bench/` | scripts de benchmark, charge stack, capture Kinect, test d'utilisation, rendus |
| `env/` | contraintes pip (torch SM_87 figé), config Cyclone DDS locale, environnement ROS |
| `results/` | latences JSON, CSV des boucles sous charge, logs tegrastats |
| `captures/` | capture Kinect (RGB, profondeur alignée en mm, intrinsèques) |
| `use_tests/` | sorties des tests d'utilisation (masques, prises, scènes .glb, poses .npz) |

Les scripts supposent le conteneur `grasp_bench` (image `leeloo_docker:aarch64-jazzy`)
avec ce dossier monté en `/bench` (cf. section 2 du rapport).

## Non versionné (à récupérer)

| Élément | Source |
|---|---|
| `GraspGenX/` | https://github.com/NVlabs/GraspGenX @ `b9429097728cb1c430dd78b92edf17ba318aad03` (+ `git lfs pull`, checkpoints auto-téléchargés depuis HF `adithyamurali/GraspGenXModel`) |
| `sam2/` | https://github.com/facebookresearch/sam2 @ `2b90b9f5ceec907a1c18123530e92e794ad901a4` |
| `HGGD/` | https://github.com/THU-VCLab/HGGD @ `69580ffefc70247e982f08f3d80d619561919b5e` |
| torchvision | https://github.com/pytorch/vision tag `v0.27.0`, compilé contre torch 2.12.1a0+git7269437 (SM_87) |
| `ckpt/sam2.1_hiera_tiny.pt` | https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt |
| `ckpt/HGGD_*_checkpoint` | https://cloud.tsinghua.edu.cn/d/e3edfc2c8b114513b7eb/ |
| `rm/unet3d.trt` | engine CapaciNet (`capacitynet2:/home/ros2_ws/src/capacitynet/config/unet3d.trt`) |
| `captures/*/depth_*.npy`, `points_*.ply` | régénérables depuis `depth_mm_*.png` + `intrinsics.json` |
