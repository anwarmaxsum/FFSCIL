# This is our repository for FFSCIL


Example of Running Scripts:
---------------------------

1. CIFAR100

5-shot: python ttt1v_xpip_dualp_main.py ffscil_cifar100_9tasks_60bases_5ways --rounds_per_task 10 --epochs 1 --fs_epochs 20 --seed 2023 --lr_fs 0.2  --fs_shots 5

1-shot: python ttt1v_xpip_dualp_main.py ffscil_cifar100_9tasks_60bases_5ways --rounds_per_task 10 --epochs 1 --fs_epochs 20 --seed 2023  --lr_fs 0.2  --fs_shots 1

2. MiniImageNet

5-shot: python ttt1v_xpip_dualp_main.py ffscil_imagenetsubset_9tasks_60bases_5ways --rounds_per_task 10 --epochs 1 --fs_epochs 20 --seed 2023 --lr_fs 0.2 --fs_shots 5

1-shot: python ttt1v_xpip_dualp_main.py ffscil_imagenetsubset_9tasks_60bases_5ways --rounds_per_task 10 --epochs 1 --fs_epochs 20 --seed 2023 --lr_fs 0.2 --fs_shots 1 

3. CUB

5-shot: python ttt1v_xpip_dualp_main.py ffscil_cub200_11tasks_100bases_10ways --rounds_per_task 10 --epochs 1 --fs_epochs 20 --seed 2023 --lr_fs 0.2 --fs_shots 5

1-shot: ttt1v_xpip_dualp_main.py ffscil_cub200_11tasks_100bases_10ways --rounds_per_task 10 --epochs 1 --fs_epochs 20 --seed 2023 --lr_fs 0.2 --fs_shots 1
