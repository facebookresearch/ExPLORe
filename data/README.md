First, download the data from here: https://drive.google.com/drive/folders/1jxBQE1adsFT1sWsfatbhiZG6Zkf3EW0Q

After downloading the data, for each environment (Pick and Place, Closed Drawer, and Blocked Drawer 1) organize it into the structure shown in the code block below. The "prior data" in the context of COG refers to the trajectories from the first subtask in the compositional task, and the "task data" are trajectories from the second subtask that leads to the completed task. The drawer environments each have their own "prior data", but share the "task data" "drawer_task.npy".

```
[ENV_NAME]/PRIOR_DATA.npy
[ENV_NAME]/TASK_DATA.npy
[ENV_NAME]/successful/prior_success.npy
[ENV_NAME]/successful/task_success.npy
```
In our code, we assume ENV_NAME is the following: pickplace for the Pick and Place task (Widow250PickTray-v0), closeddrawer_small for the Closed Drawer task (Widow250DoubleDrawerOpenGraspNeutral-v0), and blockeddrawer1_small for the Blocked Drawer 1 task (Widow250DoubleDrawerCloseOpenGraspNeutral-v0). PRIOR_DATA.npy and TASK_DATA.npy can be named as you wish.

We also provide some successful trajectories used for evaluation and visualization purposes. These trajectories can be found under ./data/successful_trajectories. Place them in a successful folder in the respective environment when setting up the data.

The drawer prior and task data may be big. The code below was used to subsample from the datasets in drawer environment and also zero out the rewards. Note that even if subsampling is not done, it is still important to zero out the rewards in the "prior data". It may be inconsistent in the datasets, but the prior data sometimes contains +1 rewards for completing the subtask, which would need to be zeroed since they do not complete the compositional task. The rewards in the task data should align with the compositional task and are fine to keep. The curating of these reward labels is primarily for the RLPD baseline.

```
path = 'NPYFILEPATH'
data = np.load(path, allow_pickle=True)
for i in range(len(data)):
    for j in range(len(data[i]['rewards'])):
        data[i]['rewards'][j] *= 0
data_small = data[np.random.choice(range(len(data)), size=SMALLER_SIZE_HERE (ex. 2500), replace=False)]
np.save('ZEROED_SMALL_PATH', data_small)
```
