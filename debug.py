import wandb
wandb.init(project="runs-from-for-loop", reinit=True)

a = {'ACC': 0.21376085504342018, 'valence_CCC': 0.0041375658074754695, 'valence_PCC': 0.07448225401572618, 'valence_RMSE': 0.5290929243159855, 'valence_SAGR': 0.5103540414161657, 'arousal_CCC': -0.0021555008776544065, 'arousal_PCC': -0.055325615358744064, 'arousal_RMSE': 0.554782151187253, 'arousal_SAGR': 0.259185036740147}
for ii,jj in a.items():
    print(ii,jj)
    wandb.log({'val_'+str(ii):jj},step=0)
