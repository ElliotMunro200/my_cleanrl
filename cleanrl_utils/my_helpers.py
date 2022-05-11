import wandb

def epsilon_by_frame(frame_idx): # decaying exponential as function of frame index. Explore chance goes from 1.0->0.01.
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500
    epsilon_by_frame = epsilon_final + (epsilon_start - epsilon_final) * np.exp(
        -1. * frame_idx / epsilon_decay)
    return epsilon_by_frame

def train_log(metric_name, metric_val, frame_idx): # general function to log a metric with certain name to WandB
    # against its frame index.
    metric_val = float(metric_val)
    wandb.log({metric_name: metric_val}, step=frame_idx)
    print("Logging "+str(metric_name)+" of: "+str(metric_val)+", at frame index: "+str(frame_idx)+", to WandB")