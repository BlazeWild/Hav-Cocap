import json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ea = EventAccumulator('HavCocap_new/logs/charades_captioning/lightning_logs/version_58/events.out.tfevents.1773862254.doece5-MS-7D88.835311.0')
ea.Reload()

with open('Metrics_Log.md', 'w') as f:
    f.write('# Training Metrics Log (version_58)\n\n')
    
    f.write('## Epoch Metrics\n\n')
    f.write('| Step | Training Loss | Validation CIDEr |\n')
    f.write('|------|---------------|------------------|\n')
    
    loss_data = {e.step: e.value for e in ea.Scalars('loss_epoch')}
    cider_data = {e.step: e.value for e in ea.Scalars('CIDEr')}
    
    all_steps = sorted(set(loss_data.keys()).union(set(cider_data.keys())))
    
    for step in all_steps:
        tl = f"{loss_data[step]:.4f}" if step in loss_data else "-"
        vc = f"{cider_data[step]:.4f}" if step in cider_data else "-"
        f.write(f'| {step} | {tl} | {vc} |\n')
        
    f.write('\n## Step Loss Info\n\n')
    step_data = ea.Scalars('loss_step')
    f.write(f'Total logged training steps: {len(step_data)}\n\n')
    
    # Let's just output every 500th step to keep it readable
    f.write('| Step | Loss |\n')
    f.write('|------|------|\n')
    for i, e in enumerate(step_data):
        if i % 500 == 0 or i == len(step_data) - 1:
            f.write(f'| {e.step} | {e.value:.4f} |\n')
            
print("Successfully wrote Metrics_Log.md")
