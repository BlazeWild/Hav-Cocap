import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ea = EventAccumulator('HavCocap_new/logs/charades_captioning/lightning_logs/version_58/events.out.tfevents.1773862254.doece5-MS-7D88.835311.0')
ea.Reload()

# Get data
try:
    epochs = [e.step for e in ea.Scalars('loss_epoch')]
    train_loss = [e.value for e in ea.Scalars('loss_epoch')]
    
    val_cider = [e.value for e in ea.Scalars('CIDEr')]
    val_cider_epochs = [e.step for e in ea.Scalars('CIDEr')]
except Exception as e:
    print("Error extracting scalars:", e)

# Plotting Training Loss
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Epoch / Step')
ax1.set_ylabel('Training Loss', color=color)
ax1.plot(epochs, train_loss, color=color, marker='o', label='Train Loss')
ax1.tick_params(axis='y', labelcolor=color)

# Create a second y-axis to plot Validation CIDEr (as a proxy for val performance since val loss wasn't logged)
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Validation CIDEr Score', color=color)  
ax2.plot(val_cider_epochs, val_cider, color=color, marker='s', linestyle='--', label='Val CIDEr')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Training Loss and Validation CIDEr over Epochs (version_58)')
fig.tight_layout()  

plt.savefig('training_validation_metrics.png', dpi=300)
print("Saved plot to training_validation_metrics.png")
