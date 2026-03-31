import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ea = EventAccumulator('HavCocap_new/logs/charades_captioning/lightning_logs/version_72/events.out.tfevents.1774196779.cslam-training.6340.0')
ea.Reload()

steps = [e.step for e in ea.Scalars('loss_step')]
train_loss = [e.value for e in ea.Scalars('loss_step')]

plt.figure(figsize=(10, 6))
plt.plot(steps, train_loss, alpha=0.6, label='Step Loss')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('Training Loss per Step (version_72)')
plt.legend()
plt.tight_layout()
plt.savefig('training_loss_steps.png', dpi=300)
print("Saved step loss plot to training_loss_steps.png")
