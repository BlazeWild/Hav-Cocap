from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator('HavCocap_new/logs/charades_captioning/lightning_logs/version_58/events.out.tfevents.1773862254.doece5-MS-7D88.835311.0')
ea.Reload()
print("Scalar tags:", ea.Tags()['scalars'])
