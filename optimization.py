from torch.optim import Adam, lr_scheduler
import torch.nn as nn


def define_optim(model, config):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=config["scheduler_step"], gamma=config["scheduler_gamma"])
    return criterion, optimizer, exp_lr_scheduler
