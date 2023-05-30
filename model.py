import torch
import torch.nn as nn
from torchvision.datasets.inaturalist import Image
from torchvision.models import ResNet50_Weights, resnet50
from torchmetrics.functional.classification import multiclass_accuracy

class MyModel:
    def __init__(self, config, writer):
        self.writer = writer
        self.train_accuracy_history = []
        self.test_accuracy_history = []
        self.device = config["device"]
        self.train_loss_history = []
        self.test_loss_history = []
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        for i, param in enumerate(self.model.parameters()):
            param.requires_grad = False
            if i == 80:
                break
        self.model.fc = nn.Sequential(
                        nn.Dropout(p=config["dropout_p"]),
                        nn.Linear(2048, config["num_classes"]))
        self.model = self.model.to(self.device)

    def train_step(self, epoch, loader, optimizer, criterion, scheduler):
        running_loss = 0
        running_correct = 0
        total = 0
        self.model.train()
        for i, (inputs, labels) in enumerate(loader['train']):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)  # Get the index of the maximum value along the second dimension (class probabilities)
            total += labels.size(0)  # Total number of samples
            running_correct += (predicted == labels).sum().item()  # Number of correctly predicted samples
        scheduler.step()
        self.writer.add_scalar("training_loss", running_loss / len(loader["train"]), epoch)
        self.writer.add_scalar("train_accuracy", running_correct / total, epoch)

    def test_step(self, epoch, loader, criterion, metrics):
        running_loss = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(loader['test']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                total += labels.size(0)  # Total number of samples
                _, preds = torch.max(outputs, 1)
                metrics.update(preds, labels)
        self.model.train()
        self.writer.add_scalar("testing_loss", running_loss / len(loader["test"]), epoch)
        self.writer.add_scalar("test_accuracy", metrics.accuracy().mean(), epoch)

    def save(self, path):
        torch.save(self.model, path)

    def save_onnx(self, path):
        dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
        torch.onnx.export(self.model, dummy_input, path)

'''class MyModel(L.LightningModule):
    def __init__(self, num_classes):
        self.outputs = None
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        for i, param in enumerate(self.model.parameters()):
            param.requires_grad = False
            if i == 80:
                break
        self.model.fc = nn.Sequential(
                        nn.Dropout(p=0.3),
                        nn.Linear(2048, num_classes))

    def forward(self, x):
        embedding = self.model(x.float())
        return embedding 

    def shared_step(self, batch, stage):
        inputs, labels = batch
        logits = self(inputs)
        loss = self.loss_fn(logits, labels)
        _, preds = torch.max(logits, dim=1)
        self.outputs = {
            "preds": preds,
            "labels": labels
        }
        return {
            "loss": loss
        }

    def shared_epoch_end(self, stage):
        if stage == "train":
            accuracy = multiclass_accuracy(self.outputs['preds'], self.outputs['labels'], num_classes=self.num_classes)

    def training_step(self):
        
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def on_train_epoch_end(self):
        return self.shared_epoch_end("train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def on_validation_epoch_end(self):
        return self.shared_epoch_end("valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def on_test_epoch_end(self):
        return self.shared_epoch_end("test")

    def configure_optimizers(self):
        optimizer = Adam(
                    self.parameters(), 
                    lr=config["learning_rate"], 
                    weight_decay=config["weight_decay"])
        exp_lr_scheduler = lr_scheduler.StepLR(
                    optimizer, 
                    step_size=config["scheduler_step"], 
                    gamma=config["scheduler_gamma"])
        scheduler_config = {
            "scheduler": exp_lr_scheduler,
            "interval": config["scheduler_interval"]
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config
        }'''


