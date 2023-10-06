import torch

def train(model, optimiser, val_loader, train_loader, device, conditional=True):
    """
    The training function.
    :param model: the model of the neural network
    :param optimiser: the optimiser
    :param val_loader: validation data, a torch DataLoader object
    :param train_loader: training data, a torch DataLoader object]
    :param noise_modification: bool, if true, random noise is added to the conditional data
    :return train_loss, val_loss
    """
    device = torch.device('cuda')
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        x, y = batch
        optimiser.zero_grad()
        if conditional == True:
            _loss = -model.log_prob(x, conditional=y).mean()
        else:
            _loss = -model.log_prob(x).mean()
        _loss.backward()
        optimiser.step()
        train_loss += _loss.item()
    train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    for batch in val_loader:
        x, y = batch
        with torch.no_grad():
            if conditional == True:
                _loss = -model.log_prob(x, conditional=y).mean().item()
            else:
                _loss = -model.log_prob(x).mean().item()
        val_loss += _loss
    val_loss = val_loss / len(val_loader)
    return train_loss, val_loss

