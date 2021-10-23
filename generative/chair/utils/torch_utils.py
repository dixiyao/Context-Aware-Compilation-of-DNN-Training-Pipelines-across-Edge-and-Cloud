import torch

def save(filename, epoch, model, optimizer, scheduler = None):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
        # ...
    }
    torch.save(state, filename)

def load(filename, model, optimizer, epoch, scheduler = None):
    
    state = torch.load(filename)

    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    epoch = state['epoch'] + 1
    scheduler.load_state_dict(state['scheduler'])

    return epoch, model, optimizer, scheduler

#with no code...
def save_model(filename, model):
    torch.save(model, filename)

#with no code...
def load_model(filename):
    model = torch.load(filename)
    return model