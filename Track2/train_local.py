from tqdm import tqdm

import torch

from models.resnext import ResNeXt


action_labels = ['Red', 'Amber', 'Green', 'MovAway', 'MovTow', 'Mov', 'Rev', 'Brake', 'Stop', 'IncatLft', 'IncatRht', 'HazLit', 'TurLft', 'TurRht', 'MovRht', 'MovLft', 'Ovtak', 'Wait2X', 'XingFmLft', 'XingFmRht', 'Xing', 'PushObj']


def train(args, model, train_loader, optimizer, criterion, epoch, writer):
    model.train()
    
    train_loss, train_acc = 0, 0
    tqdm_iter = tqdm(train_loader, desc="Epoch: {}/{} ({}%) |Training loss: NaN".format(
        epoch, args.epoch, int(epoch/args.epoch)), leave=False)
    for batch_idx, (data, label) in enumerate(tqdm_iter):
        data, target = data.cuda(), label[0]['loc_label'].cuda()
        loc = torch.stack(label[0]['bbox_pos']).T
        loc = loc.to(torch.float32).cuda()

        output = model(data, loc)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (output.argmax(dim=1) == target.argmax(dim=1)).float().sum().item()

        train_loss += loss.cpu().item()
        train_acc += acc

        tqdm_iter.set_description("Epoch: {}/{} ({}%) |Training loss: {:.6f} |Training Acc: {:.6f}".format(
            epoch, args.epoch, int(epoch/args.epoch), round(loss.item(), 6), round(acc / args.batch_size, 6)))
        
        if epoch == 1:
            writer.add_scalar("First Epoch Training Loss History", loss.item(), batch_idx)
            writer.add_scalar("First Epoch Training Accuarcy History", acc/args.batch_size, batch_idx)
    
    return train_loss / len(train_loader), train_acc / len(train_loader.dataset)
        

def test(args, model, test_loader, criterion, epoch):
    model.eval()
    
    test_loss, test_acc = 0, 0
    pred_set = []
    label_set = []
    with torch.no_grad():
        tqdm_iter = tqdm(test_loader, desc="Epoch: {}/{} ({}%) |Testing loss: NaN".format(epoch, args.epoch, int(epoch/args.epoch)), leave=False)
        for batch_idx, (data, label) in enumerate(tqdm_iter):
            data, target = data.cuda(), label[0]['loc_label'].cuda()
            loc = torch.stack(label[0]['bbox_pos']).T
            loc = loc.to(torch.float32).cuda()

            output = model(data, loc)
            loss = criterion(output, target)
            acc = (output.argmax(dim=1) == target.argmax(dim=1)).float().sum().item()
            
            test_loss += loss.cpu().item()
            test_acc += acc

            pred_set.append(output.cpu())
            label_set.append(target.cpu())

            tqdm_iter.set_description("Epoch: {}/{} ({}%) |Testing loss: {:.6f} |Testing Acc: {:.6f}".format(
                epoch, args.epoch, int(epoch/args.epoch), round(loss.item(), 6), round(acc / args.batch_size, 6)))
            
    return test_loss / len(test_loader), test_acc / len(test_loader.dataset), pred_set, label_set


if __name__ == "__main__":
    model = ResNeXt(num_blocks=[3, 3, 3], cardinality=4, bottleneck_width=4)
    x = torch.rand(1, 3, 32, 32)
    y = torch.rand(1, 12)