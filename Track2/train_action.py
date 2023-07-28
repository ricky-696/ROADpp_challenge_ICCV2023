from tqdm import tqdm

import torch

from models.resnext import ResNeXt


action_labels = ['Red', 'Amber', 'Green', 'MovAway', 'MovTow', 'Mov', 'Rev', 'Brake', 'Stop', 'IncatLft', 'IncatRht', 'HazLit', 'TurLft', 'TurRht', 'MovRht', 'MovLft', 'Ovtak', 'Wait2X', 'XingFmLft', 'XingFmRht', 'Xing', 'PushObj']


def train(args, model, train_loader, optimizer, criterion, epoch):
    model.train()
    
    train_loss, train_acc = 0, 0
    tqdm_iter = tqdm(train_loader, desc="Epoch: {}/{} ({}%) |Training loss: NaN".format(
        epoch, args.epoch, int(epoch/args.epoch)), leave=False)
    for batch_idx, (data, label) in enumerate(tqdm_iter):
        data, target = data.cuda(), label['action_label'].cuda()

        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (output.argmax(dim=1) == target.argmax(dim=1)).float().sum().item()

        train_loss += loss.cpu().item()
        train_acc += acc

        tqdm_iter.set_description("Epoch: {}/{} ({}%) |Training loss: {:.6f} |Training Acc: {:.6f}".format(
            epoch, args.epoch, int(epoch/args.epoch), round(loss.item(), 6), round(acc / args.batch_size, 6)))
    
    return train_loss / len(train_loader), train_acc / len(train_loader.dataset)
        

def test(args, model, test_loader, criterion, epoch):
    model.eval()
    
    test_loss, test_acc = 0, 0
    uncorrect_count = [0 for _ in range(len(action_labels))]
    with torch.no_grad():
        tqdm_iter = tqdm(test_loader, desc="Epoch: {}/{} ({}%) |Testing loss: NaN".format(epoch, args.epoch, int(epoch/args.epoch)), leave=False)
        for batch_idx, (data, label) in enumerate(tqdm_iter):
            data, target = data.cuda(), label['action_label'].cuda()

            output = model(data)
            loss = criterion(output, target)
            acc = (output.argmax(dim=1) == target.argmax(dim=1)).float().sum().item()
            
            test_loss += loss.cpu().item()
            test_acc += acc

            for idx, target_ in enumerate(torch.argmax(target, dim=1)):
                # print(target.tolist())
                if torch.argmax(output[idx]).item() != target_.item():
                    uncorrect_count[target_.item()] += 1

            tqdm_iter.set_description("Epoch: {}/{} ({}%) |Testing loss: {:.6f} |Testing Acc: {:.6f}".format(
            epoch, args.epoch, int(epoch/args.epoch), round(loss.item(), 6), round(acc / args.batch_size, 6)))
            
    return round(test_loss / len(test_loader), 6), round(test_acc / len(test_loader), 6), uncorrect_count


if __name__ == "__main__":
    model = ResNeXt(num_blocks=[3, 3, 3], cardinality=4, bottleneck_width=4)
    x = torch.rand(1, 3, 32, 32)
    y = torch.rand(1, 12)