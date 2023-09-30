import argparse
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from .zo_signSGD import ZOSignSGD

def test_robust(model, test_loader, device, attacker=None):
    total_num = 0
    true_num = 0
    pbar = tqdm(test_loader, total=len(test_loader), desc=f"Test {'Clean' if attacker is None else 'Adversarial'} Set", ncols=100)
    
    for x, y in pbar:
        total_num += y.size(0)
        x, y = x.to(device), y.to(device)
        
        if attacker is not None:
            x = attacker.batch_attack(x, y)
        
        fx = model(x)
        pred = torch.argmax(fx.squeeze(), dim=-1)
        true_num += pred.eq(y).float().sum().item()
        
        acc = true_num / total_num
        pbar.set_postfix_str(f"Acc {100 * acc:.2f}%")
    
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--data_path', default='./data/CIFAR10')
    parser.add_argument('--model_path', default='./results/best_model.pth')
    # ZoSignSGD specific arguments
    parser.add_argument('--delta', default=0.01, type=float)
    parser.add_argument('--T', default=100, type=int)
    parser.add_argument('--mu', default=0.1, type=float)
    parser.add_argument('--q', default=5, type=int)
    parser.add_argument('--const', default=1, type=float)
    parser.add_argument('--k', default=0, type=int)
    parser.add_argument('--variant', default='central', type=str)
    args = parser.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    test_transform = transforms.Compose([transforms.ToTensor()])
    test_data = CIFAR10(root=args.data_path, train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_data, args.batch_size, shuffle=False, num_workers=4)

    # Load the model
    model = torch.load(args.model_path)
    model = model.to(device)
    model.eval()

    # Initialize ZoSignSGD attacker
    attacker = ZOSignSGD(model, 
                         delta=args.delta, 
                         T=args.T, 
                         mu=args.mu, 
                         q=args.q, 
                         const=args.const, 
                         k=args.k, 
                         variant=args.variant)

    # Test clean accuracy
    clean_acc = test_robust(model, test_loader, device)
    print(f"Clean Accuracy: {100 * clean_acc:.2f}%")

    # Test adversarial accuracy
    adv_acc = test_robust(model, test_loader, device, attacker=attacker)
    print(f"Adversarial Accuracy: {100 * adv_acc:.2f}%")