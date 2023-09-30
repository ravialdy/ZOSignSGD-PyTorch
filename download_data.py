from torchvision.datasets import CIFAR10
import os
import os.path as osp

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_and_save(dataset, save_dir, categories):
    if osp.exists(save_dir):
        print(f'Folder "{save_dir}" already exists')
        return

    print(f'Extracting images to "{save_dir}" ...')
    mkdir_if_missing(save_dir)

    for i in range(len(dataset)):
        img, label = dataset[i]
        class_dir = osp.join(save_dir, categories[label])
        mkdir_if_missing(class_dir)
        impath = osp.join(class_dir, str(i + 1).zfill(5) + ".jpg")
        img.save(impath)

def download_and_prepare(name, root):
    print(f"Dataset: {name}")
    print(f"Root: {root}")
    
    if name == "cifar10":
        train = CIFAR10(root, train=True, download=True)
        test = CIFAR10(root, train=False, download=True)
    else:
        raise ValueError("Unknown dataset name.")
    
    train_dir = osp.join(root, name, "train")
    test_dir = osp.join(root, name, "test")

    categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    extract_and_save(train, train_dir, categories)
    extract_and_save(test, test_dir, categories)

if __name__ == "__main__":
    download_and_prepare("cifar10", './data/CIFAR10')