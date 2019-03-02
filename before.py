if __name__ == '__main__':

    import torch

    from CIFAR10Net import Network
    from config import net_configs, batch_size, classes, num_startingpoints

    print(f'generating {num_startingpoints} starting points')
    for i in range(num_startingpoints):
        current_net = Network(conv_channels=3, linear_channels=64)
        torch.save(current_net.state_dict(), f'./cifar10startingpoints/start_{i}')
    
    print('done')