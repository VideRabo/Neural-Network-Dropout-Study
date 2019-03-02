classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net_configs = [
        {
            'name': 'c000_c000_l000',
            'conv_dropout': (0.0, 0.0),
            'linear_dropout': (0.0,)
        },
        {
            'name': 'c050_c050_l050',
            'conv_dropout': (0.5, 0.5),
            'linear_dropout': (0.5,)
        },
        {
            'name': 'c000_c000_l050',
            'conv_dropout': (0.0, 0.0),
            'linear_dropout': (0.5,)
        },
        {
            'name': 'c050_c050_l000',
            'conv_dropout': (0.5, 0.5),
            'linear_dropout': (0.0,)
        },
        {
            'name': 'c020_c020_l040',
            'conv_dropout': (0.2, 0.2),
            'linear_dropout': (0.4,)
        }
    ]

batch_size = 4

training_epochs = 2

num_startingpoints = 3
