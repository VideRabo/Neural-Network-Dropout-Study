import torch
import torch.nn as nn
import torch.nn.functional as func
from functools import reduce    

class Network(nn.Module):
        def __init__(self, height=32, width=32, conv_channels=3, linear_channels=64, 
                    conv_dropout=(0.1, 0.1), linear_dropout=(0.4,)):
            super(Network, self).__init__()
            
            # field initialization
            self.linear_channels, self.conv_channels = linear_channels, conv_channels
            self.relu_slope = 0.0
            
            # layers    
            self.dropouts = nn.ModuleDict({
                'convs': nn.ModuleList([
                    nn.Dropout(conv_dropout[0]),
                    nn.Dropout(conv_dropout[1])
                ]),
                'linears': nn.ModuleList([
                    nn.Dropout(linear_dropout[0]),
                    nn.Dropout(0.0)
                ])
            })
            
            self.convs = nn.ModuleList([
                nn.Conv2d(in_channels=conv_channels, out_channels=32, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True),
            ])
            
            self.linears = nn.ModuleList([
                nn.Linear(in_features=self.calc_flattened_size(conv_channels, height, width), out_features=linear_channels, bias=True),
                nn.Linear(in_features=linear_channels, out_features=10, bias=True),
            ])
        
        def calc_flattened_size(self, conv_channels, height, width):
            '''calculate number of items in each image'''
            with torch.no_grad():
                x = self.run_convs(torch.ones((1, conv_channels, height, width)))
                return self.flatten(x).size()[1]
        
        @staticmethod
        def flatten(x):
            '''flatten tensor of size [a, b, c, ...] to [a, b*c*d...]'''
            return x.view(-1, reduce(lambda a, b: a*b, list(x.size()[1:])))

        
        def run_convs(self, x):
            '''transform x with convolutional module list'''
            for i in range(len(self.convs)):
                x = self.convs[i](x)
                # ReLU activation
                x = func.leaky_relu(x, negative_slope=self.relu_slope)
                # normalize
                x = func.normalize(x, p=2, dim=1) #batch_norm(input, running_mean, running_var, weight=None, bias=None, training=self.training, momentum=0.1, eps=1e-05) 
                # max pool
                x = func.max_pool2d(x, kernel_size=(2,2))
                # dropout
                x = self.dropouts['convs'][i](x)
            
            return x
        
        
        def run_linears(self, x):
            '''transform x with linear module list'''
            for i in range(len(self.linears) - 1):
                x = self.linears[i](x)
                # ReLU activation
                x = func.leaky_relu(x, negative_slope=self.relu_slope)
                # normalize
                x = func.normalize(x, p=2, dim=1) #batch_norm(input, running_mean, running_var, weight=None, bias=None, training=self.training, momentum=0.1, eps=1e-05) 
                # dropout
                x = self.dropouts['linears'][i](x)
                
            x = self.linears[i+1](x)   
            return x
        
        
        def forward(self, x):        
            # convolutional layers
            x = self.run_convs(x)  
            # flatten to 2 dimensions
            x = self.flatten(x)
            #linear layers
            x = self.run_linears(x)
            
            #x = torch.nn.functional.softmax(x, dim=1)
            return x
