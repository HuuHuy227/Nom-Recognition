import torch
import torch.nn as nn
from torchvision.models import resnet18

class CRNN(nn.Module):
    
    def __init__(self, vocab_size, rnn_hidden_size=256, dropout=0.1):
        super(CRNN, self).__init__()
        self.vocab_size = vocab_size
        self.rnn_hidden_size = rnn_hidden_size
        self.dropout = dropout
        
        # CNN Part 1
        resnet_modules = list(resnet18(pretrained=True).children())[:-3]
        self.cnn_p1 = nn.Sequential(*resnet_modules)
        
        # CNN Part 2
        self.cnn_p2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(2, 2), stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.linear1 = nn.Linear(512, 512)
        
        # RNN
        self.rnn1 = nn.GRU(input_size=2*rnn_hidden_size, 
                            hidden_size=rnn_hidden_size,
                            bidirectional=True, 
                            batch_first=True)
        self.rnn2 = nn.GRU(input_size=rnn_hidden_size, 
                            hidden_size=rnn_hidden_size,
                            bidirectional=True, 
                            batch_first=True)
        self.linear2 = nn.Linear(self.rnn_hidden_size*2, self.vocab_size)
        
        
    def forward(self, conv):
        """
        ------:size sequence:------
        torch.Size([batch_size, 3, 432, 48]) -- IN:
        torch.Size([batch_size, 256, 27, 3]) -- CNN blocks 1
        torch.Size([batch_size, 256, 26, 2]) -- CNN blocks 2
        torch.Size([batch_size, 26, 256, 2]) -- permuted 
        torch.Size([batch_size, 26, 512]) -- Linear #1
        torch.Size([batch_size, 26, 256]) -- IN GRU 
        torch.Size([batch_size, 26, 256]) -- OUT GRU 
        torch.Size([batch_size, 52, 512]) -- skip_connection
        torch.Size([batch_size, 52, vocab_size]) -- Linear #2
        torch.Size([52, batch_size, vocab_size]) -- :OUT
        """
        conv = self.cnn_p1(conv)

        conv = self.cnn_p2(conv)
        
        conv = conv.permute(0, 2, 1, 3) 
         
        batch_size = conv.size(0)
        T = conv.size(1)
        conv = conv.reshape(batch_size, T, -1) 
        
        feature_map = self.linear1(conv)
        
        batch, _ = self.rnn1(feature_map)
        feature_size = batch.size(2)
        batch = batch[:, :, :feature_size//2] + batch[:, :, feature_size//2:]
        
        batch, _ = self.rnn2(batch)
        x = torch.cat([feature_map, batch], 1)
        x = self.linear2(x)
        
        x = x.permute(1, 0, 2) 
        
        return x