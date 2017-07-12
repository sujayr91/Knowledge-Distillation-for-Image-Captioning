import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Loads the pretrained ResNet-50 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights."""
        self.resnet.fc.weight.data.normal_(0.0, 0.02)
        self.resnet.fc.bias.data.fill_(0)
        
    def forward(self, images):
        """Extracts the image feature vectors."""
        features = self.resnet(images)
        features = self.bn(features)
        return features
    
# This is Teacher LSTM.    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
#        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
	self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        
    def forward(self, features, captions, lengths):
        """Decodes image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    
    def sample(self, features, states):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(20):
            hiddens, states = self.lstm(inputs, states)                # (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))                  # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
        sampled_ids = torch.cat(sampled_ids, 1)                        # (batch_size, 20)
        return sampled_ids.squeeze()



class StudentCNN_Model1(nn.Module):
    def __init__(self, embed_size):
        super(StudentCNN_Model1, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights."""
        self.resnet.fc.weight.data.normal_(0.0, 0.02)
        self.resnet.fc.bias.data.fill_(0)
        
    def forward(self, images):
        """Extracts the image feature vectors."""
        features = self.resnet(images)
        features = self.bn(features)
        return features


class StudentCNN_Model2(nn.Module):
	def __init__(self,embed_size):
		super(StudentCNN_Model2, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size=5, padding=2),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(2))
		self.layer2 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=5, padding=2),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(2))
		self.fc = nn.Linear(56*56*32, 1000)
	def init_weights(self):
		self.layer1.weight.data.normal_(0.0,0.02)
		self.layer2.weight.data.normal_(0.0,0.02)
		self.fc.weight.data.normal_(0.0,0.02)	
	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.view(out.size(0), -1)
		out = self.fc(out)
		return out
