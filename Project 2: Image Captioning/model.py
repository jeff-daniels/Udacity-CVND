import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # embedding layer that converts captions into a vector of a specified size
        self.caption_embeddings_layer = nn.Embedding(vocab_size, embed_size)

        # the LSTM takes embedded word vectors (of a specified size) as inputs 
        # and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)

        # the linear layer that maps the hidden state output dimension 
        # to the number of words we want as output, vocab_size
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        ''' Define the feedforward behavior of the model.'''
        
        # pass captions through an embedding layer
        # length in words of each caption == captions.shape[1]
        # embeds.shape = (batch_size, captions.shape[1], embed_size)
        captions = captions[:, :-1]
        caption_embeddings = self.caption_embeddings_layer(captions)
        
        # Concatenate features with the caption embeddings
        # features.shape == (batch_size, embed_size)
        # caption_embeddings.shape == (batch_size, captions.shape[1], embed_size) 
        # inputs.shape == (batch_size, captions.shape[1], embed_size)
        inputs = torch.cat((features.unsqueeze(1), caption_embeddings), dim=1)

        # get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hidden state
        lstm_out, _ = self.lstm(inputs)
        outputs = self.linear(lstm_out)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sentance = []
        for w in range(max_len):
            sentance.append(self.forward(inputs))
        return sentance