from data2 import get_data_loader 
from vocab import Vocabulary
from configuration import Config
from model import EncoderCNN, DecoderRNN 
from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence
import torch
import torch.nn as nn
import numpy as np 
import pickle 
import os


def main():
    # Configuration for hyper-parameters
    config = Config()
    
    # Image preprocessing
    transform = config.train_transform
    
    # Load vocabulary wrapper
    with open(os.path.join(config.vocab_path, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

    # Build data loader
    image_path = os.path.join(config.image_path, 'train2014')
    json_path = os.path.join(config.caption_path, 'captions_train2014.json')
    train_loader = get_data_loader(image_path, json_path, vocab, 
                                   transform, config.batch_size,
                                   shuffle=True, num_workers=config.num_threads) 
    total_step = len(train_loader)

    # Build Models
    teachercnn = EncoderCNN(config.embed_size)
    teachercnn.eval()
    studentcnn = StudentCNN_Model1(config.embed_size)
    #Load the best teacher model
    teachercnn.load_state_dict(torch.load(os.path.join('../TrainedModels/TeacherCNN', config.trained_encoder))) 
    studentlstm = DecoderRNN(config.embed_size, config.hidden_size/2, 
                         len(vocab), config.num_layers/2)

    if torch.cuda.is_available():
        teachercnn.cuda()
	studentcnn.cuda()
        studentlstm.cuda()

    # Loss and Optimizer
    criterion_lstm = nn.CrossEntropyLoss()
    criterion_cnn = nn.MSELoss()
    params = list(studentlstm.parameters()) + list(studentcnn.parameters())
    optimizer_lstm = torch.optim.Adam(params, lr=config.learning_rate)    
    optimizer_cnn = torch.optim.Adam(studentcnn.parameters(), lr=config.cnn_learningrate)    
    
    print('entering in to training loop')    
    # Train the Models
	
    for epoch in range(config.num_epochs):
        for i, (images, captions, lengths, img_ids) in enumerate(train_loader):
	    images = Variable(images)
            captions = Variable(captions)
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            # Forward, Backward and Optimize
	    optimizer_lstm.zero_grad()
	    optimizer_cnn.zero_grad()
            features_tr = teachercnn(images)
	    features_st = studentcnn(images)
            outputs = studentlstm(features_st, captions, lengths)
            loss = criterion(features_st, features_tr.detach()) + criterion_lstm(outputs, targets)
            loss.backward()
            optimizer_cnn.step()
            optimizer_lstm.step()
     
           # Print log info
            if i % config.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      %(epoch, config.num_epochs, i, total_step, 
                        loss.data[0], np.exp(loss.data[0]))) 
                
            # Save the Model
            if (i+1) % config.save_step == 0:
                torch.save(studentlstm.state_dict(), 
                           os.path.join(config.student_lstm_path, 
                                        'decoder-%d-%d.pkl' %(epoch+1, i+1)))
		torch.save(studentcnn.state_dict(), 
                           os.path.join(config.student_cnn_path, 
                                        'encoder-%d-%d.pkl' %(epoch+1, i+1)))

if __name__ == '__main__':
    main()
