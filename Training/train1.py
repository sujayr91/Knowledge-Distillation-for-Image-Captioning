from data1 import get_data_loader 
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

    torch.cuda.set_device(0)
    config = Config()
    # Image preprocessing
    transform = config.train_transform
    # Load vocabulary wrapper
    with open(os.path.join(config.vocab_path, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    # Build data loader
    train_image_path = os.path.join(config.image_path, 'train2017')
    json_path = os.path.join(config.caption_path, 'captions_train2017.json')
    train_loader = get_data_loader(train_image_path, json_path, vocab, 
                                   transform, config.batch_size,
                                   shuffle=False, num_workers=config.num_threads)
    
    val_image_path = os.path.join(config.image_path, 'val2017')
    json_path = os.path.join(config.caption_path, 'captions_val2017.json')
    val_loader = get_data_loader(val_image_path, json_path, vocab, 
                                   transform, config.batch_size,
                                   shuffle=False, num_workers=config.num_threads)
     
    total_step = len(train_loader)

    # Build Models
    encoder = EncoderCNN(config.embed_size)
    encoder.eval()
    decoder = DecoderRNN(config.embed_size, config.hidden_size, 
                         len(vocab), config.num_layers)


    #encoder.load_state_dict(torch.load(os.path.join('../../TrainedModels/TeacherCNN',
#							config.trained_encoder())))
    #encoder.load_state_dict(torch.load(os.path.join('../../TrainedModels/TeacherLSTM',
#							config.trained_decoder())))

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.resnet.fc.parameters())
    optimizer = torch.optim.Adam(params, lr=config.learning_rate)    
    
    print('entering in to training loop')    
    # Train the Models

    with open('train1_log.txt', 'w') as logfile:
	    logfile.write('Validation Error,Training Error')
	    for epoch in range(0,25):
		for i, (images, captions, lengths,img_ids) in enumerate(train_loader):
		    images = Variable(images)
		    captions = Variable(captions)
		    if torch.cuda.is_available():
			images = images.cuda()
			captions = captions.cuda()
		    targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
		    # Forward, Backward and Optimize
		    optimizer.zero_grad()
		    features = encoder(images)
		    outputs = decoder(features, captions, lengths)
		    loss = criterion(outputs, targets)
		    loss.backward()
		    optimizer.step()
		    # Print log info
		    if i % config.log_step == 0:
			print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
			      %(epoch, config.num_epochs, i, total_step, 
				loss.data[0], np.exp(loss.data[0]))) 
			
		    # Save the Model
		    if (i+1) % config.save_step == 0:
			torch.save(encoder.state_dict(), 
				   os.path.join(config.teacher_cnn_path, 
						'encoder-%d-%d.pkl' %(epoch+1, i+1)))
			torch.save(decoder.state_dict(), 
				   os.path.join(config.teacher_lstm_path, 
						'decoder-%d-%d.pkl' %(epoch+1, i+1)))

		print('Just Completed an Epoch, Initite Validation Error Test')
		avgvalloss=0
		for j,(images,captions, lengths, img_ids) in enumerate(val_loader):
			images=Variable(images)
			captions=Variable(captions)
			if torch.cuda.is_available():
				images=images.cuda()
				captions=captions.cuda()
			targets=pack_padded_sequence(captions, lengths, batch_first=True)[0]
			optimizer.zero_grad()
			features=encoder(images)
			outputs=decoder(features, captions,lengths)
			valloss=criterion(outputs,targets)
			if j==0:
				avgvalloss=valloss.data[0]
			avgvalloss=(avgvalloss+valloss.data[0])/2
			if ((j+1)%1000==0):
				print('Average Validation Loss: %.4f'
			      		%(avgvalloss))
				logfile.write(str(avgvalloss)+ ',' + str(loss.data[0])+ str('\n')) 
				break	
			
if __name__ == '__main__':
    main()
