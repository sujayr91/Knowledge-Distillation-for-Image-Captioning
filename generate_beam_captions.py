from data3 import get_cnn_data_loader 
from vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from configuration import Config
from PIL import Image
from torch.autograd import Variable 
from beam_search import CaptionGenerator
import torch
import torchvision.transforms as T 
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
import json

def main():
    # Configuration for hyper-parameters
    config = Config()
    beam_search_size = 5
    # Image Preprocessing
    transform = config.test_transform
    # Load vocabulary
    with open(os.path.join(config.vocab_path, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    # Build Models
    encoder = EncoderCNN(config.embed_size)
    encoder.eval()  # evaluation mode (BN uses moving mean/variance)
    decoder = DecoderRNN(config.embed_size, config.hidden_size, 
                         len(vocab), config.num_layers)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(os.path.join('../TrainedModels/TeacherCNN', 
                                                    config.trained_encoder)))
    decoder.load_state_dict(torch.load(os.path.join('../TrainedModels/TeacherLSTM', 
                                                    config.trained_decoder)))
    # Build data loader
    image_path = os.path.join(config.image_path, 'train2014')
    json_path = os.path.join(config.caption_path, 'captions_train2014.json')
    train_loader = get_cnn_data_loader(image_path, json_path, vocab, 
                                   transform, 1,
                                   shuffle=False, num_workers=config.num_threads) 
    data_store = {}
    my_list = []
    loop_count = 0
    for i, (image_tensor, img_id) in enumerate(train_loader):
		if img_id[0] in data_store:
			continue
		image_tensor = Variable(image_tensor)
		state = (Variable(torch.zeros(config.num_layers, 1, config.hidden_size)),
             		Variable(torch.zeros(config.num_layers, 1, config.hidden_size)))
		# If use gpu
		if torch.cuda.is_available():
			encoder.cuda()
			decoder.cuda()
			state = [s.cuda() for s in state]
			image_tensor = image_tensor.cuda()
		cap_gen = CaptionGenerator(embedder= decoder.embed,
                                   rnn= decoder.lstm,
                                   classifier= decoder.linear,
                                   eos_id=2,
                                   beam_size= beam_search_size,
                                   max_caption_length=20,
                                   length_normalization_factor=0)		
		# Generate caption from image
		feature = encoder(image_tensor)
	    	sentences, score = cap_gen.beam_search(feature)
		data_store[img_id[0]] = sentences
		sampled_caption = []
		for word_id in sentences[-1]:
			word = vocab.idx2word[word_id]
			if word_id==96 :
				sampled_caption.append(word)
				break
			if word == '<end>':
				break
			if word == '<start>':
				continue
			sampled_caption.append(word)
		
		sampled_caption.append(".")
		caption = ' '.join(sampled_caption)		
		my_list.append({"image_id" : img_id[0], "caption" : caption})
		print(loop_count)		
		loop_count = loop_count + 1
		break

    with open('beam_captions.json','w') as fp:
	json.dump(my_list, fp)
    with open('../Beamdatabase/beam_database.txt' , 'w') as myfile:
	pickle.dump(data_store,myfile)

if __name__ == '__main__':
    main()
