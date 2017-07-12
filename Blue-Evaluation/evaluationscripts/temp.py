import sys
from data import get_data_loader 
from vocab import Vocabulary
from model import EncoderCNN, DecoderRNN 
from configuration import Config
from PIL import Image
from torch.autograd import Variable 
from data import get_data_loader
from beam_search_temp import CaptionGenerator
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
    encoder = StudentCNN(config.embed_size)
    encoder.eval()  # evaluation mode (BN uses moving mean/variance)
    decoder = DecoderRNN(config.embed_size, config.hidden_size/2, 
                         len(vocab), config.num_layers/2)
    

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(os.path.join('./student_refine', 
                                                    config.trained_encoder)))
    decoder.load_state_dict(torch.load(os.path.join('./student_refine', 
                                                    config.trained_decoder)))
    # Build data loader
    image_path = os.path.join(config.image_path, 'val2014')
    json_path = os.path.join(config.caption_path, 'captions_val2014.json')
    train_loader = get_data_loader(image_path, json_path, vocab, 
                                   transform, 1,
                                   shuffle=False, num_workers=config.num_threads) 
    total_step = len(train_loader)
    data_store = {}
    my_list = []
    loop_count = 0
    for i, (image_tensor, captions, lengths, img_id) in enumerate(train_loader):
		if img_id[0] in data_store:
			continue
		#print('image_id is')
		#print(img_id)	
		image_tensor = Variable(image_tensor)
		  
		# Set initial states
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
		#print(sentences)
		#print("SCORE = " + str(score))
        	#sentences_english = [' '.join([vocab.idx2word[idx] for idx in sent]) for sent in sentences]
		#print(sentences_english)
		
		data_store[img_id[0]] = sentences[-1]
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





#    with open('beam_val_model3_test3_captions.txt', 'wb') as outfile:
#	pickle.dump(data_store, outfile)
    with open('beam_val_model3_refined.json','w') as fp:
	json.dump(my_list, fp)
	
	'''
    with open('beam_captions.txt', 'rb') as handle:
	new_data = pickle.loads(handle.read())
	print("read back data is ")
	print (new_data)
	'''
	
if __name__ == '__main__':
    main()
