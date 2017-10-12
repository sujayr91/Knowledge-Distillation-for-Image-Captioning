from data1 import get_data_loader 
from vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from configuration import Config
from PIL import Image
from torch.autograd import Variable 
from beam_search import CaptionGenerator
import torch
import get_cider
import torchvision.transforms as T 
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
import json
import get_cider
from pdb import set_trace as st
from vocab import Vocabulary 
mydatabaselist=[]
with open(os.path.join('../COCO_Dataset', 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)
def sortnreturnindex(ciderlist):
	indexlist=[i for i in range(0,len(ciderlist))]
	for index1 in range (0, len(ciderlist)):
		smallest=index1
		for index2 in range(index1, len(ciderlist)):
		    if(ciderlist[index2] < ciderlist[smallest]):
		        smallest=index2
		temp=ciderlist[index1]
		temp_index=indexlist[index1]
		ciderlist[index1]=ciderlist[smallest]
		indexlist[index1]=indexlist[smallest]
		ciderlist[smallest]=temp
		indexlist[smallest]=temp_index
	return indexlist	

def get_topcider(imageid,sentences):
	untokenizedsentences=[]
	for sentence in sentences:
		untokenizedsentence=[]
		for word_id in sentence:
			if(word_id==-1):
				continue
			if(word_id==2):
				break
			word=vocab.idx2word[word_id]
			untokenizedsentence.append(word)
		untokenizedsentence=" ".join(untokenizedsentence)
		untokenizedsentences.append(untokenizedsentence)
			
	cider=[]
	for sentence in untokenizedsentences:
		mylist=[]
		mydic={}
		mydic["image_id"]=imageid
		mydic["caption"]=sentence
		mylist.append(mydic)
		filename='cider.json'
   		with open(filename , 'w') as myfile:
			json.dump(mylist,myfile)
		cider.append(get_cider.get_cider())
	print(cider)
	indexlist=sortnreturnindex(cider)
	print(indexlist)
	for index in indexlist:
		mydic={}
		mydic["image_id"]=imageid
		mydic["caption"]=sentences[index]
		mydatabaselist.append(mydic)
		
		
			

		

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
    image_path = os.path.join(config.image_path, 'train2017')
    json_path = os.path.join(config.caption_path, 'captions_train2017.json')
    train_loader = get_data_loader(image_path, json_path, vocab, 
                                 transform, 1,
                                   shuffle=False, num_workers=config.num_threads) 
    my_list = []
    img_ids=[]
    loop_count = 0
    for i, (image_tensor,captions, lengths, img_id) in enumerate(train_loader):
		if img_id in img_ids:
			continue
		loop_count+=1
		if(loop_count==5):
			break
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
		get_topcider(img_id[0], sentences)	
		if ((i+1)%100==0):
			print('Completed generation captions for ' + str(loop_count)+ ' Images')
    filename='./beam'+ str(beam_search_size)+'database.txt'
    with open(filename , 'w') as myfile:
	pickle.dump(mydatabaselist,myfile)

if __name__ == '__main__':
	main()
