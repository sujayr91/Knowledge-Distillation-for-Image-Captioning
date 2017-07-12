import sys
from vocab import Vocabulary
sys.path.append('../')
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from model import EncoderCNN, DecoderRNN, StudentCNN_Model1
from torch.autograd import Variable
import pickle
import os
import numpy as np
from configuration import Config
import torchvision.models as models
from data2 import cocoimage_data_loader
def main():
	num_epochs = 5
	batch_size = 16
	learning_rate = 0.0001
	config = Config()
	transform=config.train_transform
	with open(os.path.join(config.vocab_path, 'vocab.pkl'), 'rb') as f:
        	vocab = pickle.load(f)
	train_image_path = os.path.join(config.image_path, 'train2014')
	train_json_path = os.path.join(config.caption_path, 'captions_train2014.json')
	val_image_path=os.path.join(config.image_path,'val2014')
	val_json_path=os.path.join(config.caption_path,'captions_val2014.json')
	

        train_loader = cocoimage_data_loader(train_image_path,
                                   transform, 2,
                                   shuffle=False, num_workers=config.num_threads)
	val_loader = cocoimage_data_loader(val_image_path, 
                                   transform, config.batch_size,
                                   shuffle=True, num_workers=config.num_threads)
     
	student_cnn=StudentCNN_Model1(config.embed_size)	
	student_cnn.cuda()
	teacher_cnn = EncoderCNN(config.embed_size)
	teacher_cnn.eval()  # evaluation mode (BN uses moving mean/variance)
	
	teacher_cnn.load_state_dict(torch.load(os.path.join('../TrainedModels/TeacherCNN',         
                                                    config.trained_encoder)))
	teacher_cnn.cuda()
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(list(student_cnn.parameters()), lr=learning_rate)
	# Build Loss and Optimizer.
	# Train the Model
	with open('train2_log.txt', 'w') as logfile:
		for epoch in range(num_epochs):
			for i, (images) in enumerate(train_loader):
				images = Variable(images).cuda()
				optimizer.zero_grad()
				outputs_student = student_cnn(images)
				outputs_teacher=  teacher_cnn(images)
				outputs_teacher=outputs_teacher.detach()
				loss = criterion(outputs_student,outputs_teacher)
				loss.backward()
				optimizer.step()
				
				if (i+1) % 10 == 0:
					print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
						   %(epoch+1, num_epochs, i+1, len(train_loader), loss.data[0]))
							
				if((i+1)%1000)==0:
					print('Check validation loss')
					for i,(images) in enumerate(val_loader):
						images=Variable(images).cuda()
						optimizer.zero_grad()
						outputs_student=student_cnn(images)
						outputs_teacher=teacher_cnn(images)
						outputs_teacher=outputs_teacher.detach()
						valloss=criterion(outputs_student,outputs_teacher)
						print ('Validation Loss: %.4f' %(valloss.data[0]))
						logfile.write((str(valloss.data[0]))+ ','+ str(loss.data[0]))
						break
		torch.save(student_cnn.state_dict(),os.path.join(config.student_cnn_path,'encoder-5-25000.pkl'))
				   
if __name__ == '__main__':
    main()
										




									  
