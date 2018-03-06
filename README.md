# Knowledge Distillation for ImageCaptioning
*  Implementation of Knowledge distillation for Image Captioning. This project
combines the knowledge distillation techniques introduced in https://arxiv.org/abs/1503.02531, 
https://arxiv.org/abs/1606.07947 for a multimodal task, i.e Image Captioning

* Requirements: 
	*	Pytorch
	* 	Coco dataset in the folder coco
	
* Training

	* Image Captioning model is trained using teacher_train.py [Only embedding layer of resnet is learnt, all other layers are pretrained model taken from torch]
	* Create a database with beam search captions from Teacher. Vary beam size per requirment, has utility to get captions with top CIDER scores.[generate_beamsearch_captions.py]
	* Do joint distillation of CNN + LSTM: Loss = CrossEntropy(studentcaptions, teachercaptions) + MSELoss(studentcnnout, teachercnnout)

* Networks:
	*	Teacher CNN: Resnet 50
	* 	Embedding : 512
	*	Teacher LSTM: 2 layer, 512 hidden
	* 	Student CNN: Resnet 18
	*	Student LSTM: 1 layer, 256 hidden

* Evaluation:

	* Use CIDER, Blue Evaluation available in folder Evaluation

* Results:

	* Teacher trained on Ground Truth: Cider 0.867
	* Student trained using distillation Cider: 0.82

