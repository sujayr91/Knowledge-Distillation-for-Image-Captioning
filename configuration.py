import torchvision.transforms as T 


class Config(object):
    """Wrapper class for hyper-parameters."""
    def __init__(self):
        """Set the default hyper-parameters."""
        # Preprocessing
        self.image_size = 256
        self.crop_size = 224
        self.word_count_threshold = 4
        self.num_threads = 1
        
        # Image preprocessing in training phase
        self.train_transform = T.Compose([
            T.Scale(self.image_size),    
            T.RandomCrop(self.crop_size),
            T.RandomHorizontalFlip(), 
            T.ToTensor(), 
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        # Image preprocessing in test phase
        self.test_transform = T.Compose([
            T.Scale(self.crop_size),
            T.CenterCrop(self.crop_size),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        # Training 
        self.num_epochs = 10
        self.batch_size = 64
        self.learning_rate = 0.0001
	self.cnn_learningrate=1e-5
        self.log_step = 10
        self.save_step = 1000
        
        # Model
        self.embed_size = 512
        self.hidden_size = 512
        self.num_layers = 2
        
        # Path 
        self.image_path = '../coco/'
        self.caption_path = '../coco/annotations/'
        self.vocab_path = '../coco/'
	self.teacher_cnn_path='../TrainedModels/TeacherCNN'
	self.teacher_lstm_path='../TrainedModels/TeacherLSTM'
	self.student_cnn_path='../TrainedModels/StudentCNN'
	self.student_lstm_path= '../TrainedModels/StudentLSTM'
	self.baseline_cnn_path= '../TrainedModels/BaselineCNN'
	self.baseline_lstm_path= '../TrainedModels/BaselineLSTM'
	self.trained_encoder= 'encoder-38-7000_finetune.pkl'
	self.trained_decoder= 'decoder-38-7000_finetune.pkl'

