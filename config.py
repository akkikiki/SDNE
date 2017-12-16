class Config(object):
    def __init__(self):
        ## graph data
        #self.file_path = "../Data_SDNE/Flickr2_1.txt"
        #self.file_path = "GraphData/blogCatalog3.txt"
        self.file_path = "GraphData/A_10_nn_eng.txt"
        #self.label_file_path = "GraphData/blogCatalog3-groups.txt"
        ## embedding data
        #self.embedding_filename = "embeddingResult/blogCatolog" 
        self.embedding_filename = "embeddingResult/eng_wikipedia.txt" 
        ## hyperparameter
        #self.struct = [None, 1000, 128]
        #self.struct = [None, 500, 100]
        self.struct = [None, 500, 10]
        ## the loss func is  // gamma * L1 + alpha * L2 + reg * regularTerm // 
        self.alpha = 500
        self.gamma = 1
        self.reg = 1
        ## the weight balanced value to reconstruct non-zero element more.
        self.beta = 10
        
        ## para for training
        #self.rN = 0.9
        self.batch_size = 128
        self.epochs_limit = 5
        #self.epochs_limit = 1
        self.learning_rate = 0.01
        self.display = 1

        self.DBN_init = True
        #self.dbn_epochs = 500
        self.dbn_epochs = 20
        #self.dbn_epochs = 1
        self.dbn_batch_size = 128
        self.dbn_learning_rate = 0.1

        self.sparse_dot = False
        self.ng_sample_ratio = 0.0 # negative sample ratio
        
        #self.sample_ratio = 1
        #self.sample_method = "node"
