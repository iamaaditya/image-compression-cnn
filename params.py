from pprint import pprint

class HyperParams() :
    def __init__(self, verbose):
        # Hard params and magic numbers
        self.sparse      = False
        self.vgg_weights = './data/caffe_layers_value.pickle'
        self.model_path  = 'models/model-20'
        self.n_labels    = 257
        self.top_k       = 5  
        self.stddev      = 0.2
        self.fine_tuning = False
        self.image_h     = 224
        self.image_w     = 224
        self.image_c     = 3 
        self.cnn_struct  = 'msroi' # ['vgg', 'msroi']
        self.filter_h    = 3
        self.filter_w    = 3

        if verbose:
            pprint(self.__dict__)
        
class CNNParams():
    def __init__(self, verbose):
        self.layer_shapes = self.get_layer_shapes()
        self.pool_window   = [1, 2, 2, 1]
        self.pool_stride   = [1, 2, 2, 1]
        self.last_features = self.layer_shapes['conv6/W'][-1]

        if verbose:
            pprint(self.__dict__)

    def get_layer_shapes(self):
        shapes = {}
        hyper = HyperParams(verbose=False)
        if hyper.cnn_struct == 'vgg':
            f = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 1024]
        else:
            f = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 16, 1024]
            conv_3d_size = 1024

        shapes['conv1_1/W'] = (hyper.filter_h, hyper.filter_w, hyper.image_c, f[0])
        shapes['conv1_1/b'] = (f[0],)
        shapes['conv1_2/W'] = (hyper.filter_h, hyper.filter_w, f[0], f[1])
        shapes['conv1_2/b'] = (f[1],)
        shapes['conv2_1/W'] = (hyper.filter_h, hyper.filter_w, f[1], f[2])
        shapes['conv2_1/b'] = (f[2],)
        shapes['conv2_2/W'] = (hyper.filter_h, hyper.filter_w, f[2], f[3])
        shapes['conv2_2/b'] = (f[3],)
        shapes['conv3_1/W'] = (hyper.filter_h, hyper.filter_w, f[3], f[4])
        shapes['conv3_1/b'] = (f[4],)
        shapes['conv3_2/W'] = (hyper.filter_h, hyper.filter_w, f[4], f[5])
        shapes['conv3_2/b'] = (f[5],)
        shapes['conv3_3/W'] = (hyper.filter_h, hyper.filter_w, f[5], f[6])
        shapes['conv3_3/b'] = (f[6],)
        shapes['conv4_1/W'] = (hyper.filter_h, hyper.filter_w, f[6], f[7])
        shapes['conv4_1/b'] = (f[7],)
        shapes['conv4_2/W'] = (hyper.filter_h, hyper.filter_w, f[7], f[8])
        shapes['conv4_2/b'] = (f[8],)
        shapes['conv4_3/W'] = (hyper.filter_h, hyper.filter_w, f[8], f[9])
        shapes['conv4_3/b'] = (f[9],)
        shapes['conv5_1/W'] = (hyper.filter_h, hyper.filter_w, f[9], f[10])
        shapes['conv5_1/b'] = (f[10],)
        shapes['conv5_2/W'] = (hyper.filter_h, hyper.filter_w, f[10], f[11])
        shapes['conv5_2/b'] = (f[11],)
        shapes['conv5_3/W'] = (hyper.filter_h, hyper.filter_w, f[11], f[12])
        shapes['conv5_3/b'] = (f[12],)
        shapes['conv6/W']   = (hyper.filter_h, hyper.filter_w, conv_3d_size, f[13])
        shapes['conv6/b']   = (f[13],)
        shapes['GAP/W']     = (f[13], hyper.n_labels)
        return shapes


class TrainingParams():
    def __init__(self, verbose):
        self.model_path         = './models/'
        self.num_epochs          = 20
        self.learning_rate      = 0.002
        self.weight_decay_rate  = 0.0005
        self.momentum           = 0.9
        self.batch_size         = 128
        self.max_iters          = 200000
        self.test_every_iter    = 200
        self.data_train_path    = './data/train.pickle'
        self.data_test_path     = './data/test.pickle'
        self.resume_training    = True

        if verbose:
            pprint(self.__dict__)
