from train import main, parse_argument

args = {}

args['data_path'] = '/data/sdv2/taobao/data/embedding/'
args['feat_dim'] = 1024
args['embed_dim'] = 256
args['classes'] = 5345
args['batch_num'] = 10000
args['batch_size'] = 100
args['batch_k'] = 10
args['gpus'] = '0,1,2,3'
args['model'] = 'resnet50'
args['save_prefix'] = 'triplet'
args['use_pretrained'] = False
args['start_epoch'] = 0
args['workers'] = 4
args['epochs'] = 12
args['lr'] = 0.005
args['lr_beta'] = 0.05
args['margin'] = 0.2
args['momentum'] = 0.9
args['beta'] = 1.2
args['nu'] = 0.0
args['factor'] = 0.1
args['steps'] = '6,10'
args['resume'] = None
args['wd'] = 0.0001
args['seed'] = None
args['normalize_weights'] = True
args['print_freq'] = 100
args['loss'] = 'triplet'

main(args)