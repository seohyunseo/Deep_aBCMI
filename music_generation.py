import os
import pickle
import torch
from transformer.utils import write_midi
from transformer.models import TransformerModel, network_paras

path_dictionary = './dataset/co-representation/dictionary.pkl'
assert os.path.exists(path_dictionary)

dictionary = pickle.load(open(path_dictionary, 'rb'))
event2word, word2event = dictionary

# config
n_class = []   # num of classes for each token
for key in event2word.keys():
    n_class.append(len(dictionary[0][key]))
n_token = len(n_class)

os.listdir('./transformer/exp/pretrained_transformer')

path_saved_ckpt = './transformer/exp/pretrained_transformer/loss_25_params.pt'
assert os.path.exists(path_saved_ckpt)

# init model
net = TransformerModel(n_class, is_training=False)
net.cuda()
net.eval()

net.load_state_dict(torch.load(path_saved_ckpt))

emotion = [1, 2, 3, 4] # number of emotion classes
max_bar = 4 # max number of bars of generated music piece

for i in range(len(emotion)):
    emotion_tag = emotion[i] # the target emotion class you want. It should belongs to [1,2,3,4].
    path_outfile = f'./midi/test{i+1}' # output midi file name

    res, _ = net.inference_from_scratch(dictionary, emotion_tag, n_token=8, display=False, max_bar=max_bar) # generate
    
    write_midi(res, path_outfile + '.mid', word2event, max_bar)
    print(f"\nMidi example {i+1} completed") 