from configparser import ConfigParser

config = ConfigParser()
config['data'] = {
    'data_path': '../data/deap/',
    'window_size': 10,
    'overlap_size': 5,
    'sampling_rate': 128,
}

config['model'] = {
    'batch_size': 64,
    'learning_rate': 1e-3,
    'num_epochs': 100,
    'hidden_size': 50,
    'weight_decay':.0,
    'model_path': './save/models/',
    'model_name': 'MLP'
    'dropout_rate': 0.5
}

with open('./config.ini', 'w') as f:
      config.write(f)