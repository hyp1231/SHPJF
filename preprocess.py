from logging import getLogger

from config import Config
from data.dataset import create_datasets
from utils import init_seed, init_logger, dynamic_load


config = Config('SHPJF')
init_seed(config['seed'], config['reproducibility'])

# logger initialization
init_logger(config)
logger = getLogger()
logger.info(config)

# data preparation
pool = dynamic_load(config, 'data.pool', 'Pool')(config)
logger.info(pool)

config.params['model'] = 'RawSHPJF'
datasets = create_datasets(config, pool)
for ds in datasets:
    logger.info(ds)
