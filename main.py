import argparse
from logging import getLogger

from config import Config
from data.dataset import create_datasets
from data.dataloader import construct_dataloader
from trainer import Trainer
from utils import init_seed, init_logger, dynamic_load


def main(model, config_dict=None, saved=True):
    """Main process API for experiments of VPJF

    Args:
        model (str): Model name.
        config_dict (dict): Parameters dictionary used to modify experiment parameters.
            Defaults to ``None``.
        saved (bool): Whether to save the model parameters. Defaults to ``True``.
    """

    # configurations initialization
    config = Config(model, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # data preparation
    pool = dynamic_load(config, 'data.pool', 'Pool')(config)
    logger.info(pool)

    datasets = create_datasets(config, pool)
    for ds in datasets:
        logger.info(ds)

    train_data, valid_data, test_data = construct_dataloader(config, datasets)

    # model loading and initialization
    model = dynamic_load(config, 'model')(config, pool).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, saved=saved)
    logger.info('best valid result: {}'.format(best_valid_result))

    # model evaluation
    test_result, test_result_str = trainer.evaluate(test_data, load_best_model=saved)
    logger.info('test result [all]: {}'.format(test_result_str))

    test_result_low, test_result_low_str = trainer.evaluate(test_data, load_best_model=saved, group='low')
    logger.info('test result [low]: {}'.format(test_result_low_str))

    test_result_high, test_result_high_str = trainer.evaluate(test_data, load_best_model=saved, group='high')
    logger.info('test result [high]: {}'.format(test_result_high_str))

    return {
        'best_valid_score': best_valid_score,
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SHPJF', help='Model to test.')
    args = parser.parse_args()

    main(model=args.model)
