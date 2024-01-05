from env.workflow import *
from tools.param_parser import *


if __name__ == '__main__':
    log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(log_level)

    args, params = parse_params()
    model_param_dict = params['model_param_dict']
    model = AlphaGoZero(**model_param_dict)
    # to regenerate new default model
    save_model(model, 'models/default_train.pth')

    logging.info('success')
