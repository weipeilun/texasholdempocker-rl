from env.workflow import *
from tools.param_parser import *
import tensorrt as trt


EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


if __name__ == '__main__':
    log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(log_level)

    model_name = 'models/default_train'
    model_path = 'models/default_train.pth'

    args, params = parse_params()
    model_param_dict = params['model_param_dict']
    model = AlphaGoZero(**model_param_dict)

    save_model_by_state_dict(model.state_dict(), model.optimizer.state_dict(), model_path, model, ModelType.TENSORRT, params)

    logging.info('success')
