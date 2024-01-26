from env.workflow import *
from tools.param_parser import *
import tensorrt as trt


if __name__ == '__main__':
    log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(log_level)

    model_name = 'models/default_train'

    args, params = parse_params()
    model_param_dict = params['model_param_dict']
    model = AlphaGoZero(**model_param_dict)
    # to regenerate new default model
    save_model(model, f'{model_name}.pth')
    # to regenerate new default onnx model
    torch.onnx.export(model, torch.zeros((1, 28), dtype=torch.int32).to(model.device), f'{model_name}.onnx')

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt_runtime = trt.Runtime(TRT_LOGGER)

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(f"{model_name}.onnx", 'rb') as model_file:
            parser.parse(model_file.read())

        builder.build_cuda_engine(onnx_model.network)
        trt_runtime.save_engine(engine, f"{model_name}.trt")

    logging.info('success')
