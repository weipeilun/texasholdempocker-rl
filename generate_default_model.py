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
        profile = builder.create_optimization_profile()
        # profile.set_shape(input_tensor_name, (1, 3, 224, 224), (max_batch_size, 3, 224, 224), (max_batch_size, 3, 224, 224))

        config = builder.create_builder_config()
        config.add_optimization_profile(profile)

        trt_model_engine = builder.build_engine(network, config)
        # trt_model_context = trt_model_engine.create_execution_context()

        # builder.build_cuda_engine(onnx_model.network)
        # trt_runtime.save_engine(trt_model_engine, f"{model_name}.trt")
        with open(f"{model_name}.trt", "wb") as f:
            f.write(trt_model_engine.serialize())

    logging.info('success')
