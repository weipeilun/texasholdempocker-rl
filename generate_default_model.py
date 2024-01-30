from env.workflow import *
from tools.param_parser import *
import tensorrt as trt
import onnxsim
import onnx


EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def build_engine(model_file):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(model_file, 'rb') as model:
        assert parser.parse(model.read())
        return builder.build_serialized_network(network, config)


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
    torch.onnx.export(model, torch.zeros(1, 28, dtype=torch.int32).to(model.device), f'{model_name}.onnx', opset_version=15)

    onnx_checkpoint = onnx.load(f"{model_name}.onnx")
    model_simple, is_simplify_success = onnxsim.simplify(onnx_checkpoint)
    assert is_simplify_success
    onnx.save(model_simple, f"{model_name}_simplified.onnx")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt_runtime = trt.Runtime(TRT_LOGGER)

    trt_model_engine = build_engine(f"{model_name}_simplified.onnx")
    with open(f"{model_name}.trt", "wb") as f:
        f.write(trt_model_engine)

    logging.info('success')
