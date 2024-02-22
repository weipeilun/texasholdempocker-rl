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

    args, params = parse_params()
    model_param_dict = params['model_param_dict']
    model = AlphaGoZero(**model_param_dict)
    # to regenerate new default model
    save_model(model, f'{model_name}.pth')
    # to regenerate new default onnx model
    onnx_checkpoint = f"{model_name}.onnx"
    input_names = ['input']
    dynamic_axes = {'input': {0: 'batch_size'}}
    torch.onnx.export(model, torch.zeros((params['predict_batch_size_max'], *params['predict_feature_size_list']), dtype=torch.int32).to(model.device), onnx_checkpoint, opset_version=14, input_names=input_names, dynamic_axes=dynamic_axes)

    # onnxsim.simplify在未知原因的特定情况下会不抛异常导致进程直接崩溃：Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)。干掉这个步骤
    # onnx_checkpoint = f"{model_name}.onnx"
    # model_simple, is_simplify_success = onnxsim.simplify(onnx.load(onnx_checkpoint_tmp))
    # assert is_simplify_success
    # onnx.save(model_simple, onnx_checkpoint)
    # os.remove(onnx_checkpoint_tmp)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt_runtime = trt.Runtime(TRT_LOGGER)

    trt_model_engine = build_engine(onnx_checkpoint, params['predict_batch_size_min'], params['predict_batch_size'], params['predict_batch_size_max'], params['predict_feature_size_list'])
    with open(f"{model_name}.trt", "wb") as f:
        f.write(trt_model_engine)

    logging.info('success')
