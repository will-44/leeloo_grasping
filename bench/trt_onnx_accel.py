"""Remplace le chemin torch_tensorrt de GraspGenX par ONNX -> TensorRT natif.

torch_tensorrt n'existe pas pour torch 2.12 (SM_87, aarch64). On garde les
wrappers upstream (TensorRTDiffusionHead / TensorRTModule) et on ne remplace
que la compilation : torch.onnx.export (batch dynamique) puis builder TensorRT
10.16 (python3-libnvinfer), engine mis en cache sur disque.

Usage : import trt_onnx_accel; trt_onnx_accel.patch()  (avant GraspGenXSampler)
"""
import hashlib
import logging
import os
import time

import tensorrt as trt
import torch
import torch.nn as nn

import graspgenx.models.tensorrt_utils as tu

log = logging.getLogger("trt_onnx_accel")
ENGINE_DIR = os.environ.get("GGX_TRT_DIR", "/bench/trt_engines")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
BUILD_LOG = []  # (label, export_s, build_s, ok, msg)


class TRTEngineModule(nn.Module):
    """Exécute un engine TensorRT sur des tenseurs torch CUDA (batch dynamique)."""

    def __init__(self, engine_bytes):
        super().__init__()
        self.runtime = trt.Runtime(TRT_LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

    def forward(self, *args):
        stream = torch.cuda.current_stream()
        for name, x in zip(self.inputs, args):
            x = x.contiguous().float()
            self.ctx.set_input_shape(name, tuple(x.shape))
            self.ctx.set_tensor_address(name, x.data_ptr())
            setattr(self, f"_keep_{name}", x)
        outs = []
        for name in self.outputs:
            y = torch.empty(tuple(self.ctx.get_tensor_shape(name)), device="cuda", dtype=torch.float32)
            self.ctx.set_tensor_address(name, y.data_ptr())
            outs.append(y)
        self.ctx.execute_async_v3(stream.cuda_stream)
        return outs[0] if len(outs) == 1 else tuple(outs)


def _build(module, example, dyn_names, label, precision, mn, opt, mx):
    sig = hashlib.sha1(f"{label}|{[tuple(e.shape) for e in example]}|{precision}|{mn}|{opt}|{mx}|{trt.__version__}".encode()).hexdigest()[:12]
    os.makedirs(ENGINE_DIR, exist_ok=True)
    eng_path = f"{ENGINE_DIR}/{label}_{precision}_{sig}.engine"
    if os.path.exists(eng_path):
        return TRTEngineModule(open(eng_path, "rb").read())
    onnx_path = eng_path.replace(".engine", ".onnx")
    t0 = time.perf_counter()
    in_names = [f"in{i}" for i in range(len(example))]
    torch.onnx.export(
        module, tuple(example), onnx_path, input_names=in_names, output_names=["out"],
        dynamic_axes={**{n: {0: "B"} for n in in_names}, "out": {0: "B"}},
        opset_version=17, dynamo=False,
    )
    t_exp = time.perf_counter() - t0
    builder = trt.Builder(TRT_LOGGER)
    net = builder.create_network(0)
    parser = trt.OnnxParser(net, TRT_LOGGER)
    if not parser.parse_from_file(onnx_path):
        errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed: {errs}")
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    if precision == "fp16":
        cfg.set_flag(trt.BuilderFlag.FP16)
    prof = builder.create_optimization_profile()
    for n, e in zip(in_names, example):
        tail = tuple(e.shape[1:])
        prof.set_shape(n, (mn, *tail), (opt, *tail), (mx, *tail))
    cfg.add_optimization_profile(prof)
    t1 = time.perf_counter()
    ser = builder.build_serialized_network(net, cfg)
    if ser is None:
        raise RuntimeError("TensorRT build failed")
    t_build = time.perf_counter() - t1
    open(eng_path, "wb").write(bytes(ser))
    BUILD_LOG.append((label, t_exp, t_build, True, eng_path))
    log.warning(f"[trt] {label}: export {t_exp:.1f}s, build {t_build:.1f}s -> {eng_path}")
    return TRTEngineModule(bytes(ser))


def compile_diffusion_head(head, *, obs_dim, sample_dim, device, min_batch=1, opt_batch=100,
                           max_batch=2000, precision="fp32", engine_cache_dir=None):
    head = head.to(device).eval()

    class _Wrap(nn.Module):  # signature positionnelle fixe pour l'export
        def __init__(self, h):
            super().__init__()
            self.h = h

        def forward(self, obs, ts, sample):
            return self.h(obs, ts, sample)

    ex = [torch.randn(opt_batch, obs_dim, device=device),
          torch.full((opt_batch,), 5.0, device=device),
          torch.randn(opt_batch, sample_dim, device=device)]
    try:
        trt_mod = _build(_Wrap(head), ex, None, "diffusion_head", precision, min_batch, opt_batch, max_batch)
        return tu.TensorRTDiffusionHead(head, trt_mod, min_batch, max_batch)
    except Exception as e:
        BUILD_LOG.append(("diffusion_head", 0, 0, False, repr(e)[:500]))
        log.warning(f"[trt] diffusion_head failed: {e!r}")
        return tu.TensorRTDiffusionHead(head, None, min_batch, max_batch)


def compile_mlp(module, *, in_dim, device, min_batch=1, opt_batch=500, max_batch=4000,
                precision="fp32", label="mlp", engine_cache_dir=None):
    module = module.to(device).eval()
    try:
        trt_mod = _build(module, [torch.randn(opt_batch, in_dim, device=device)], None,
                         label.replace(".", "_"), precision, min_batch, opt_batch, max_batch)
        return tu.TensorRTModule(module, trt_mod, min_batch, max_batch)
    except Exception as e:
        BUILD_LOG.append((label, 0, 0, False, repr(e)[:500]))
        log.warning(f"[trt] {label} failed: {e!r}")
        return tu.TensorRTModule(module, None, min_batch, max_batch)


def patch():
    tu.tensorrt_available = lambda: True
    tu.compile_diffusion_head = compile_diffusion_head
    tu.compile_mlp = compile_mlp
