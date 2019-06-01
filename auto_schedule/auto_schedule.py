import tvm
import logging
import sys
import os
from tvm import autotvm

def auto_schedule(func, args):
    """Automatic scheduler
    
    Args:
    -----------------
    func: function object
        similar to batch_gemm function mentioned above
    args: tuple
        inputs to func
    -----------------
    Returns:
    s: tvm.schedule.Schedule
    bufs: list of tvm.tensor.Tensor
    """

    ops, bufs = func(*args)

    @autotvm.template
    def gemm():
        assert len(bufs) == 3
        s = tvm.create_schedule(ops)
        C = bufs[2]

        _, y, x = s[C].op.axis
        k = s[C].op.reduce_axis[0]

        cfg = autotvm.get_config()

        cfg.define_split("split_k", k, num_outputs=2)
        cfg.define_split("tile_y", y, num_outputs=2)
        cfg.define_split("tile_x", x, num_outputs=2)

        # schedule according to config
        ko, ki = cfg["split_k"].apply(s, C, k)
        yo, yi = cfg["tile_y"].apply(s, C, y)
        xo, xi = cfg["tile_x"].apply(s, C, x)

        s[C].reorder(xo, yo, ko, xi, ki, yi)
        s[C].vectorize(yi)

        return s, bufs

    @autotvm.template
    def conv():
        s = tvm.create_schedule(ops)
        return s, bufs

    if len(args) == 4:
        template = gemm
    elif len(args) == 13:
        template = conv
    else:
        print('unknown operation')
        s = tvm.create_schedule(ops)
        return s, bufs

    task = autotvm.task.create(template, args=(), target='llvm')
    print(task.config_space)

    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

    measure_option = autotvm.measure_option(
        builder='local',
        runner=autotvm.LocalRunner(number=5)
    )

    print('=====begin tuning for ' + str(args))
    filename = 'tune.log'
    if os.path.exists(filename):
        os.remove(filename)

    tuner = autotvm.tuner.XGBTuner(task, loss_type='rank')
    tuner.tune(n_trial=100,
               measure_option=measure_option,
               callbacks=[autotvm.callback.log_to_file(filename)])

    print('=====finish tuning for ' + str(args))

    with autotvm.apply_history_best(filename):
        with tvm.target.create("llvm"):
            return template()
