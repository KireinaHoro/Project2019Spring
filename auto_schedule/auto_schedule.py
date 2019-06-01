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
    #################################################
    # do some thing with `ops`, `bufs` and `args`
    # to analyze which schedule is appropriate

    # TODO: replace this with autoschedule
    s = tvm.create_schedule(ops)

    @autotvm.template
    def gemm():
        assert len(bufs) == 3
        s = tvm.create_schedule(ops)

        C = bufs[2]

        _, y, x = s[C].op.axis
        k = s[C].op.reduce_axis[0]
        ko, ki = s[C].split(k, factor=4)

        cfg = autotvm.get_config()

        #cfg.define_split("split_k", k, num_outputs=2)
        cfg.define_split("tile_y", y, num_outputs=2)
        cfg.define_split("tile_x", x, num_outputs=2)
        ##### define space end #####

        # schedule according to config
        #ko, ki = cfg["split_k"].apply(s, C, k)
        yo, yi = cfg["tile_y"].apply(s, C, y)
        xo, xi = cfg["tile_x"].apply(s, C, x)

        s[C].reorder(xo, yo, ko, xi, ki, yi)
        s[C].vectorize(yi)

        return s, bufs

    if len(args) == 4:
        task = autotvm.task.create(gemm, args=(), target='llvm')
        print(task.config_space)

        logging.getLogger('autotvm').setLevel(logging.DEBUG)
        logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

        measure_option = autotvm.measure_option(
            builder='local',
            runner=autotvm.LocalRunner(number=5)
        )

        print('=====begin tuning for ' + str(args))
        filename = 'gemm.log'
        if os.path.exists(filename):
            os.remove(filename)

        tuner = autotvm.tuner.GATuner(task)
        tuner.tune(n_trial=20,
                   measure_option=measure_option,
                   callbacks=[autotvm.callback.log_to_file(filename)])

        print('=====finish tuning for ' + str(args))

        with autotvm.apply_history_best(filename):
            with tvm.target.create("llvm"):
                return gemm()

    elif len(args) == 13:
        print('conv2d')
    else:
        print('unknown operation')
    
    #################################################
    # perform real schedule according to 
    # decisions made above, using primitives 
    # such as split, reorder, parallel, unroll...
    
    #################################################
    # finally, remember to return these two results
    # we need `bufs` to build function via `tvm.build`
    return s, bufs
