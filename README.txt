Install
-------

virtualenv venv
source venv/bin/activate
pip install -r requirements.txt


Run
---

accelerate launch --config_file accelerate.cfg train_sentiment.py

or

accelerate launch --config_file accelerate.cfg train_mlm.py


Error Output
------------

Traceback (most recent call last):
  File "train_sentiment.py", line 63, in <module>
    accelerator.backward(loss)
  File "/home/agrizzli/test_use_orig_params/venv/lib/python3.8/site-packages/accelerate/accelerator.py", line 1737, in backward
    loss.backward(**kwargs)
  File "/home/agrizzli/test_use_orig_params/venv/lib/python3.8/site-packages/torch/_tensor.py", line 488, in backward
    torch.autograd.backward(
  File "/home/agrizzli/test_use_orig_params/venv/lib/python3.8/site-packages/torch/autograd/__init__.py", line 204, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
  File "/home/agrizzli/test_use_orig_params/venv/lib/python3.8/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/home/agrizzli/test_use_orig_params/venv/lib/python3.8/site-packages/torch/distributed/fsdp/_runtime_utils.py", line 708, in _post_backward_hook
    handle._use_unsharded_grad_views()
  File "/home/agrizzli/test_use_orig_params/venv/lib/python3.8/site-packages/torch/distributed/fsdp/flat_param.py", line 1747, in _use_unsharded_grad_views
    param.grad = view
RuntimeError: assigned grad has data of a different type
Traceback (most recent call last):
  File "train_sentiment.py", line 63, in <module>
    accelerator.backward(loss)
  File "/home/agrizzli/test_use_orig_params/venv/lib/python3.8/site-packages/accelerate/accelerator.py", line 1737, in backward
    loss.backward(**kwargs)
  File "/home/agrizzli/test_use_orig_params/venv/lib/python3.8/site-packages/torch/_tensor.py", line 488, in backward
    torch.autograd.backward(
  File "/home/agrizzli/test_use_orig_params/venv/lib/python3.8/site-packages/torch/autograd/__init__.py", line 204, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
  File "/home/agrizzli/test_use_orig_params/venv/lib/python3.8/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/home/agrizzli/test_use_orig_params/venv/lib/python3.8/site-packages/torch/distributed/fsdp/_runtime_utils.py", line 708, in _post_backward_hook
    handle._use_unsharded_grad_views()
  File "/home/agrizzli/test_use_orig_params/venv/lib/python3.8/site-packages/torch/distributed/fsdp/flat_param.py", line 1747, in _use_unsharded_grad_views
    param.grad = view
RuntimeError: assigned grad has data of a different type
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 3545430 closing signal SIGTERM
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 1 (pid: 3545432) of binary: /home/agrizzli/test_use_orig_params/venv/bin/python
Traceback (most recent call last):
  File "/home/agrizzli/test_use_orig_params/venv/bin/accelerate", line 8, in <module>
    sys.exit(main())
  File "/home/agrizzli/test_use_orig_params/venv/lib/python3.8/site-packages/accelerate/commands/accelerate_cli.py", line 45, in main
    args.func(args)
  File "/home/agrizzli/test_use_orig_params/venv/lib/python3.8/site-packages/accelerate/commands/launch.py", line 921, in launch_command
    multi_gpu_launcher(args)
  File "/home/agrizzli/test_use_orig_params/venv/lib/python3.8/site-packages/accelerate/commands/launch.py", line 612, in multi_gpu_launcher
    distrib_run.run(args)
  File "/home/agrizzli/test_use_orig_params/venv/lib/python3.8/site-packages/torch/distributed/run.py", line 785, in run
    elastic_launch(
  File "/home/agrizzli/test_use_orig_params/venv/lib/python3.8/site-packages/torch/distributed/launcher/api.py", line 134, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/home/agrizzli/test_use_orig_params/venv/lib/python3.8/site-packages/torch/distributed/launcher/api.py", line 250, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError:
============================================================
train_sentiment.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2023-04-06_10:50:01
  host      : spartaw02
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 3545432)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
