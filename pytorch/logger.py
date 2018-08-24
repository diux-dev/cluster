from tensorboardX import SummaryWriter
import torch


class Logger:
  def __init__(self, output_dir):
    self.output_dir = output_dir
  def log(val):
    raise NotImplementedError()
    
  
class TensorboardLogger(Logger):
  def __init__(self, output_dir, is_master=False):
    super().__init__(output_dir)
    self.current_step = 0
    if is_master: self.writer = SummaryWriter(self.output_dir)
    else: self.writer = NoOp()
    self.log('first', time.time())

  def log(self, tag, val):
    """Log value to tensorboard (relies on global_example_count being set properly)"""
    if not self.writer: return
    self.writer.add_scalar(tag, val, self.current_step)

  def update_step_count(self, batch_total):
    self.current_step += batch_total

  def close(self):
    self.writer.export_scalars_to_json(self.output_dir+'/scalars.json')
    self.writer.close()

  # Convenience logging methods
  def log_size(self, bs=None, sz=None):
    if bs: self.log('sizes/batch', bs)
    if sz: self.log('sizes/image', sz)
    
  def log_eval(self, top1, top5, time):
    self.log('losses/test_1', top1)
    self.log('losses/test_5', top5)
    self.log('times/eval_sec', time)

  def log_trn_loss(self, loss, top1, top5):
    self.log("losses/xent", loss)      # cross_entropy
    self.log("losses/train_1", top1)   # precision@1
    self.log("losses/train_5", top5)   # precision@5

  def log_memory(self):
    if not self.writer: return
    self.log("memory/allocated_gb", torch.cuda.memory_allocated()/1e9)
    self.log("memory/max_allocated_gb", torch.cuda.max_memory_allocated()/1e9)
    self.log("memory/cached_gb", torch.cuda.memory_cached()/1e9)
    self.log("memory/max_cached_gb", torch.cuda.max_memory_cached()/1e9)

  def log_trn_times(self, batch_time, data_time, batch_size):
    if not self.writer: return
    self.log("times/step", 1000*batch_time)
    self.log("times/data", 1000*data_time)
    images_per_sec = batch_size/batch_time
    self.log("times/1gpu_images_per_sec", images_per_sec)
    self.log("times/8gpu_images_per_sec", 8*images_per_sec)


class FileLogger(Logger):
  def __init__(self, output_dir, is_master=False):
    super().__init__(output_dir)
    self.current_step = 0
    print('File log:', output_dir+'/verbose.log')
    if is_master:
      bufsize = 5
      self.verbose_log = open(output_dir+'/verbose.log', 'a', 1)
      self.event_log = open(output_dir+'/event.log', 'a', 1)
    else:
      self.verbose_log = NoOp()
      self.event_log = NoOp()

  def log(self, val):
    print(val)

  def log_verbose(self, val):
    self.log(val)
    self.verbose_log.write(str(val)+'\n')

  def log_event(self, val): 
    self.log_verbose(val)
    self.event_log.write(str(val)+'\n')

  def close(self):
    self.verbose_log.close()
    self.event_log.close()

# no_op method/object that accept every signature
class NoOp:
  def __getattr__(self, *args):
    def no_op(*args, **kwargs): pass
    return no_op