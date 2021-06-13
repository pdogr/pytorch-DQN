from pkgs import *


class Logger():
    def __init__(self, session_name, log_dir):
        filename = os.path.join(log_dir, session_name+'.log')
        file_handler = logging.FileHandler(filename=filename)
        stdout_handler = logging.StreamHandler(sys.stdout)
        handlers = [file_handler, stdout_handler]

        logging.basicConfig(format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
                            level=logging.DEBUG, handlers=handlers)
        self.logger = logging.getLogger(session_name)
        self.log_fns = ['debug', 'info', 'warning']

        self.writer = None
        self.selected_module = ""

        succeeded = False
        for module in ["torch.utils.tensorboard", "tensorboardX"]:
            try:
                self.writer = importlib.import_module(
                    module).SummaryWriter('runs/'+session_name)
                succeeded = True
                break
            except ImportError:
                succeeded = False
            self.selected_module = module

        if not succeeded:
            message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
            logger.warning(message)

        self.step = 0
        self.mode = ''

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images',
            'add_audio', 'add_text', 'add_histogram', 'add_pr_curve',
            'add_embedding'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.timer = datetime.now()
        self.step = 0
        self.session_name = session_name

    def __getattr__(self, name):
        if name in self.log_fns:
            log_fn = getattr(self.logger, name, None)
            return log_fn
        elif name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(self.session_name, tag)
                    add_data(tag, data, self.step, *args, **kwargs)

            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError(
                    "type object '{}' has no attribute '{}'".format(
                        self.selected_module, name))
            return attr

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    def __exit__(self, exc_type, exc_value, traceback):
        self.writer.close()
