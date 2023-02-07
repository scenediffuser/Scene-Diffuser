from torch.utils.tensorboard import SummaryWriter

def singleton(cls):
    _instance = {}

    def inner(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]
    return inner

@singleton
class _Writer():
    """ A singleton class that can hold the SummaryWriter Object.\n
    So we can initialize it once and use it everywhere.
    """
    def __init__(self) -> None:
        self.writer = None

    def write(self, write_dict: dict) -> None:
        """ Write the input dict data into writer object.

        Args:
            write_dict: a dict object containing data that need to be plotted. 
                Format is ```{key1: {'plot': bool, 'value':  float, 'step': long}}```. 
                `plot` means this value corresponding to this key needs to be plotted or not. 
                `value` is the specific value. `step` is the training step.
        """
        if self.writer is None:
            raise Exception('[ERR-CFG] Writer is None!')
        
        for key in write_dict.keys():
            if write_dict[key]['plot']:
                self.writer.add_scalar(key, write_dict[key]['value'], write_dict[key]['step'])

    def setWriter(self, writer: SummaryWriter) -> None:
        self.writer = writer

class Ploter():
    """ Ploter class for providing static methods to write data into SummaryWriter.
    """
    def __init__(self) -> None:
        pass

    @staticmethod
    def setWriter(writer: SummaryWriter) -> None:
        w = _Writer()
        w.setWriter(writer)
    
    @staticmethod
    def write(write_dict: dict) -> None:
        """ Plot input dict data.

        Args:
            write_dict: a dict object containing data that need to be plotted. 
                Format is ```{key1: {'plot': bool, 'value':  float, 'step': long}}```. 
                `plot` means this value corresponding to this key needs to be plotted or not. 
                `value` is the specific value. `step` is the training step.
        """
        w = _Writer()
        w.write(write_dict)