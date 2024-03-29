import json
import tempfile
import shutil
import os

class LogReport():

    """Trainer extension to output the accumulated results to a log file.
    This extension accumulates the observations of the trainer to
    :class:`~chainer.DictSummary` at a regular interval specified by a supplied
    trigger, and writes them into a log file in JSON format.
    There are two triggers to handle this extension. One is the trigger to
    invoke this extension, which is used to handle the timing of accumulating
    the results. It is set to ``1, 'iteration'`` by default. The other is the
    trigger to determine when to emit the result. When this trigger returns
    True, this extension appends the summary of accumulated values to the list
    of past summaries, and writes the list to the log file. Then, this
    extension makes a new fresh summary object which is used until the next
    time that the trigger fires.
    It also adds some entries to each result dictionary.
    - ``'epoch'`` and ``'iteration'`` are the epoch and iteration counts at the
      output, respectively.
    - ``'elapsed_time'`` is the elapsed time in seconds since the training
      begins. The value is taken from :attr:`Trainer.elapsed_time`.
    Args:
        keys (iterable of strs): Keys of values to accumulate. If this is None,
            all the values are accumulated and output to the log file.
        trigger: Trigger that decides when to aggregate the result and output
            the values. This is distinct from the trigger of this extension
            itself. If it is a tuple in the form ``<int>, 'epoch'`` or
            ``<int>, 'iteration'``, it is passed to :class:`IntervalTrigger`.
        postprocess: Callback to postprocess the result dictionaries. Each
            result dictionary is passed to this callback on the output. This
            callback can modify the result dictionaries, which are used to
            output to the log file.
        log_name (str): Name of the log file under the output directory. It can
            be a format string: the last result dictionary is passed for the
            formatting. For example, users can use '{iteration}' to separate
            the log files for different iterations. If the log name is None, it
            does not output the log to any file.
    """

    def __init__(self, keys=None,
                 log_name='log'):
        self._keys = keys
        self._log_name = log_name
        self._log = []

    def __call__(self, trainer):
        # accumulate the observations
        keys = self._keys
        observation = trainer.observation
        
        summary = {}
        if keys is None:
            summary.update(observation)
        else:
            summary.update({k: observation[k] for k in keys if k in observation})

        self._log.append(summary)

        # write to the log file
        if self._log_name is not None:
            log_name = self._log_name.format(**summary)
            fd, path = tempfile.mkstemp(prefix=log_name, dir=trainer.out)
            with os.fdopen(fd, 'w') as f:
                json.dump(self._log, f, indent=4)

            new_path = os.path.join(trainer.out, log_name)
            shutil.move(path, new_path)

    @property
    def log(self):
        """The current list of observation dictionaries."""
        return self._log

    def serialize(self, serializer):
        if hasattr(self._trigger, 'serialize'):
            self._trigger.serialize(serializer['_trigger'])

        # Note that this serialization may lose some information of small
        # numerical differences.
        if isinstance(serializer, serializer_module.Serializer):
            log = json.dumps(self._log)
            serializer('_log', log)
        else:
            log = serializer('_log', '')
            self._log = json.loads(log)

