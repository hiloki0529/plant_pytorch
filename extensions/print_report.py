import sys
import os

class PrintReport:
    def __init__(self, entries, log_report="LogReport", out=sys.stdout):
        self._entries = entries
        self._log_report = log_report
        self._out = out

        self._log_len = 0  # number of observations already printed

        # format information
        entry_widths = [max(10, len(s)) for s in entries]

        header = '  '.join(('{:%d}' % w for w in entry_widths)).format(
            *entries) + '\n'
        self._header = header  # printed at the first call

        templates = []
        for entry, w in zip(entries, entry_widths):
            templates.append((entry, '{:<%dg}  ' % w, ' ' * (w + 2)))
        self._templates = templates

    def __call__(self, trainer):
        out = self._out

        if self._header:
            out.write(self._header)
            self._header = None

        log_report = self._log_report
        if isinstance(log_report, str):
            log_report = trainer.get_extension(log_report)
        elif isinstance(log_report, log_report_module.LogReport):
            log_report(trainer)  # update the log report
        else:
            raise TypeError('log report has a wrong type %s' %
                            type(log_report))

        log = log_report.log
        log_len = self._log_len
        while len(log) > log_len:
            # delete the printed contents from the current cursor
            if os.name == 'nt':
                util.erase_console(0, 0)
            else:
                out.write('\033[J')
            self._print(log[log_len])
            log_len += 1
        self._log_len = log_len

    def serialize(self, serializer):
        log_report = self._log_report
        if isinstance(log_report, log_report_module.LogReport):
            log_report.serialize(serializer['_log_report'])

    def _print(self, observation):
        out = self._out
        for entry, template, empty in self._templates:
            if entry in observation:
                out.write(template.format(observation[entry]))
            else:
                out.write(empty)
        out.write('\n')
