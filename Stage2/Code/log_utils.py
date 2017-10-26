from datetime import datetime

def usetime(desc=None):
    def _usetime(func):
        def __usetime(*args, **kwargs):
            if desc: print('[Desc]: ' +desc)
            print('Runing function: [%s] ' % func.__name__)
            stime = datetime.now()
            res = func(*args, **kwargs)
            utime = datetime.now() - stime
            print('Use Time: {}'.format(utime))
            return res
        return __usetime
    return _usetime
