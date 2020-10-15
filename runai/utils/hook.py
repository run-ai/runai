class Hook(object):
    def __init__(self, module, method, hook=None, recursion=False):
        self.module = module
        self.method = method
        self.hook = hook if hook else self.__hook__
        self.recursion = recursion
        self.__original__ = getattr(module, method)

    def enable(self):
        def hook(*args, **kwargs):
            if hook.called and not self.recursion:
                return self.__original__(*args, **kwargs)

            hook.called = True

            try:
                result = self.hook(*args, **kwargs)
            finally:
                hook.called = False

            return result

        hook.called = False

        setattr(self.module, self.method, hook)

    def disable(self):
        setattr(self.module, self.method, self.__original__)

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable()

    def __hook__(self):
        raise NotImplementedError()
