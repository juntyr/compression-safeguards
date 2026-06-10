import importlib.metadata


def define_env(env):
    @env.macro
    def version(x):
        return importlib.metadata.version(x)
