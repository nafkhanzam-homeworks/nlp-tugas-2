import os
import pickle


def cache(cachefile: str):
    """
    A function that creates a decorator which will use "cachefile" for caching the results of the decorated function "fn".
    """
    cachefile = f".cache/{cachefile}.pkl"

    def decorator(fn):  # define a decorator for a function "fn"
        # define a wrapper that will finally call "fn" with all arguments
        def wrapped(*args, **kwargs):
            # if cache exists -> load it and return its content
            if os.path.exists(cachefile):
                with open(cachefile, 'rb') as cachehandle:
                    print("using cached result from '%s'" % cachefile)
                    return pickle.load(cachehandle)

            # execute the function with all arguments passed
            res = fn(*args, **kwargs)

            os.makedirs(os.path.dirname(cachefile), exist_ok=True)

            # write to cache file
            with open(cachefile, 'wb') as cachehandle:
                print("saving result to cache '%s'" % cachefile)
                if hasattr(res, 'to_pickle'):
                    res.to_pickle(cachefile)
                else:
                    pickle.dump(res, cachehandle)

            return res

        return wrapped

    return decorator   # return this "customized" decorator that uses "cachefile"


def build_cache_decorator(prefix: str):
    def cache_wrapper(cachefile: str):
        return cache(f"{prefix}:{cachefile}")

    return cache_wrapper
