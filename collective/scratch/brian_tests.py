import ray
#Brian's Local Tests. 

ray.init()

class CollectiveActorClass:

    def __init__(self):
        self._pos = {}
        # temp hash for keys into pos.
        self._hash = 1

    # def __call__(self, *args, **kwargs):
    #     rv = self._func(*args, **kwargs)
    #     return objref

    def _gen_pos_ref(self, value):
        #TODO: generate some hash that actually makes sense. 
        objref = str(self._hash)
        self._hash += 1
        return objref

    def set_pos(self, objref, value):
        self._pos[objref] = value

    def get_pos(self, objref):
        return self._pos[objref]

    def promote_to_raystore(self, objref):
        #TODO: mechanism to make things to raystore. 
        pass


def pos(method):
    """
    POS Wrapper for CollectiveActor methods.
    """
    def _modified(*args, **kwargs):
        #first must be object for method
        self = args[0]
        print(type(self))


        rv = method(*args, **kwargs)
        objref = self._gen_pos_ref(rv)
        self.set_pos(objref, rv)
        print("DEBUG: stored {} at {}".format(rv, objref))
        return

    return _modified


def CollectiveActor(cls):

    #Let me know if there's a better way to do this than multiple inheritance. 
    class _CollectiveActor(CollectiveActorClass, cls):
        def __init__(self, *args, **kwargs):
            CollectiveActorClass.__init__(self)
            cls.__init__(self, *args, **kwargs)

    return _CollectiveActor


"""
Testing Code. 
"""
@ray.remote
@CollectiveActor
class MyActor:
    def __init__(self):
        self.buffer = "objresct"

    # get_buffer was a user function that returns self.buffer.
    @pos
    def get_buffer(self):
        buffer = self.buffer
        return buffer


if __name__ == '__main__':
    actor = MyActor.remote()
    print(actor.get_buffer.remote())
    
