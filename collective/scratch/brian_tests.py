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
        if objref not in self._pos:
            raise KeyError("objref does not exist in {}'s pos store.".format(self))
        obj = self._pos.pop(objref)
        if obj:
            ray_ref = ray.put(obj)
            return ray_ref

        # didn't work.    
        return -1


def pos(method):
    """
    POS Wrapper for CollectiveActor methods.
    """
    def _modified(*args, **kwargs):
        if not len(args) > 0:
            # Brian's notes:
            # We can implement an @pos for non-methods, e.g. standalone functions.
            # let me know if that seems like something we would be interested in
            # doing. So far the API specifications only list for methods. 
            raise TypeError("Must be a method!")
        #first must be object for method
        collective_actor = args[0]

        if not isinstance(collective_actor, CollectiveActorClass):
            raise TypeError("Can only wrap @pos for Collective Actors.")

        rv = method(*args, **kwargs)
        objref = collective_actor._gen_pos_ref(rv)
        collective_actor.set_pos(objref, rv)
        print("DEBUG: stored {} at objref {}".format(
                                                    collective_actor.get_pos(objref), 
                                                    objref))

        # I wrote "return" explicitly because it's important that
        # this function returns nothing. This way Ray knows not to
        # generate an objref, and not to store anything. Otherwise,
        # there'd be no point of pos. 
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
        self.buffer = "obj"

    # get_buffer was a user function that returns self.buffer.
    @pos
    def get_buffer(self):
        buffer = self.buffer
        self.buffer = "changed"
        return buffer


if __name__ == '__main__':
    print("what")
    actor = MyActor.remote()
    actor.get_buffer.remote()
    ref = actor.get_buffer.remote()
    print('DEBUG: reached')
    ## Synchronizatiion issue. POS may finish before promote or get. What to do?
    raay_ref = actor.promote_to_raystore.remote(ref)
    print(raay_ref)
    print(ray.get(raay_ref))
    
