from oneibl.one import ONE

one = ONE()


def eid2ref(eid):
    one.path_from_eid(eid)
    pass


def ref2eid(ref):
    session = one._alyxClient.get('sessions/' + eid)
    pass
