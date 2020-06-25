import re
from oneibl.one import ONE


def ref2eid(ref, one=None):
    if not one:
        one = ONE()

    session = one.search(
        subjects=ref['subject'],
        date_range=(ref['date'], ref['date']),
        number=ref['sequence'])
    assert len(session) == 1, 'session not found'
    return session[0]


def eid2ref(eid, one=None):
    """
    Get human-readable session ref from path
    :param eid: The experiment uuid to find reference for
    :param one: An ONE instance
    :return: dict containing 'subject', 'date' and 'sequence'
    """
    if not one:
        one = ONE()

    return path2ref(one.path_from_eid(eid))


def path2ref(path_str):
    pattern = r'(?P<subject>[\w-]+)([\\/])(?P<date>\d{4}-\d{2}-\d{2})(\2)(?P<sequence>\d{3})'
    match = re.search(pattern, str(path_str))
    return match.groupdict()
