from wampy.status import WAMStatus


def test_wam_status_codes():
    expected = {
        "SUCCESS": 0,
        "EXHAUSTED": 1,
        "UNDEFINED_PREDICATE": 100,
        "ARITY_MISMATCH": 101,
        "INVALID_QUERY": 102,
        "INVALID_FUNCTOR": 103,
        "STACK_OVERFLOW": 200,
        "TRAIL_OVERFLOW": 201,
        "HEAP_OVERFLOW": 202,
        "ENVIRONMENT_OVERFLOW": 203,
        "STEP_LIMIT": 204,
        "UNIFY_STEP_LIMIT": 205,
        "DEPTH_LIMIT": 206,
        "INVALID_OPCODE": 300,
        "INVALID_PC": 301,
        "CORRUPT_CHOICEPOINT": 302,
        "CORRUPT_ENVIRONMENT": 303,
    }

    actual = {status.name: status.value for status in WAMStatus}

    assert actual == expected
    assert all(WAMStatus(value) is WAMStatus[name] for name, value in expected.items())
