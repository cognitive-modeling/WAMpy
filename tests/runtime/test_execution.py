from wampy.compiler.compiled_program import init_compiled_program
from wampy.compiler.compiled_query import init_compiled_query
from wampy.runtime.interpreter import execution
from wampy.runtime.machine import init_machine
from wampy.status import WAMStatus


def test_run_initializes_query_cut_boundary(monkeypatch) -> None:
    machine = init_machine()
    state = machine.registers
    state.B[0] = -1
    state.B0[0] = 123
    observed = {}

    def fake_emulate(machine, compiled_program, compiled_query):
        observed["b"] = int(machine.registers.B[0])
        observed["b0"] = int(machine.registers.B0[0])
        return WAMStatus.SUCCESS

    monkeypatch.setattr(execution, "emulate", fake_emulate)

    execution.run(
        machine,
        init_compiled_program(),
        init_compiled_query(),
    )

    assert observed["b0"] == observed["b"]
