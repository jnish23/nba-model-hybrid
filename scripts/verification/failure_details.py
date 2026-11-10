"""Expose captured subprocess diagnostics for local fixture failures."""
import pytest

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item,call):
    outcome=yield
    report=outcome.get_result()
    if call.excinfo:
        e=call.excinfo.value
        parts=[str(getattr(e,k,'')) for k in ('stdout','stderr') if getattr(e,k,None)]
        if parts:report.sections.append(('local subprocess diagnostic','\n'.join(parts)))
