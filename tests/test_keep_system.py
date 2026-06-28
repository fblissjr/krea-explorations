"""Logic tests for the Krea2 'keep system turn' encode hook.

The node feeds Comfy's existing ``encode_token_weights(..., template_end=...)`` escape hatch via a tiny,
reversible per-instance attribute so a system-role prompt survives into the conditioning. These tests pin
the wrapper's behavior without needing comfy/torch (the node imports comfy lazily), so they run in the
project venv.
"""

from krea2_explorations.krea2_keep_system_node import _ATTR, _HOOK_FLAG, _install, _wrap


def _orig(self, token_weight_pairs, template_end=-1):
    # stand-in for Krea2TEModel.encode_token_weights: echo the resolved template_end
    return (token_weight_pairs, template_end)


def test_default_is_unchanged_strip_behavior():
    w = _wrap(_orig)

    class M:
        pass

    m = M()
    # no attribute set -> default -1 preserved == Comfy's original auto-strip
    assert w(m, "tok") == ("tok", -1)


def test_instance_attribute_overrides_default():
    w = _wrap(_orig)

    class M:
        pass

    m = M()
    setattr(m, _ATTR, 0)  # 0 = keep the whole sequence incl. the system turn
    assert w(m, "tok") == ("tok", 0)


def test_none_attribute_is_passthrough():
    w = _wrap(_orig)

    class M:
        pass

    m = M()
    setattr(m, _ATTR, None)
    assert w(m, "tok") == ("tok", -1)


def test_explicit_arg_wins_over_attribute():
    w = _wrap(_orig)

    class M:
        pass

    m = M()
    setattr(m, _ATTR, 5)
    # an explicit non-default template_end from a caller is respected, attribute ignored
    assert w(m, "tok", template_end=3) == ("tok", 3)


def test_wrapper_exposes_original():
    w = _wrap(_orig)
    assert w.__wrapped__ is _orig


def test_install_is_idempotent():
    # _install wraps once and flags the fn; a second call must be a no-op (no double-wrap).
    class Fake:
        def encode_token_weights(self, token_weight_pairs, template_end=-1):
            return template_end

    _install(Fake)
    first = Fake.encode_token_weights
    assert getattr(first, _HOOK_FLAG, False) is True

    _install(Fake)  # second install -> same wrapped fn, not re-wrapped
    assert Fake.encode_token_weights is first

    # and the installed hook still honors the per-instance attribute
    f = Fake()
    assert f.encode_token_weights("tok") == -1
    setattr(f, _ATTR, 0)
    assert f.encode_token_weights("tok") == 0
