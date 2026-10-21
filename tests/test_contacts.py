"""create_contact: a confirmed, validated, idempotent, persisted write."""
import shutil

import pytest

from mocks import Backend
from mocks.loader import load_contacts
from tests.conftest import TEST_DATA
from tests.fakes import Script, reissue_confirmed, tc
from tools import build_tool_groups, is_write_tool, tools_by_name
from tools.base import ErrorCode

NEW = dict(name="Alok Kumar", email="alok@gmail.com", title="Engineer", company="Agyom", phone="4989899382")


def run(backend, tool_name, /, **args):
    return tools_by_name(build_tool_groups(backend))[tool_name].metadata["runner"](args)


@pytest.fixture
def data_dir(tmp_path):
    folder = tmp_path / "data"
    shutil.copytree(TEST_DATA, folder)
    return folder


def persistent(settings, clock, folder):
    return Backend(settings.model_copy(update={"mock_data_dir": folder, "mock_persist": True}), clock=clock)


def test_create_contact_is_a_crm_write_tool(settings, clock):
    tool = tools_by_name(build_tool_groups(Backend(settings, clock=clock)))["create_contact"]
    assert is_write_tool(tool) and tool.metadata["group"] == "crm" and "DO NOT" in tool.description


def test_create_contact_then_find_it(settings, clock):
    b = Backend(settings, clock=clock)
    res = run(b, "create_contact", **NEW)
    assert res.ok and res.data["contact"] == {"id": "con_006", **NEW}
    found = run(b, "get_contact", email="ALOK@gmail.com")
    assert found.ok and found.data["contact"]["name"] == "Alok Kumar"
    assert run(b, "get_contact", name="alok").data["contact"]["id"] == "con_006"
    assert run(b, "list_interactions", contact_id="con_006").data["interactions"] == []  # known contact, no history


def test_optional_fields_and_whitespace(settings, clock):
    b = Backend(settings, clock=clock)
    res = run(b, "create_contact", name="  Zoe Park ", email=" zoe@park.io ", company="  ", phone="")
    assert res.ok and res.data["contact"]["name"] == "Zoe Park" and res.data["contact"]["email"] == "zoe@park.io"
    assert res.data["contact"]["company"] is None and res.data["contact"]["phone"] is None


@pytest.mark.parametrize(
    "args",
    [{"name": "No Email"}, {"email": "a@b.co"}, {"name": "X", "email": "not-an-email"}, {"name": "X", "email": "a b@c.io"},
     {"name": "   ", "email": "a@b.co"}, {"name": "X", "email": "a@b.co", "id": "con_9"}],
)
def test_invalid_arguments_are_rejected(settings, clock, args):
    res = run(Backend(settings, clock=clock), "create_contact", **args)
    assert res.error.code is ErrorCode.INVALID_ARGS and not res.error.retryable


def test_duplicate_email_is_a_conflict_not_a_second_contact(settings, clock):
    b = Backend(settings, clock=clock)
    res = run(b, "create_contact", name="Another Priya", email="PRIYA.SHARMA@acme.com")
    assert res.error.code is ErrorCode.CONFLICT and not res.error.retryable
    assert res.data["existing"]["id"] == "con_001" and "already exists" in res.error.message
    assert len(b.crm._contacts) == 5


def test_same_name_different_email_is_allowed_and_becomes_ambiguous(settings, clock):
    b = Backend(settings, clock=clock)
    assert run(b, "create_contact", name="John Smith", email="john@elsewhere.com").ok
    assert run(b, "get_contact", name="John Smith").error.code is ErrorCode.AMBIGUOUS


def test_retried_create_after_a_lost_response_makes_one_contact(settings, clock):
    b = Backend(settings, clock=clock)
    runner = tools_by_name(build_tool_groups(b))["create_contact"].metadata["runner"]
    b.fail_next("create_contact", "SERVER_ERROR", 1, after_commit=True)
    first = runner(NEW, request_id="req-1")
    assert first.error.code is ErrorCode.SERVER_ERROR and first.error.retryable
    again = runner(NEW, request_id="req-1")
    assert again.ok and again.data["idempotent_replay"] and again.data["contact"]["id"] == "con_006"
    assert sum(1 for c in b.crm._contacts if c["email"] == "alok@gmail.com") == 1


def test_contact_is_written_to_the_csv_and_survives_a_restart(settings, clock, data_dir):
    b = persistent(settings, clock, data_dir)
    assert run(b, "create_contact", **NEW).ok
    saved = {c["id"]: c for c in load_contacts(data_dir)}
    assert saved["con_006"]["email"] == "alok@gmail.com" and saved["con_006"]["company"] == "Agyom"
    assert len(saved) == 6 and saved["con_001"]["name"] == "Priya Sharma"  # existing rows are preserved
    again = persistent(settings, clock, data_dir)
    assert again.crm.get_contact(None, "alok@gmail.com")["contact"]["phone"] == "4989899382"
    assert run(again, "create_contact", name="Next", email="next@x.io").data["contact"]["id"] == "con_007"


def test_failed_or_duplicate_create_leaves_the_csv_untouched(settings, clock, data_dir):
    b = persistent(settings, clock, data_dir)
    before = (data_dir / "contacts.csv").read_text()
    assert not run(b, "create_contact", name="Dup", email="john.smith@umbrella.io").ok
    b.fail_next("create_contact", "SERVER_ERROR")
    assert not run(b, "create_contact", **NEW).ok
    assert (data_dir / "contacts.csv").read_text() == before


def test_no_file_is_written_unless_persistence_is_on(settings, clock, data_dir):
    before = (data_dir / "contacts.csv").read_text()
    b = Backend(settings.model_copy(update={"mock_data_dir": data_dir}), clock=clock)
    assert run(b, "create_contact", **NEW).ok
    assert (data_dir / "contacts.csv").read_text() == before


# --------------------------------------------------------------------------- through the agent
def test_agent_asks_for_confirmation_then_adds_the_contact(make_agent, settings, clock, data_dir):
    backend = persistent(settings, clock, data_dir)
    script = Script([
        tc("create_contact", **NEW),
        "Add Alok Kumar <alok@gmail.com>, Engineer at Agyom, phone 4989899382. Shall I go ahead?",
        lambda c: reissue_confirmed(c),
        lambda c: "Added Alok Kumar to your contacts." if c.result_of("create_contact")["ok"] else "failed",
    ])
    agent, _ = make_agent(script, backend=backend)
    first = agent.invoke("add contact of alok kumar with email alok@gmail.com who is engineer in agyom with phone 4989899382")
    assert first.groups == ["crm"] and first.tool_events == []
    assert all(c["email"] != "alok@gmail.com" for c in load_contacts(data_dir))  # nothing saved before "yes"
    pending = agent.memory().pending_action
    assert pending and pending.tool == "create_contact" and "Alok Kumar" in pending.summary and "Agyom" in pending.summary

    done = agent.invoke("yes")
    assert done.tools_called == ["create_contact"] and "Added" in done.text
    assert any(c["email"] == "alok@gmail.com" for c in load_contacts(data_dir))
    mem = agent.memory()
    assert mem.pending_action is None and mem.last_contact["email"] == "alok@gmail.com"


def test_agent_cannot_add_a_contact_with_changed_details_after_yes(make_agent, settings, clock):
    script = Script([
        tc("create_contact", **NEW), "Confirm?",
        tc("create_contact", **{**NEW, "email": "other@gmail.com"}), "That email differs from what you confirmed.",
    ])
    agent, _ = make_agent(script, backend=Backend(settings, clock=clock))
    agent.invoke("add contact alok kumar alok@gmail.com")
    res = agent.invoke("yes")
    assert res.tool_events == []
    assert all(c["email"] != "other@gmail.com" for c in agent.backend.crm._contacts)


# ------------------------------------------------------------------------------ delete_contact
def test_delete_contact_removes_it_and_is_idempotent(settings, clock):
    b = Backend(settings, clock=clock)
    run(b, "create_contact", **NEW)
    res = run(b, "delete_contact", contact_id="con_006")
    assert res.ok and res.data["status"] == "deleted" and res.data["name"] == "Alok Kumar"
    assert run(b, "get_contact", email="alok@gmail.com").error.code is ErrorCode.NOT_FOUND
    again = run(b, "delete_contact", contact_id="con_006")
    assert again.ok and again.data["status"] == "already_gone"  # not a failure
    assert run(b, "list_interactions", contact_id="con_006").error.code is ErrorCode.NOT_FOUND


def test_delete_contact_error_codes(settings, clock):
    b = Backend(settings, clock=clock)
    assert run(b, "delete_contact", contact_id="con_999").error.code is ErrorCode.NOT_FOUND
    assert run(b, "delete_contact", contact_id="bogus").error.code is ErrorCode.INVALID_ARGS
    assert run(b, "delete_contact", contact_id="con_001", name="x").error.code is ErrorCode.INVALID_ARGS  # id only
    b.fail_next("delete_contact", "SERVER_ERROR")
    boom = run(b, "delete_contact", contact_id="con_001")
    assert boom.error.code is ErrorCode.SERVER_ERROR and boom.error.retryable
    assert len(b.crm._contacts) == 5  # a failed delete deletes nothing


def test_deleting_a_contact_drops_its_interactions_and_ids_are_not_reused(settings, clock):
    b = Backend(settings, clock=clock)
    assert run(b, "list_interactions", contact_id="con_003").data["count"] == 2
    run(b, "delete_contact", contact_id="con_003")
    assert all(i["contact_id"] != "con_003" for i in b.crm._interactions)
    assert run(b, "create_contact", name="New", email="new@x.io").data["contact"]["id"] == "con_006"


def test_delete_contact_is_written_to_the_csv(settings, clock, data_dir):
    b = persistent(settings, clock, data_dir)
    run(b, "create_contact", **NEW)
    run(b, "delete_contact", contact_id="con_002")
    run(b, "delete_contact", contact_id="con_006")
    ids = [c["id"] for c in load_contacts(data_dir)]
    assert ids == ["con_001", "con_003", "con_004", "con_005"]
    assert [c["id"] for c in persistent(settings, clock, data_dir).crm._contacts] == ids


def test_agent_looks_up_then_confirms_then_deletes(make_agent, settings, clock, data_dir):
    backend = persistent(settings, clock, data_dir)
    run(backend, "create_contact", **NEW)
    script = Script([
        tc("get_contact", email="alok@gmail.com"),
        lambda c: tc("delete_contact", contact_id=c.result_of("get_contact")["data"]["contact"]["id"]),
        "Delete Alok Kumar <alok@gmail.com>? Please confirm.",
        lambda c: reissue_confirmed(c),
        lambda c: "Deleted Alok Kumar." if c.result_of("delete_contact")["ok"] else "failed",
    ], route=lambda c: {"groups": ["crm"], "confidence": 0.9})  # the fake keyword router would also pick calendar for "remove"
    agent, _ = make_agent(script, backend=backend)
    first = agent.invoke("remove contact with mail alok@gmail.com")
    assert first.tools_called == ["get_contact"] and first.groups == ["crm"]
    pending = agent.memory().pending_action
    assert pending and pending.tool == "delete_contact" and "Alok Kumar" in pending.summary and "alok@gmail.com" in pending.summary
    assert any(c["email"] == "alok@gmail.com" for c in load_contacts(data_dir))  # still there before "yes"

    done = agent.invoke("yes")
    assert done.tools_called == ["delete_contact"] and "Deleted" in done.text
    assert all(c["email"] != "alok@gmail.com" for c in load_contacts(data_dir))
    mem = agent.memory()
    assert mem.pending_action is None and mem.last_contact["status"] == "deleted"


def test_guessed_contact_id_is_rejected_and_other_ids_cannot_ride_a_confirmation(make_agent, settings, clock):
    script = Script([
        tc("delete_contact", contact_id="con_003"),  # never looked up
        tc("get_contact", name="Rahul Verma"),
        lambda c: tc("delete_contact", contact_id="con_003"),
        "Delete Rahul Verma? Please confirm.",
        tc("delete_contact", contact_id="con_001"),  # after "yes", but not the contact that was approved
        "I can only delete the contact you approved.",
    ])
    agent, _ = make_agent(script, backend=Backend(settings, clock=clock))
    first = agent.invoke("delete contact rahul")
    assert [e.guard_reason for e in first.events if e.event == "guard_block"][:1] == ["INVALID_ARGS"]
    second = agent.invoke("yes")
    assert second.tool_events == [] and len(agent.backend.crm._contacts) == 5


# ------------------------------------------------------------------------------ update_contact
def test_update_contact_changes_only_the_given_fields(settings, clock):
    b = Backend(settings, clock=clock)
    res = run(b, "update_contact", contact_id="con_001", title="CPO", phone="+91 99999 00000")
    assert res.ok and res.data["changed"] == ["title", "phone"]
    contact = res.data["contact"]
    assert contact["title"] == "CPO" and contact["phone"] == "+91 99999 00000"
    assert contact["name"] == "Priya Sharma" and contact["company"] == "Acme Corp"  # untouched
    assert run(b, "get_contact", name="Priya Sharma").data["contact"]["title"] == "CPO"


def test_update_name_and_email_and_clearing_optional_fields(settings, clock):
    b = Backend(settings, clock=clock)
    res = run(b, "update_contact", contact_id="con_004", name="  Ananya Iyer-Rao ", email="ananya@new.io", company="", phone="")
    assert res.ok and set(res.data["changed"]) == {"name", "email", "company", "phone"}
    contact = res.data["contact"]
    assert contact["name"] == "Ananya Iyer-Rao" and contact["email"] == "ananya@new.io"
    assert contact["company"] is None and contact["phone"] is None and contact["title"] == "Designer"
    assert run(b, "get_contact", email="ANANYA@new.io").ok
    assert run(b, "get_contact", email="ananya.iyer@initech.com").error.code is ErrorCode.NOT_FOUND


def test_update_with_identical_values_is_a_harmless_no_op(settings, clock):
    b = Backend(settings, clock=clock)
    res = run(b, "update_contact", contact_id="con_001", title="VP Product")
    assert res.ok and res.data["changed"] == [] and "nothing was changed" in res.data["note"]


def test_update_contact_errors(settings, clock):
    b = Backend(settings, clock=clock)
    assert run(b, "update_contact", contact_id="con_999", title="x").error.code is ErrorCode.NOT_FOUND
    assert run(b, "update_contact", contact_id="bogus", title="x").error.code is ErrorCode.INVALID_ARGS
    assert run(b, "update_contact", contact_id="con_001").error.code is ErrorCode.INVALID_ARGS  # nothing to change
    assert run(b, "update_contact", contact_id="con_001", email="nope").error.code is ErrorCode.INVALID_ARGS
    assert run(b, "update_contact", contact_id="con_001", name="  ").error.code is ErrorCode.INVALID_ARGS
    clash = run(b, "update_contact", contact_id="con_001", email="RAHUL.verma@globex.com")
    assert clash.error.code is ErrorCode.CONFLICT and clash.data["existing"]["id"] == "con_003" and not clash.error.retryable
    assert run(b, "get_contact", name="Priya Sharma").data["contact"]["email"] == "priya.sharma@acme.com"
    b.fail_next("update_contact", "SERVER_ERROR")
    boom = run(b, "update_contact", contact_id="con_001", title="x")
    assert boom.error.code is ErrorCode.SERVER_ERROR and boom.error.retryable
    assert run(b, "get_contact", name="Priya Sharma").data["contact"]["title"] == "VP Product"  # failed -> unchanged


def test_updating_a_deleted_contact_says_it_was_deleted(settings, clock):
    b = Backend(settings, clock=clock)
    run(b, "delete_contact", contact_id="con_005")
    res = run(b, "update_contact", contact_id="con_005", title="x")
    assert res.error.code is ErrorCode.NOT_FOUND and "deleted" in res.error.message


def test_update_contact_is_written_to_the_csv(settings, clock, data_dir):
    b = persistent(settings, clock, data_dir)
    assert run(b, "update_contact", contact_id="con_003", company="Globex Intl", phone="").ok
    saved = {c["id"]: c for c in load_contacts(data_dir)}
    assert saved["con_003"]["company"] == "Globex Intl" and saved["con_003"]["phone"] is None
    assert saved["con_003"]["name"] == "Rahul Verma" and len(saved) == 5
    assert persistent(settings, clock, data_dir).crm.get_contact("Rahul Verma", None)["contact"]["company"] == "Globex Intl"
    before = (data_dir / "contacts.csv").read_text()
    run(b, "update_contact", contact_id="con_003", company="Globex Intl")  # no-op
    assert not run(b, "update_contact", contact_id="con_003", email="priya.nair@initech.com").ok  # conflict
    assert (data_dir / "contacts.csv").read_text() == before


def test_agent_updates_a_contact_after_lookup_and_confirmation(make_agent, settings, clock, data_dir):
    backend = persistent(settings, clock, data_dir)
    script = Script([
        tc("get_contact", name="Rahul Verma"),
        lambda c: tc("update_contact", contact_id=c.result_of("get_contact")["data"]["contact"]["id"], title="CRO"),
        "Change Rahul Verma's title to CRO? Please confirm.",
        lambda c: reissue_confirmed(c),
        lambda c: "Updated Rahul's title to CRO." if c.result_of("update_contact")["ok"] else "failed",
    ], route=lambda c: {"groups": ["crm"], "confidence": 0.9})
    agent, _ = make_agent(script, backend=backend)
    first = agent.invoke("change Rahul Verma's title to CRO")
    pending = agent.memory().pending_action
    assert first.tools_called == ["get_contact"]
    assert pending and pending.tool == "update_contact" and "Rahul Verma" in pending.summary and "title -> CRO" in pending.summary
    assert next(c for c in load_contacts(data_dir) if c["id"] == "con_003")["title"] == "Head of Sales"  # not yet

    done = agent.invoke("yes")
    assert done.tools_called == ["update_contact"] and "CRO" in done.text
    assert next(c for c in load_contacts(data_dir) if c["id"] == "con_003")["title"] == "CRO"
    mem = agent.memory()
    assert mem.pending_action is None and mem.last_contact["title"] == "CRO"


def test_update_cannot_be_changed_after_confirmation_or_use_a_guessed_id(make_agent, settings, clock):
    script = Script([
        tc("update_contact", contact_id="con_003", title="CRO"),  # never looked up
        tc("get_contact", name="Rahul Verma"),
        tc("update_contact", contact_id="con_003", title="CRO"),
        "Change the title to CRO? Please confirm.",
        tc("update_contact", contact_id="con_003", title="CEO"),  # not what was approved
        "That is not the change you confirmed.",
    ], route=lambda c: {"groups": ["crm"], "confidence": 0.9})
    agent, _ = make_agent(script, backend=Backend(settings, clock=clock))
    first = agent.invoke("update rahul's title")
    assert [e.guard_reason for e in first.events if e.event == "guard_block"][:1] == ["INVALID_ARGS"]
    second = agent.invoke("yes")
    assert second.tool_events == []
    assert agent.backend.crm.get_contact("Rahul Verma", None)["contact"]["title"] == "Head of Sales"
