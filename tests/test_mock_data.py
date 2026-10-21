"""Mock data comes from CSV files and is validated on load."""
from datetime import time, timedelta

import pytest

from mocks import Backend
from mocks.loader import DEFAULT_DATA_DIR, MockDataError, load_contacts, load_events, load_interactions
from tests.conftest import NOW, TEST_DATA, tomorrow
from tools.base import ErrorCode


def write_data(folder, events=None, contacts=None, interactions=None):
    folder.mkdir(exist_ok=True)
    (folder / "events.csv").write_text(events or "title,day_offset,start,end\n")
    (folder / "contacts.csv").write_text(contacts or "id,name,email\n")
    (folder / "interactions.csv").write_text(interactions or "contact_id,days_ago,type,subject\n")
    return folder


def test_seed_csv_files_load():
    events, contacts, interactions = load_events(TEST_DATA), load_contacts(TEST_DATA), load_interactions(TEST_DATA)
    assert [e.title for e in events][:2] == ["Lunch with Ananya", "Team standup"]
    standup = events[1]
    assert (standup.day_offset, standup.start, standup.end, standup.location) == (1, time(10, 0), time(10, 30), "Zoom")
    assert events[0].attendees == ("ananya.iyer@initech.com",) and events[3].location is None
    assert [c["id"] for c in contacts] == ["con_001", "con_002", "con_003", "con_004", "con_005"]
    assert contacts[4]["phone"] == "+1 555 0100"
    assert interactions[0].contact_id == "con_001" and interactions[0].days_ago == 2


def test_shipped_data_files_are_valid():
    """The live files in mocks/data (which the agent edits) must always load."""
    assert load_events(DEFAULT_DATA_DIR) and load_contacts(DEFAULT_DATA_DIR) and load_interactions(DEFAULT_DATA_DIR)


def test_backend_is_seeded_from_the_csv_files_relative_to_today(settings, clock):
    b = Backend(settings, clock=clock)
    standup = b.calendar.get("evt_002")
    assert standup.title == "Team standup" and standup.start.isoformat() == tomorrow(10)
    assert b.crm.get_contact("Rahul Verma", None)["contact"]["email"] == "rahul.verma@globex.com"
    newest = b.crm.list_interactions("con_001")["interactions"][0]
    assert newest["date"].startswith("2026-10-19")  # 2 days before NOW


def test_custom_data_dir_replaces_the_seed_data(settings, clock, tmp_path):
    folder = write_data(
        tmp_path / "mine",
        events="title,day_offset,start,end,attendees,location,description\n"
               "Board prep,0,16:00,17:00,a@x.com; b@x.com,HQ,Slides\n",
        contacts="id,name,email,company\ncon_100,Zoe Park,zoe@park.io,ParkCo\n",
        interactions="contact_id,days_ago,type,subject,summary\ncon_100,3,call,Hello,Said hi\n",
    )
    b = Backend(settings.model_copy(update={"mock_data_dir": folder}), clock=clock)
    assert [e.title for e in b.calendar.all_events()] == ["Board prep"]
    event = b.calendar.get("evt_001")
    assert event.attendees == ["a@x.com", "b@x.com"] and event.description == "Slides"
    assert b.crm.get_contact("zoe", None)["contact"]["company"] == "ParkCo"
    assert b.crm.list_interactions("con_100")["count"] == 1
    created = b.calendar.create_event("New", NOW.replace(hour=18), NOW.replace(hour=19), [], None, None)
    assert created["event"]["id"] == "evt_002"  # ids continue after the CSV rows


def test_empty_files_give_an_empty_backend(settings, clock, tmp_path):
    b = Backend(settings.model_copy(update={"mock_data_dir": write_data(tmp_path / "empty")}), clock=clock)
    assert b.calendar.all_events() == []
    with pytest.raises(Exception) as exc:
        b.crm.get_contact("anyone", None)
    assert exc.value.code is ErrorCode.NOT_FOUND


@pytest.mark.parametrize(
    "events,message",
    [
        ("title,day_offset,start\nX,1,10:00\n", r"missing column\(s\) end"),
        ("title,day_offset,start,end\nX,tomorrow,10:00,11:00\n", "line 2: 'day_offset' must be a whole number"),
        ("title,day_offset,start,end\nX,1,10am,11:00\n", "'start' must be HH:MM"),
        ("title,day_offset,start,end\nX,1,11:00,10:00\n", "'end' must be after 'start'"),
        ("title,day_offset,start,end\n,1,10:00,11:00\n", "'title' is required"),
    ],
)
def test_malformed_event_rows_fail_with_file_and_line(tmp_path, events, message):
    with pytest.raises(MockDataError, match=message):
        load_events(write_data(tmp_path / "bad", events=events))


def test_other_validation_errors(tmp_path):
    with pytest.raises(MockDataError, match="duplicate contact id con_1"):
        load_contacts(write_data(tmp_path / "a", contacts="id,name,email\ncon_1,A,a@x.com\ncon_1,B,b@x.com\n"))
    with pytest.raises(MockDataError, match="'days_ago' must be a whole number"):
        load_interactions(write_data(tmp_path / "b", interactions="contact_id,days_ago,type,subject\ncon_1,x,call,Hi\n"))
    with pytest.raises(MockDataError, match="Cannot read mock data file"):
        load_events(tmp_path / "does-not-exist")


def test_blank_lines_whitespace_and_bom_are_tolerated(tmp_path):
    folder = write_data(tmp_path / "ok", events="\ufefftitle,day_offset,start,end\n\n  Sync , 1 , 09:00 , 09:30 \n\n")
    (event,) = load_events(folder)
    assert (event.title, event.day_offset, event.start) == ("Sync", 1, time(9, 0))


def test_settings_reads_mock_data_dir_from_env(monkeypatch, tmp_path):
    from config import DEFAULT_MOCK_DATA_DIR, Settings

    monkeypatch.delenv("MOCK_DATA_DIR", raising=False)
    assert Settings.from_env(env_file=tmp_path / "none.env").mock_data_dir == DEFAULT_MOCK_DATA_DIR
    monkeypatch.setenv("MOCK_DATA_DIR", str(tmp_path / "x"))
    assert Settings.from_env(env_file=tmp_path / "none.env").mock_data_dir == tmp_path / "x"


# ------------------------------------------------------------------------------ persistence
import shutil  # noqa: E402

from tests.fakes import Script, reissue_confirmed, tc  # noqa: E402


@pytest.fixture
def data_dir(tmp_path):
    folder = tmp_path / "data"
    shutil.copytree(TEST_DATA, folder)
    return folder


def persistent(settings, clock, folder):
    return Backend(settings.model_copy(update={"mock_data_dir": folder, "mock_persist": True}), clock=clock)


def csv_titles(folder):
    return [e.title for e in load_events(folder)]


def test_persistence_is_off_unless_enabled(settings, clock, data_dir):
    before = (data_dir / "events.csv").read_text()
    b = Backend(settings.model_copy(update={"mock_data_dir": data_dir}), clock=clock)  # mock_persist defaults to False
    b.calendar.create_event("Scratch", NOW.replace(hour=18), NOW.replace(hour=19), [], None, None)
    assert (data_dir / "events.csv").read_text() == before


def test_created_event_is_written_to_the_csv_and_survives_a_restart(settings, clock, data_dir):
    b = persistent(settings, clock, data_dir)
    res = b.calendar.create_event(
        "interview with client", NOW.replace(hour=12) + timedelta(days=1),
        NOW.replace(hour=12, minute=30) + timedelta(days=1),
        ["rahul.verma@globex.com"], "Room 1", "Final round",
    )
    event_id = res["event"]["id"]
    assert event_id == "evt_007"
    rows = {e.id: e for e in load_events(data_dir)}
    saved = rows[event_id]
    assert (saved.title, saved.date.isoformat(), saved.start, saved.end) == ("interview with client", "2026-10-22", time(12, 0), time(12, 30))
    assert saved.attendees == ("rahul.verma@globex.com",) and saved.location == "Room 1" and saved.day_offset is None

    again = persistent(settings, clock, data_dir)  # "restart"
    event = again.calendar.get(event_id)
    assert event.title == "interview with client" and event.start.isoformat() == tomorrow(12)
    assert again.calendar.create_event("Next", NOW.replace(hour=18), NOW.replace(hour=19), [], None, None)["event"]["id"] == "evt_008"


def test_untouched_seed_rows_stay_relative_and_moved_ones_become_absolute(settings, clock, data_dir):
    b = persistent(settings, clock, data_dir)
    b.calendar.update_event("evt_002", {"title": "Daily sync"})  # title only: still relative
    b.calendar.update_event("evt_003", {"start": NOW.replace(hour=17) + timedelta(days=1),
                                        "end": NOW.replace(hour=18) + timedelta(days=1)})
    rows = {e.id: e for e in load_events(data_dir)}
    assert rows["evt_002"].title == "Daily sync" and rows["evt_002"].day_offset == 1 and rows["evt_002"].date is None
    assert rows["evt_003"].date.isoformat() == "2026-10-22" and rows["evt_003"].start == time(17, 0)
    assert rows["evt_001"].day_offset == 0 and rows["evt_006"].day_offset == 3  # untouched


def test_delete_removes_the_row(settings, clock, data_dir):
    b = persistent(settings, clock, data_dir)
    b.calendar.delete_event("evt_002")
    assert "Team standup" not in csv_titles(data_dir) and len(csv_titles(data_dir)) == 5
    assert "Team standup" not in [e.title for e in persistent(settings, clock, data_dir).calendar.all_events()]


def test_failed_operations_do_not_touch_the_file(settings, clock, data_dir):
    b = persistent(settings, clock, data_dir)
    before = (data_dir / "events.csv").read_text()
    with pytest.raises(Exception):  # conflicts with the standup
        b.calendar.create_event("Clash", NOW.replace(hour=10) + timedelta(days=1),
                                NOW.replace(hour=11) + timedelta(days=1), [], None, None)
    with pytest.raises(Exception):
        b.calendar.delete_event("evt_999")
    b.fail_next("delete_calendar_event", "SERVER_ERROR")
    with pytest.raises(Exception):
        b.calendar.delete_event("evt_002")
    assert (data_dir / "events.csv").read_text() == before


def test_events_ending_on_a_later_day_round_trip(settings, clock, data_dir):
    b = persistent(settings, clock, data_dir)
    start = NOW.replace(hour=22) + timedelta(days=5)
    b.calendar.create_event("Night deploy", start, start + timedelta(hours=4), [], None, None)
    saved = next(e for e in load_events(data_dir) if e.title == "Night deploy")
    assert saved.end_date.isoformat() == "2026-10-27" and saved.end == time(2, 0)
    event = next(e for e in persistent(settings, clock, data_dir).calendar.all_events() if e.title == "Night deploy")
    assert (event.end - event.start) == timedelta(hours=4)


def test_unwritable_csv_does_not_fail_the_tool(settings, clock, data_dir, caplog):
    b = persistent(settings, clock, data_dir)
    (data_dir / "events.csv").unlink()
    (data_dir / "events.csv").mkdir()  # a directory where the file should be: os.replace fails
    res = b.calendar.delete_event("evt_002")
    assert res["status"] == "deleted" and b.calendar.get("evt_002") is None  # memory is still right


def test_special_characters_survive_csv_round_trip(settings, clock, data_dir):
    b = persistent(settings, clock, data_dir)
    start = NOW.replace(hour=20) + timedelta(days=2)
    b.calendar.create_event('Q&A, "live" session', start, start + timedelta(hours=1), ["a@x.com", "b@x.com"],
                            "Hall, 2nd floor", "line one\nline two")
    event = next(e for e in persistent(settings, clock, data_dir).calendar.all_events() if e.title.startswith("Q&A"))
    assert event.title == 'Q&A, "live" session' and event.attendees == ["a@x.com", "b@x.com"]
    assert event.location == "Hall, 2nd floor" and event.description == "line one\nline two"


def test_agent_booking_is_saved_to_csv_end_to_end(make_agent, settings, clock, data_dir):
    backend = persistent(settings, clock, data_dir)
    create = dict(title="interview with client", start=tomorrow(12), end=tomorrow(12, 30), attendees=["rahul.verma@globex.com"])
    script = Script([tc("create_calendar_event", **create), "Please confirm.", lambda c: reissue_confirmed(c), "Booked."])
    agent, _ = make_agent(script, backend=backend)
    agent.invoke("add a meeting with rahul.verma@globex.com at 12 pm tomorrow on my calendar")
    assert "interview with client" not in csv_titles(data_dir)  # nothing is written before the user confirms
    agent.invoke("yes")
    saved = next(e for e in load_events(data_dir) if e.title == "interview with client")
    assert saved.date.isoformat() == "2026-10-22" and saved.start == time(12, 0) and saved.end == time(12, 30)


@pytest.mark.parametrize(
    "row,message",
    [
        ("x,,,10:00,11:00,,,,", "give exactly one of 'date' or 'day_offset'"),
        ("x,2026-10-22,1,10:00,11:00,,,,", "give exactly one of 'date' or 'day_offset'"),
        ("x,10/22/2026,,10:00,11:00,,,,", "'date' must be YYYY-MM-DD"),
        ("x,2026-10-22,,10:00,09:00,,,,", "'end' must be after 'start'"),
    ],
)
def test_event_date_validation(tmp_path, row, message):
    folder = write_data(tmp_path / "v", events="title,date,day_offset,start,end,end_date,attendees,location,description\n" + row + "\n")
    with pytest.raises(MockDataError, match=message):
        load_events(folder)


def test_event_id_validation(tmp_path):
    header = "id,title,day_offset,start,end\n"
    with pytest.raises(MockDataError, match="'id' must look like evt_001"):
        load_events(write_data(tmp_path / "a", events=header + "e1,X,1,10:00,11:00\n"))
    with pytest.raises(MockDataError, match="duplicate event id evt_001"):
        load_events(write_data(tmp_path / "b", events=header + "evt_001,X,1,10:00,11:00\nevt_001,Y,1,12:00,13:00\n"))


def test_settings_reads_mock_persist_from_env(monkeypatch, tmp_path):
    from config import Settings

    assert Settings().mock_persist is False  # library default: tests and embedders never write files
    monkeypatch.delenv("MOCK_PERSIST", raising=False)
    assert Settings.from_env(env_file=tmp_path / "none.env").mock_persist is True  # the app default
    monkeypatch.setenv("MOCK_PERSIST", "false")
    assert Settings.from_env(env_file=tmp_path / "none.env").mock_persist is False
