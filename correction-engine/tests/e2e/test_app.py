#!/usr/bin/env python3
"""GTK4 app driving a precise in-place edit for the AT-SPI2 e2e test.

Ordering is load-bearing. GTK4's AT-SPI bridge only emits accessibility events
for a toplevel if an assistive-technology client was already listening when that
toplevel is realized/mapped. If the app maps its window first and the AT client
connects afterwards, GTK emits *nothing*. So the app must not create its window
until the Rust test has connected and subscribed.

Handshake with the Rust test (see `e2e_tests.rs` / `entrypoint.sh`), via marker
files under /tmp:

  1. Rust test connects to AT-SPI2, subscribes, then writes /tmp/e2e-subscribed.
  2. This app waits for /tmp/e2e-subscribed, then maps a window, sets the entry
     to "the cash is here", and writes /tmp/e2e-ready.
  3. Rust test waits for /tmp/e2e-ready, then writes /tmp/e2e-do-edit.
  4. This app turns "cash" into "cache" in place (GtkEditable delete_text +
     insert_text), then writes /tmp/e2e-edit-done.
  5. The app keeps running so the AT-SPI2 events flush; the Rust test controls
     overall timing / teardown.
"""

import os

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

SUBSCRIBED_MARKER = "/tmp/e2e-subscribed"
READY_MARKER = "/tmp/e2e-ready"
TRIGGER_MARKER = "/tmp/e2e-do-edit"
DONE_MARKER = "/tmp/e2e-edit-done"


class TestApp(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="com.test.CorrectionE2E")
        self._entry = None
        self._edited = False

    def do_activate(self):
        # Hold the application alive without a window until the AT client is
        # subscribed. (An explicit hold is released once we map the window.)
        self.hold()
        with open("/tmp/test-app.pid", "w") as f:
            f.write(str(os.getpid()))
        GLib.timeout_add(100, self._poll_subscribed)

    def _poll_subscribed(self):
        if not os.path.exists(SUBSCRIBED_MARKER):
            return True  # keep waiting

        # AT client is listening — now it is safe to map the window so GTK's
        # AT-SPI bridge activates and emits events for our edits.
        window = Gtk.ApplicationWindow(application=self, title="Test Window")
        entry = Gtk.Entry()
        entry.set_text("the cash is here")
        window.set_child(entry)
        window.present()
        self._entry = entry
        self.release()  # the window now keeps the app alive

        with open(READY_MARKER, "w") as f:
            f.write("ready\n")

        GLib.timeout_add(100, self._poll_trigger)
        return False  # stop this poll

    def _poll_trigger(self):
        if self._edited:
            return True
        if not os.path.exists(TRIGGER_MARKER):
            return True  # keep polling

        self._edited = True

        # Precise in-place edit: "the cash is here" -> "the cache is here".
        editable = self._entry
        editable.delete_text(4, 8)          # remove "cash"
        # PyGObject binding: Gtk.Editable.insert_text(text, position).
        editable.insert_text("cache", 4)    # insert "cache" at position 4

        with open(DONE_MARKER, "w") as f:
            f.write("done\n")

        return True  # keep the app (and its a11y object) alive


if __name__ == "__main__":
    app = TestApp()
    app.run(None)
