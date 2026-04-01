#!/usr/bin/env python3
"""Minimal GTK4 app with a text entry for AT-SPI2 e2e testing."""

import os
import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk


class TestApp(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="com.test.CorrectionE2E")

    def do_activate(self):
        window = Gtk.ApplicationWindow(application=self, title="Test Window")
        entry = Gtk.Entry()
        entry.set_accessible_name("test-entry")
        window.set_child(entry)
        window.present()

        # Write PID so the test harness can manage this process
        with open("/tmp/test-app.pid", "w") as f:
            f.write(str(os.getpid()))


if __name__ == "__main__":
    app = TestApp()
    app.run(None)
