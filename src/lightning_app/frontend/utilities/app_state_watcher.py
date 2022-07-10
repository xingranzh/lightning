"""The AppStateWatcher enables a Frontend to.

- subscribe to app state changes
- to access and change the app state.

This is particularly useful for the PanelFrontend but can be used by other Frontends too.
"""
from __future__ import annotations

import logging

import param

from lightning_app.frontend.utilities.app_state_comm import watch_app_state
from lightning_app.frontend.utilities.other import get_flow_state
from lightning_app.utilities.imports import requires
from lightning_app.utilities.state import AppState

_logger = logging.getLogger(__name__)


class AppStateWatcher(param.Parameterized):
    """The AppStateWatcher enables a Frontend to.

    - subscribe to app state changes
    - to access and change the app state.

    This is particularly useful for the PanelFrontend, but can be used by
    other Frontends too.

    Example:

    .. code-block:: python

        import param

        app = AppStateWatcher()

        app.state.counter = 1


        @param.depends(app.param.state, watch=True)
        def update(state):
            print(f"The counter was updated to {state.counter}")


        app.state.counter += 1

    This would print 'The counter was updated to 2'.

    The AppStateWatcher is build on top of Param which is a framework like dataclass, attrs and
    Pydantic which additionally provides powerful and unique features for building reactive apps.

    Please note the AppStateWatcher is a singleton, i.e. only one instance is instantiated
    """

    state: AppState = param.ClassSelector(
        class_=AppState, constant=True,
        doc="""
    The AppState holds the state of the app reduced to the scope of the Flow""",
    )

    def __new__(cls):
        # This makes the AppStateWatcher a *singleton*.
        # The AppStateWatcher is a singleton to minimize the number of requests etc..
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    @requires("param")
    def __init__(self):
        # Its critical to initialize only once
        # See https://github.com/holoviz/param/issues/643
        if not hasattr(self, "_initilized"):
            super().__init__(name="singleton")
            self._start_watching()
            self.param.state.allow_None=False
            self._initilized = True

        # The below was observed when using mocking during testing
        if not self.state:
            raise Exception(".state has not been set.")
        if not self.state._state:
            raise Exception(".state._state has not been set.")

    def _start_watching(self):
        watch_app_state(self._handle_state_changed)
        self._request_state()

    def _get_flow_state(self):
        return get_flow_state()

    def _request_state(self):
        with param.edit_constant(self):
            self.state = self._get_flow_state()
        _logger.debug("Request app state")

    def _handle_state_changed(self):
        _logger.debug("Handle app state changed")
        self._request_state()